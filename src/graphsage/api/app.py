"""FastAPI application — POST /api/graph/analyze per the Member 4 contract.

All heavy state (graph, checkpoint, cached scores) lives in a GraphPredictor
constructed once at startup; request handling is pure lookup + extraction,
which is what meets the p95 < 500 ms NFR. GET /demo serves the PP2 demo page.

Usage:
    python scripts/serve_api.py
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError

from graphsage.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorResponse,
    ResponseMetadata,
    RiskLevel,
)
from graphsage.inference.predictor import MODEL_VERSION, GraphPredictor

REPO_ROOT = Path(__file__).resolve().parents[3]
START_TS = time.time()
STATIC_DIR = Path(__file__).resolve().parent / "static"

def score_to_risk_level(score: float, bands: dict[str, float]) -> RiskLevel:
    """Map a calibrated score to a contract §5 risk level.

    Band edges come from the predictor, which derives them from the tuned
    decision threshold and the served score distribution — see
    GraphPredictor.risk_bands for why the config's fixed cutoffs don't apply
    to calibrated probabilities.
    """
    if score >= bands["critical"]:
        return RiskLevel.CRITICAL
    if score >= bands["high"]:
        return RiskLevel.HIGH
    if score >= bands["medium"]:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def create_app(predictor: GraphPredictor | None = None) -> FastAPI:
    """App factory. Tests inject a predictor built on a synthetic graph."""
    app = FastAPI(title="GraphSAGE Relational Fraud Detector", version=MODEL_VERSION)
    app.state.predictor = predictor

    # The DeepSentinel dashboard (Member 4) calls this service from another
    # origin; contract §1 declares an internal trusted network with no auth.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def load_predictor() -> None:
        if app.state.predictor is None:
            print("Loading serving bundle...")
            p = GraphPredictor(REPO_ROOT)
            app.state.predictor = p
            print(
                f"Ready in {p.startup_seconds:.1f}s — {p.stage}, "
                f"threshold={p.threshold:.4f}, bands={p.risk_bands}"
            )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first["loc"] if p != "body")
        body = exc.body if isinstance(exc.body, dict) else {}
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                transaction_id=body.get("transaction_id"),
                error="BadRequest",
                message=f"Field '{loc}': {first['msg']}",
            ).model_dump(),
        )

    @app.get("/health")
    def health() -> dict:
        p: GraphPredictor = app.state.predictor
        return {
            "status": "ok",
            "model_version": MODEL_VERSION,
            "stage": p.stage,
            "graph_version": p.graph_version,
            "num_nodes": int(p.data.num_nodes),
            "num_edges": int(p.data.num_edges),
            "tuned_threshold": p.threshold,
            "risk_bands": p.risk_bands,
            "model_meta": p.meta,
        }

    @app.post("/api/graph/analyze", response_model=AnalyzeResponse)
    def analyze(req: AnalyzeRequest):
        t0 = time.time()
        p: GraphPredictor = app.state.predictor

        def respond(score: float, level: RiskLevel, confidence: float, subgraph):
            return AnalyzeResponse(
                transaction_id=req.transaction_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                model_version=MODEL_VERSION,
                stage=p.stage,
                relational_risk_score=score,
                risk_level=level,
                confidence=confidence,
                input_transaction=req,
                suspicious_subgraph=subgraph,
                metadata=ResponseMetadata(
                    inference_latency_ms=int((time.time() - t0) * 1000),
                    graph_version=p.graph_version,
                ),
            )

        # Contract §2 note: no fraud exists outside TRANSFER/CASH_OUT.
        if not p.is_applicable(req.type.value):
            return respond(0.0, RiskLevel.NOT_APPLICABLE, 1.0, None)

        result = p.analyze(req.nameOrig, req.nameDest, req.step)
        if result is None:
            return JSONResponse(
                status_code=404,
                content=ErrorResponse(
                    transaction_id=req.transaction_id,
                    error="NotFound",
                    message=(
                        f"No {req.nameOrig} -> {req.nameDest} edge in the graph; "
                        "cannot anchor a subgraph"
                    ),
                ).model_dump(),
            )

        score = result["relational_risk_score"]
        return respond(
            score,
            score_to_risk_level(score, p.risk_bands),
            result["confidence"],
            result["suspicious_subgraph"],
        )

    @app.get("/demo")
    def demo_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "demo.html")

    @app.get("/api/graph/runtime")
    def runtime() -> dict:
        """Is the trained network loaded and serving right now?

        Exposed so an operator can see the model as a running thing — uptime,
        forward passes performed, mean latency — rather than taking it on
        trust.
        """
        p: GraphPredictor = app.state.predictor
        live = getattr(p, "live", None)
        return {
            "service_uptime_seconds": round(time.time() - START_TS, 1),
            "serving_mode": "live_inference" if live else "precomputed_only",
            "precomputed": {
                "accounts": int(p.data.num_nodes),
                "transactions": int(p.data.num_edges),
            },
            "model": live.stats if live else {
                "loaded": False, "reason": getattr(p, "live_error", None),
            },
        }

    @app.get("/api/graph/neighbourhood")
    def neighbourhood(account: str, hops: int = 1, max_edges: int = 150):
        """The graph immediately around one account.

        For exploring rather than deciding: /analyze answers a question about a
        transaction, this answers "show me this account". Bounded on purpose —
        the served graph is 3.27M accounts, and no request here may try to hand
        a browser more than a screenful of it. The caller walks outward a node
        at a time instead, and the response reports whether it was truncated so
        the UI can say so rather than implying it drew everything.
        """
        from graphsage.extraction.subgraph import neighbourhood as build

        p: GraphPredictor = app.state.predictor
        out = build(
            p.extractor, account, p.probs, p.edge_attention,
            hops=hops, max_edges=max(10, min(int(max_edges), 400)),
        )
        if out is None:
            return JSONResponse(
                status_code=404,
                content={"error": "NotFound",
                         "message": f"No account {account!r} in the graph."},
            )
        return out

    @app.get("/api/graph/sample-transactions")
    def sample_transactions(n: int = 20, fraud_ratio: float = 0.08) -> dict:
        """Real transactions for a live monitor to replay.

        These are genuine PaySim edges, so a downstream monitor screens real
        records through the real model rather than replaying a canned script.
        """
        p: GraphPredictor = app.state.predictor
        n = max(1, min(int(n), 200))
        ratio = max(0.0, min(float(fraud_ratio), 1.0))
        return {"transactions": p.sample_transactions(n, ratio)}

    @app.get("/api/graph/demo-transactions")
    def demo_transactions() -> FileResponse:
        return FileResponse(REPO_ROOT / "data" / "demo" / "demo_transactions.json")

    return app


app = create_app()
