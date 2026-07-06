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
from graphsage.inference.predictor import (
    MODEL_VERSION,
    STAGE_NAME,
    GraphPredictor,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Contract §5 placeholders (configs/model_config.yaml inference.risk_thresholds).
RISK_THRESHOLDS = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 0.9}


def score_to_risk_level(score: float) -> RiskLevel:
    if score >= RISK_THRESHOLDS["critical"]:
        return RiskLevel.CRITICAL
    if score >= RISK_THRESHOLDS["high"]:
        return RiskLevel.HIGH
    if score >= RISK_THRESHOLDS["medium"]:
        return RiskLevel.HIGH  # contract §5: 0.75-0.90 band is still HIGH
    if score >= RISK_THRESHOLDS["low"]:
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
            print("Loading graph + checkpoint (first start computes score cache)...")
            app.state.predictor = GraphPredictor(REPO_ROOT)
            print(
                f"Ready in {app.state.predictor.startup_seconds:.1f}s — "
                f"threshold={app.state.predictor.threshold:.4f}"
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
            "stage": STAGE_NAME,
            "graph_version": p.graph_version,
            "num_nodes": int(p.data.num_nodes),
            "num_edges": int(p.data.num_edges),
            "tuned_threshold": p.threshold,
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
                stage=STAGE_NAME,
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
            score_to_risk_level(score),
            result["confidence"],
            result["suspicious_subgraph"],
        )

    @app.get("/demo")
    def demo_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "demo.html")

    @app.get("/api/graph/demo-transactions")
    def demo_transactions() -> FileResponse:
        return FileResponse(REPO_ROOT / "data" / "demo" / "demo_transactions.json")

    return app


app = create_app()
