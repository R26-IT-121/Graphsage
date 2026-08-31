"""FastAPI application — POST /api/graph/analyze per the Member 4 contract.

All heavy state (graph, checkpoint, cached scores) lives in a GraphPredictor
constructed once at startup; request handling is pure lookup + extraction,
which is what meets the p95 < 500 ms NFR. GET /demo serves the PP2 demo page.

Usage:
    python scripts/serve_api.py
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
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

# Where data/ lives. Defaults to the checkout this file sits in, which is
# right for a single working copy. It is overridable because the serving
# bundle is 162 MB and gitignored: a second checkout of this code has the
# source but no model, and pointing it at the one copy on disk beats
# either duplicating the file or committing a machine-specific symlink.
REPO_ROOT = Path(os.getenv("GRAPHSAGE_DATA_ROOT")
                 or Path(__file__).resolve().parents[3])
START_TS = time.time()
STATIC_DIR = Path(__file__).resolve().parent / "static"

# ── Demo CSV validation ──────────────────────────────────────────────────────
# The demo panel takes a file from whoever is standing in front of it, so the
# answer to a file that is not transactions has to be "this is not a
# transactions file", not a table of blank rows. Without these checks an
# unrelated CSV parsed fine, every lookup missed, and every row came back
# unscored — which reads as the model failing rather than the input being wrong.
#
# The column names mirror backend/batch.py so the two uploads accept and reject
# the same files. Balances are not required: the feature builder defaults them
# to zero, and a file that lacks them still scores.
DEMO_REQUIRED = ("nameorig", "namedest", "amount", "type", "step")
DEMO_VALID_TYPES = {"TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"}
DEMO_MAX_BYTES = 2 * 1024 * 1024
DEMO_MAX_ROWS = 500
_DEMO_CANONICAL = {
    "nameorig": "nameOrig", "namedest": "nameDest", "amount": "amount",
    "type": "type", "step": "step", "oldbalanceorg": "oldbalanceOrg",
    "newbalanceorig": "newbalanceOrig", "oldbalancedest": "oldbalanceDest",
    "newbalancedest": "newbalanceDest", "isfraud": "isFraud",
}


class DemoCsvError(Exception):
    """A file the user needs to fix, with a message that says how."""

    def __init__(self, message: str, **detail):
        super().__init__(message)
        self.message = message
        self.detail = detail


def read_demo_csv(filename: str, data: bytes) -> tuple[list[dict], list[str]]:
    """Parse and check an uploaded demo file.

    Returns the rows, keyed by canonical column name, and any advisory notes.
    Raises DemoCsvError for anything the caller has to fix.
    """
    import csv as _csv
    import io

    name = (filename or "").lower()
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        raise DemoCsvError(
            "This panel reads .csv only. Save the sheet as CSV and upload that "
            "— or use Batch upload on the Fusion page, which reads .xlsx.")
    if name and not name.endswith((".csv", ".txt", ".tsv")):
        raise DemoCsvError("Upload a .csv file. Other formats cannot be read.")

    if len(data) > DEMO_MAX_BYTES:
        raise DemoCsvError(
            f"The file is {len(data) / 1024 / 1024:.1f} MB. The limit is "
            f"{DEMO_MAX_BYTES // 1024 // 1024} MB.")

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise DemoCsvError(
            "That file is not text. It looks like a spreadsheet or another "
            "binary format saved with a .csv name.") from None

    reader = _csv.DictReader(io.StringIO(text))
    header = [h for h in (reader.fieldnames or []) if str(h).strip()]
    if not header:
        raise DemoCsvError("That file has no header row.")

    index = {str(h).strip().lower(): str(h) for h in header}
    missing = [c for c in DEMO_REQUIRED if c not in index]
    if missing:
        raise DemoCsvError(
            "The file is missing required column(s): "
            + ", ".join(_DEMO_CANONICAL[m] for m in missing)
            + ". A transactions file needs step, type, amount, nameOrig and "
              "nameDest.",
            missing=[_DEMO_CANONICAL[m] for m in missing],
            found=header)

    rows, notes = [], []
    for offset, raw_row in enumerate(reader, start=2):   # row 1 is the header
        if not any(str(v).strip() for v in raw_row.values() if v is not None):
            continue                                     # blank line
        if len(rows) >= DEMO_MAX_ROWS:
            notes.append(f"Only the first {DEMO_MAX_ROWS} rows were scored.")
            break

        row = {}
        for lower, original in index.items():
            row[_DEMO_CANONICAL.get(lower, original)] = raw_row.get(original)

        tx_type = str(row.get("type") or "").strip().upper()
        if tx_type not in DEMO_VALID_TYPES:
            raise DemoCsvError(
                f"Row {offset}: '{tx_type or '(blank)'}' is not a transaction "
                f"type. Expected one of {', '.join(sorted(DEMO_VALID_TYPES))}.",
                row=offset)
        row["type"] = tx_type

        for field in ("amount", "step"):
            value = str(row.get(field) or "").strip()
            try:
                float(value)
            except ValueError:
                raise DemoCsvError(
                    f"Row {offset}: {field} is '{value or '(blank)'}', which is "
                    f"not a number.", row=offset) from None

        if not str(row.get("nameOrig") or "").strip() or \
                not str(row.get("nameDest") or "").strip():
            raise DemoCsvError(
                f"Row {offset}: a transaction with no sender or no recipient "
                f"cannot be scored.", row=offset)

        rows.append(row)

    if not rows:
        raise DemoCsvError("No rows in that file.")

    if "isFraud" not in index and "isfraud" not in index:
        notes.append("No isFraud column, so accuracy cannot be measured — "
                     "only what the model flags.")
    return rows, notes


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
    def neighbourhood(account: str, hops: int = 1, max_edges: int = 150,
                      scope: str = "component"):
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
            scope="component" if scope == "component" else "hops",
        )
        if out is None:
            return JSONResponse(
                status_code=404,
                content={"error": "NotFound",
                         "message": f"No account {account!r} in the graph."},
            )
        return out

    # ── Demo mode ────────────────────────────────────────────────────────────
    #
    # A separate surface from /analyze on purpose. /analyze answers about a
    # transaction between accounts the snapshot already contains; these two
    # answer about accounts it has never seen, which is the one thing this
    # architecture can do that a transductive model cannot.
    #
    # Nothing here writes to the graph. Every node exists for one request.

    @app.post("/api/graph/demo/score-account")
    def demo_score_account(body: dict):
        """Score an account that is not in the graph, from its transactions alone.

        The caller supplies transactions, never a feature vector — so there is
        no way to hand the network a number that did not come from a stated
        transaction. The twelve features are derived here by the same
        arithmetic the training set was built with (pinned by
        tests/test_inductive.py), and the embedding is aggregated from whoever
        the account is attached to.

        The score returned is the raw network probability. It is deliberately
        not dressed up as a calibrated one: the isotonic calibrator that
        produced the precomputed scores was never saved as a reusable artefact,
        so a calibrated number cannot honestly be produced on this path. What
        is comparable — and is what the demo turns on — is this account against
        its own neighbours scored the same way, which is returned alongside.
        """
        from graphsage.inference.inductive import (
            NODE_COLS, derive_node_features, edge_features)

        p: GraphPredictor = app.state.predictor
        live = getattr(p, "live", None)
        if live is None:
            return JSONResponse(status_code=503, content={
                "error": "ModelNotLoaded",
                "message": getattr(p, "live_error", "live inference unavailable"),
            })

        account = str(body.get("account") or "").strip()
        txns = body.get("transactions") or []
        if not account or not txns:
            return JSONResponse(status_code=422, content={
                "error": "Invalid",
                "message": "Supply 'account' and at least one transaction.",
            })

        out_txns = [t for t in txns if t.get("nameOrig") == account]
        in_txns = [t for t in txns if t.get("nameDest") == account]
        if not out_txns and not in_txns:
            return JSONResponse(status_code=422, content={
                "error": "Invalid",
                "message": f"No transaction names {account!r} as sender or receiver.",
            })

        horizon = int(p.meta["step_range"][1])
        feats = derive_node_features(out_txns, in_txns, horizon)

        # Resolve the counterparties. An account the graph has never heard of
        # cannot anchor an embedding, so say which ones were dropped rather
        # than quietly scoring against fewer neighbours than the caller named.
        name_to_id = p.extractor.name_to_id
        out_edges, in_edges, unknown = [], [], []
        for t in out_txns:
            nid = name_to_id.get(t.get("nameDest"))
            (out_edges.append((nid, edge_features(t))) if nid is not None
             else unknown.append(t.get("nameDest")))
        for t in in_txns:
            nid = name_to_id.get(t.get("nameOrig"))
            (in_edges.append((nid, edge_features(t))) if nid is not None
             else unknown.append(t.get("nameOrig")))
        if not out_edges and not in_edges:
            return JSONResponse(status_code=422, content={
                "error": "NoKnownNeighbours",
                "message": ("None of the counterparties are in the graph, so "
                            "there is no neighbourhood to aggregate from."),
                "unknown_accounts": unknown,
            })

        try:
            score, ms, prov = live.score_new_node(feats, out_edges, in_edges)
        except ValueError as exc:
            return JSONResponse(status_code=422,
                                content={"error": "Invalid", "message": str(exc)})

        # The comparison that carries the meaning: the same network, the same
        # raw output space, run over each neighbour this account attached to.
        neighbours = []
        for nid, _ in out_edges + in_edges:
            try:
                raw, _ = live.score_node(int(nid))
            except Exception:                            # noqa: BLE001
                continue
            neighbours.append({
                "account": str(p.node_names[int(nid)]),
                "raw_score": round(raw, 4),
                "precomputed_score": round(float(p.probs[int(nid)]), 4),
            })

        return {
            "account": account,
            "in_graph": account in name_to_id,
            "raw_score": round(score, 4),
            "score_space": "raw_network_output",
            "calibrated": False,
            "features": {
                name: round(float(v), 4)
                for name, v in zip(NODE_COLS, feats)
            },
            "neighbours": neighbours,
            "unknown_accounts": unknown,
            "provenance": {**prov, "inference_ms": round(ms, 1)},
            "model": {"stage": p.stage, "version": MODEL_VERSION},
        }

    @app.post("/api/graph/demo/score-csv")
    async def demo_score_csv(file: UploadFile = File(...)):
        """Score a CSV through the relational model and nothing else.

        Deliberately not the platform's batch endpoint: no fusion, no other
        detectors, no alerting. One model, one column of answers, so what is
        on screen is attributable to this component alone.
        """
        from graphsage.inference.inductive import derive_node_features, edge_features

        p: GraphPredictor = app.state.predictor
        live = getattr(p, "live", None)
        horizon = int(p.meta["step_range"][1])
        try:
            rows, notes = read_demo_csv(file.filename or "", await file.read())
        except DemoCsvError as exc:
            return JSONResponse(status_code=422, content={
                "error": "Invalid", "message": exc.message, **exc.detail})

        name_to_id = p.extractor.name_to_id
        out, counts = [], {"precomputed": 0, "inductive": 0, "unscored": 0}
        for i, r in enumerate(rows):
            dest, orig = r.get("nameDest"), r.get("nameOrig")
            rec = {"row": i + 1, "nameOrig": orig, "nameDest": dest,
                   "amount": r.get("amount"), "type": r.get("type")}
            dst_id = name_to_id.get(dest)
            if dst_id is not None:
                score = float(p.probs[int(dst_id)])
                rec.update(score=round(score, 4), source="precomputed",
                           risk_level=score_to_risk_level(score, p.risk_bands).value)
                counts["precomputed"] += 1
            else:
                # The interesting case, not a failure. The destination is new,
                # so there is no precomputed score to look up — but the sender
                # is usually known, and that is a neighbourhood to aggregate
                # from. This is the row where a transductive model would have
                # to return nothing at all.
                src_id = name_to_id.get(orig)
                scored = False
                if live is not None and src_id is not None:
                    try:
                        feats = derive_node_features([], [r], horizon)
                        raw, ms, prov = live.score_new_node(
                            feats, [], [(src_id, edge_features(r))])
                        rec.update(score=round(raw, 4), source="inductive",
                                   risk_level=None, score_space="raw_network_output",
                                   note=("scored from its neighbourhood — this "
                                         "account is not in the graph"),
                                   neighbourhood_accounts=prov["neighbourhood_accounts"])
                        counts["inductive"] += 1
                        scored = True
                    except Exception as exc:            # noqa: BLE001
                        rec["note"] = f"could not score inductively: {exc}"
                if not scored:
                    rec.update(score=None, source="unscored", risk_level=None)
                    rec.setdefault("note", "neither account is in the served graph")
                    counts["unscored"] += 1
            if r.get("isFraud") is not None:
                rec["isFraud"] = r.get("isFraud")
            out.append(rec)

        return {"rows": out, "counts": counts, "notes": notes,
                "scored_by": "graph_only",
                "model": {"stage": p.stage, "version": MODEL_VERSION},
                "bands": {k: round(float(v), 4) for k, v in p.risk_bands.items()}}

    @app.get("/api/graph/performance")
    def performance() -> dict:
        """How the served model scores on the held-out window.

        Computed here rather than read from a report, because the reports in
        reports/ cover stages 1 through 3a and the model being served is 3b —
        quoting one for the other is exactly the mislabelling this project has
        already had to correct once.

        Everything below comes from the bundle already in memory: the labels
        from `edge_isFraud`, the window from `edge_step`, the scores from the
        same calibrated vector the API answers with. Evaluated only on accounts
        that received money inside the test window, which is the population the
        model is asked about in production.

        Cached after the first call — it is a handful of vectorised passes over
        3.3M nodes, but there is no reason to repeat them.
        """
        import numpy as np
        import torch

        p: GraphPredictor = app.state.predictor
        cached = getattr(app.state, "_performance", None)
        if cached is not None:
            return cached

        lo, hi = 701, 743                       # the test label window
        d = p.data
        dst = d.edge_index[1]
        step = d.edge_step.to(torch.int32)
        fraud = d.edge_isFraud.to(torch.bool)

        in_window = (step >= lo) & (step <= hi)

        # A node is a mule if it received fraud; it is evaluated if it received
        # anything at all. Both restricted to the window.
        y = torch.zeros(int(d.num_nodes), dtype=torch.bool)
        y[dst[in_window & fraud]] = True
        evaluated = torch.zeros(int(d.num_nodes), dtype=torch.bool)
        evaluated[dst[in_window]] = True

        idx = evaluated.nonzero(as_tuple=True)[0]
        scores = p.probs[idx].float().numpy()
        labels = y[idx].numpy()

        thr = float(p.threshold)
        pred = scores >= thr
        tp = int((pred & labels).sum())
        fp = int((pred & ~labels).sum())
        fn = int((~pred & labels).sum())
        tn = int((~pred & ~labels).sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / max(1, len(labels))

        # AUC by rank, which needs no sklearn and no sorting of pairs.
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1)
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        auc = ((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
               if n_pos and n_neg else None)

        out = {
            "window": {"from_step": lo, "to_step": hi, "name": "held-out test"},
            "threshold": round(thr, 4),
            "evaluated_accounts": int(len(labels)),
            "actual_mules": n_pos,
            "metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "accuracy": round(accuracy, 4),
                "auroc": round(float(auc), 4) if auc is not None else None,
            },
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "note": ("Computed from the serving bundle on the held-out window, "
                     "for the model actually being served."),
        }
        app.state._performance = out
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
