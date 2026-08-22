"""Contract conformance test — run against a LIVE service.

Verifies that a running instance honours docs/integration/graph_api_contract.md:
every required field present, correct types, sane values, all response branches
reachable, and latency inside the NFR. Intended to be run by consumers (the
fusion engine) in CI so a breaking change is caught immediately rather than at
demo time.

Unlike tests/test_api.py (synthetic fixtures, no data), this talks to a real
service over HTTP and needs no local checkout of the model.

Usage:
    python scripts/serve_api.py &                  # in another shell
    python scripts/contract_test.py
    python scripts/contract_test.py --url http://graphsage:8000 --max-latency 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

REQUIRED_TOP = [
    "transaction_id", "timestamp", "model_version", "stage",
    "relational_risk_score", "risk_level", "confidence",
    "input_transaction", "suspicious_subgraph", "metadata",
]
REQUIRED_SUBGRAPH = [
    "k_hop", "node_count", "edge_count", "nodes", "edges",
    "sink_account", "pattern", "pattern_confidence", "structural_evidence",
]
REQUIRED_NODE = [
    "account_id", "role", "node_risk_score", "in_degree", "out_degree",
    "first_seen_step", "last_seen_step", "fraud_count_received",
    "total_received_amount",
]
REQUIRED_EDGE = [
    "src", "dst", "amount", "step", "edge_attention_weight",
    "edge_features", "is_trigger_edge", "is_fraud_predicted",
]
VALID_LEVELS = {"LOW", "MEDIUM", "HIGH", "CRITICAL", "NOT_APPLICABLE"}
VALID_PATTERNS = {
    "HUB_AND_SPOKE", "SMURFING", "LAYERING", "ACCOUNT_TAKEOVER", "UNKNOWN",
}


class Results:
    def __init__(self) -> None:
        self.passed = 0
        self.failures: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        if ok:
            self.passed += 1
            print(f"  PASS  {label}")
        else:
            self.failures.append(f"{label}{' — ' + detail if detail else ''}")
            print(f"  FAIL  {label}{' — ' + detail if detail else ''}")
        return ok


def post(url: str, body: dict, timeout: float = 30.0) -> tuple[int, dict, float]:
    req = urllib.request.Request(
        f"{url}/api/graph/analyze",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload, code = json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        payload, code = json.loads(e.read()), e.code
    return code, payload, (time.time() - t0) * 1000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--max-latency", type=float, default=500.0, help="NFR1, ms")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    r = Results()

    print(f"Contract test against {url}\n")

    # --- health ---------------------------------------------------------
    print("health:")
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=30) as h:
            health = json.loads(h.read())
    except Exception as e:
        print(f"  FAIL  service unreachable — {e}")
        return 1
    for key in ("status", "model_version", "stage", "tuned_threshold", "risk_bands"):
        r.check(key in health, f"/health exposes '{key}'")
    bands = health.get("risk_bands", {})
    r.check(
        all(k in bands for k in ("medium", "high", "critical")),
        "risk_bands has medium/high/critical",
    )
    r.check(
        bands.get("medium", 1) <= bands.get("high", 0) <= bands.get("critical", 0),
        "risk bands are monotonically ordered",
        str(bands),
    )

    # --- scenarios ------------------------------------------------------
    try:
        with urllib.request.urlopen(f"{url}/api/graph/demo-transactions", timeout=30) as s:
            scenarios = json.loads(s.read())["scenarios"]
    except Exception as e:
        print(f"\n  FAIL  demo scenarios unavailable — {e}")
        return 1

    seen_levels: set[str] = set()
    saw_422 = False
    saw_subgraph = False

    for sc in scenarios:
        print(f"\nscenario: {sc['label']}")
        code, payload, ms = post(url, sc["request"])

        if code == 422:
            saw_422 = True
            for key in ("error", "message"):
                r.check(key in payload, f"422 body has '{key}'")
            r.check(payload.get("error") == "BadRequest", "422 error is BadRequest")
            continue
        if code == 404:
            r.check(payload.get("error") == "NotFound", "404 error is NotFound")
            continue
        if not r.check(code == 200, f"HTTP 200 (got {code})", str(payload)[:160]):
            continue

        r.check(ms < args.max_latency, f"latency {ms:.0f}ms < {args.max_latency:.0f}ms (NFR1)")
        for key in REQUIRED_TOP:
            r.check(key in payload, f"top-level '{key}' present")

        score, level = payload.get("relational_risk_score"), payload.get("risk_level")
        seen_levels.add(level)
        r.check(isinstance(score, (int, float)) and 0 <= score <= 1,
                "relational_risk_score in [0,1]", str(score))
        r.check(level in VALID_LEVELS, "risk_level is a valid enum", str(level))
        r.check(payload["transaction_id"] == sc["request"]["transaction_id"],
                "transaction_id echoed back")
        r.check("inference_latency_ms" in payload.get("metadata", {}),
                "metadata.inference_latency_ms present")

        sg = payload.get("suspicious_subgraph")
        if level == "NOT_APPLICABLE":
            r.check(sg is None, "NOT_APPLICABLE has null subgraph")
            r.check(score == 0.0, "NOT_APPLICABLE scores 0.0")
            continue
        if sg is None:
            continue

        saw_subgraph = True
        for key in REQUIRED_SUBGRAPH:
            r.check(key in sg, f"subgraph '{key}' present")
        r.check(sg.get("pattern") in VALID_PATTERNS, "pattern is a valid enum",
                str(sg.get("pattern")))
        r.check(sg.get("node_count") == len(sg.get("nodes", [])),
                "node_count matches len(nodes)")
        r.check(sg.get("edge_count") == len(sg.get("edges", [])),
                "edge_count matches len(edges)")
        r.check(sum(1 for e in sg.get("edges", []) if e.get("is_trigger_edge")) == 1,
                "exactly one trigger edge")
        accounts = {n["account_id"] for n in sg.get("nodes", [])}
        r.check(sg.get("sink_account") in accounts, "sink_account is among nodes")
        r.check(
            all(e["src"] in accounts and e["dst"] in accounts for e in sg.get("edges", [])),
            "every edge endpoint is a listed node",
        )
        if sg.get("nodes"):
            r.check(all(k in sg["nodes"][0] for k in REQUIRED_NODE),
                    "node objects carry all contract fields")
        if sg.get("edges"):
            r.check(all(k in sg["edges"][0] for k in REQUIRED_EDGE),
                    "edge objects carry all contract fields")

    # --- coverage -------------------------------------------------------
    print("\ncoverage:")
    r.check(saw_422, "a validation failure (422) was exercised")
    r.check(saw_subgraph, "at least one subgraph was returned")
    r.check("NOT_APPLICABLE" in seen_levels, "NOT_APPLICABLE branch exercised")
    r.check(len(seen_levels - {"NOT_APPLICABLE"}) >= 2,
            "at least two distinct risk levels seen", str(sorted(seen_levels)))

    print(f"\n{'=' * 60}")
    if r.failures:
        print(f"FAILED — {len(r.failures)} of {r.passed + len(r.failures)} checks")
        for f in r.failures:
            print(f"  - {f}")
        return 1
    print(f"PASSED — all {r.passed} checks green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
