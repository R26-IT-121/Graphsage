#!/usr/bin/env bash
# Serve the network detector from this folder.
#
# Two things need saying, because neither is obvious:
#
#   PYTHONPATH — the `graphsage` package is pip-installed editable against the
#   component repo, so a bare `import graphsage` loads that copy no matter which
#   directory you start from. Putting this src first is what makes the code in
#   THIS folder the code that runs.
#
#   Port 8002 — scripts/serve_api.py defaults to 8000, but the fusion backend
#   looks for the network detector on 8002.
#
#   GRAPHSAGE_DATA_ROOT — the serving bundle is 162 MB and gitignored, so it
#   cannot live in this repo. This points at the one copy on disk.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${GRAPHSAGE_PYTHON:-$HOME/Documents/Graphsage/.venv/bin/python}"

if [ ! -x "$PY" ]; then
  echo "No interpreter at $PY — set GRAPHSAGE_PYTHON to one with torch installed." >&2
  exit 1
fi
DATA="${GRAPHSAGE_DATA_ROOT:-$HOME/Documents/Graphsage}"
if [ ! -f "$DATA/data/graph/serving_bundle.pt" ]; then
  echo "No serving_bundle.pt under $DATA/data/graph — set GRAPHSAGE_DATA_ROOT." >&2
  exit 1
fi
export GRAPHSAGE_DATA_ROOT="$DATA"

cd "$HERE"
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
export GRAPHSAGE_API_HOST="${GRAPHSAGE_API_HOST:-127.0.0.1}"
export GRAPHSAGE_API_PORT="${GRAPHSAGE_API_PORT:-8002}"

echo "network detector  →  http://$GRAPHSAGE_API_HOST:$GRAPHSAGE_API_PORT"
exec "$PY" -c "
import os, uvicorn
uvicorn.run('graphsage.api.app:app',
            host=os.environ['GRAPHSAGE_API_HOST'],
            port=int(os.environ['GRAPHSAGE_API_PORT']),
            log_level='warning')"
