#!/usr/bin/env bash
# Launch the browser 3D demo.
#
#   bash scripts/run_demo.sh [port]
#
# Requires: runs/ppo_main/data/traces.json (produced by scripts/evaluate.py)
set -euo pipefail
PORT="${1:-8080}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "$ROOT/web/data/traces.json" ]; then
  if [ -f "$ROOT/runs/ppo_main/data/traces.json" ]; then
    mkdir -p "$ROOT/web/data"
    cp "$ROOT/runs/ppo_main/data/traces.json" "$ROOT/web/data/traces.json"
    echo "[demo] copied traces.json into web/data/"
  else
    echo "[demo] !! web/data/traces.json missing."
    echo "[demo]    run:  python scripts/evaluate.py --run runs/ppo_main"
    exit 1
  fi
fi

echo "[demo] serving $ROOT/web at http://localhost:$PORT"
cd "$ROOT/web"
python -m http.server "$PORT"
