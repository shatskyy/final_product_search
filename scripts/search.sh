#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$ROOT_DIR/.venv/bin/activate"

# Ensure outputs directory exists and build a unique filename
OUT_DIR="$ROOT_DIR/outputs"
mkdir -p "$OUT_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
if command -v uuidgen >/dev/null 2>&1; then
  ID="$(uuidgen | tr 'A-Z' 'a-z' | cut -c1-8)"
else
  TSN="$(date +%s%N)"; ID="${TSN: -8}"
fi
OUT_JSON="$OUT_DIR/search_${TS}_${ID}.json"

python "$ROOT_DIR/search_cli.py" "$@" --artifacts-dir "$ROOT_DIR" --top-n 10 --out-json "$OUT_JSON"
echo "Saved JSON: $OUT_JSON"
