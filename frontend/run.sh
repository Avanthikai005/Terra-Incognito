#!/usr/bin/env bash
# Terra Incognita demo UI — single command launcher.
#   bash run.sh
# Installs deps (if needed), generates placeholder assets (if missing), pulls
# real pipeline results (when results/*_matrix.json exist), then starts Vite.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d node_modules ]; then
    echo ">> installing frontend dependencies (node_modules/)"
    npm install --no-audit --no-fund
fi

if [ ! -f public/assets/overview.png ]; then
    echo ">> generating placeholder demo assets"
    PY=python3
    if [ -x ../venv/bin/python ] && ../venv/bin/python -c "import PIL, matplotlib, numpy" >/dev/null 2>&1; then
        PY=../venv/bin/python
    fi
    "$PY" tools/make_assets.py
fi

if [ -f ../results/naive_matrix.json ]; then
    echo ">> exporting real results from ../results"
    npm run data
fi

exec npm run dev