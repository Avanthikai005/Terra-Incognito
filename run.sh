#!/usr/bin/env bash
# Terra Incognita — one-command end-to-end run.
#   bash run.sh [--subset N]        then each training step uses --subset N
#   SUBSET=200 bash run.sh          same as above via env
#   SKIP_PREP=1 bash run.sh         skip data preparation (use existing cache)
set -euo pipefail
cd "$(dirname "$0")"

############ Configuration ############
SUBSET="${SUBSET:-${1:-0}}"              # 0 = use all cached patches
SEED="${SEED:-42}"
REPLAY_CAP="${REPLAY_CAP:-300}"          # replay buffer size per region
EPOCHS="${EPOCHS:-5}"                    # training epochs per task
REGIONS=("hurricane-michael" "palu" "santa-rosa-fire")
########################################

echo "== Terra Incognita: Continual Learning for Cross-Regional Disaster Response =="
echo "== regions: ${REGIONS[*]} | seed=${SEED} | replay_cap=${REPLAY_CAP} | epochs=${EPOCHS} =="

# --- environment ------------------------------------------------------------
if [ ! -x venv/bin/python ]; then
    echo ">> creating virtualenv (venv/)"
    python3 -m venv venv
fi
. venv/bin/activate
python - <<'PY'
import importlib, sys
missing = [m for m in ("torch", "torchvision", "numpy", "PIL", "matplotlib") if importlib.util.find_spec(m) is None]
if missing:
    print(">> installing dependencies:", missing)
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
PY

# --- data ---------------------------------------------------------------------
if [ "${SKIP_PREP:-0}" != "1" ]; then
    echo ">> preparing patches (real xBD if present, synthetic fallback otherwise)"
    python data/prepare_xbd.py --xbd-root data/xbd --subset "$SUBSET" --seed "$SEED"
fi

# --- train all regimes ---------------------------------------------------------
for MODE in naive joint cl; do
    EXTRA=()
    [ "$MODE" = "cl" ] && EXTRA=("--replay-cap" "$REPLAY_CAP")
    echo ">> training   : $MODE"
    python src/train.py --mode "$MODE" --seed "$SEED" --epochs "$EPOCHS" \
                        --regions "${REGIONS[@]}" --subset "$SUBSET" "${EXTRA[@]}"
done

# --- evaluation + plots ---------------------------------------------------------
echo ">> evaluating : accuracy matrices + forgetting"
python src/evaluate.py
echo ">> ablation   : domain distance vs forgetting"
python src/domain_distance.py

echo "== DONE. Results in report/results.md, plots in plots/ =="