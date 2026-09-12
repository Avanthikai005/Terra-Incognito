#!/usr/bin/env bash
# Terra Incognita — one-command end-to-end run.
#   bash run.sh                 full pipeline, both sequences, all modes
#   MINI=1 bash run.sh          quick demo: 1 epoch, 20 samples
#   SKIP_PREP=1 bash run.sh     skip data preparation (existing patch cache)
set -euo pipefail
cd "$(dirname "$0")"

############ Configuration ############
SUBSET="${SUBSET:-0}"                       # 0 = use all cached patches
SEED="${SEED:-42}"
REPLAY_CAP="${REPLAY_CAP:-80}"              # fixed replay buffer size (cl only)
EPOCHS="${EPOCHS:-5}"                       # training epochs per task
MINI="${MINI:-0}"
if [ "$MINI" = "1" ]; then
    EPOCHS=1
    SUBSET=20
fi
# adaptive replay (cl_adaptive)
ADAPT_THRESHOLD="${ADAPT_THRESHOLD:-0.10}"
ADAPT_CAP_HIGH="${ADAPT_CAP_HIGH:-300}"
ADAPT_CAP_LOW="${ADAPT_CAP_LOW:-80}"
ADAPT_RATIO_HIGH="${ADAPT_RATIO_HIGH:-0.5}"
ADAPT_RATIO_LOW="${ADAPT_RATIO_LOW:-0.3}"
# sequences from configs/task_sequences.json
SEQUENCES="${SEQUENCES:-similar_domain cross_disaster}"
########################################

echo "== Terra Incognita: Disaster-type-Sequential CL on xBD =="
echo "== sequences: ${SEQUENCES} | mini=${MINI} | epochs=${EPOCHS} | subset=${SUBSET} =="

# --- environment ------------------------------------------------------------
if [ ! -x venv/bin/python ]; then
    echo ">> creating virtualenv (venv/)"
    python3 -m venv venv
fi
. venv/bin/activate
python - <<'PY'
import importlib.util, sys
missing = [m for m in ("torch", "torchvision", "numpy", "PIL", "matplotlib")
           if importlib.util.find_spec(m) is None]
if missing:
    print(">> installing dependencies:", missing)
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
PY

# --- data --------------------------------------------------------------------
if [ "${SKIP_PREP:-0}" != "1" ]; then
    echo ">> preparing patches: similar_domain (hurricane A/B/C) + cross_disaster (hurricane, palu)"
    python data/prepare_selected.py --xbd-root data --seed "$SEED" --build all
fi

# --- domain distance (needed by cl_adaptive) ---------------------------------
for SEQ in $SEQUENCES; do
    echo ">> domain distance  : $SEQ"
    python -m src.domain_distance --sequence "$SEQ"
done

# --- train: every mode x every sequence --------------------------------------
for SEQ in $SEQUENCES; do
    for MODE in baseline naive joint cl cl_adaptive; do
        EXTRA=()
        if [ "$MODE" = "cl" ]; then
            EXTRA=("--replay-cap" "$REPLAY_CAP")
        elif [ "$MODE" = "cl_adaptive" ]; then
            EXTRA=("--adapt-threshold" "$ADAPT_THRESHOLD"
                   "--adapt-cap-high"  "$ADAPT_CAP_HIGH"
                   "--adapt-cap-low"   "$ADAPT_CAP_LOW"
                   "--adapt-ratio-high" "$ADAPT_RATIO_HIGH"
                   "--adapt-ratio-low"  "$ADAPT_RATIO_LOW")
        fi
        echo ">> training   : $SEQ / $MODE"
        python -m src.train --mode "$MODE" --sequence "$SEQ" --seed "$SEED" \
                            --epochs "$EPOCHS" --subset "$SUBSET" "${EXTRA[@]}"
    done
done

# --- evaluation + plots -------------------------------------------------------
echo ">> evaluating : all sequences + benchmark table"
python -m src.evaluate
echo ">> ablation   : domain distance plots (both sequences)"
python -m src.domain_distance --sequence all

echo "== DONE. report/benchmark_results.md, plots/ =="