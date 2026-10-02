#!/usr/bin/env bash
# GPU host only. New checkout/root, no existing artifact or environment mutation.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo 'Usage: checkpoint_r2_fresh_gpu_setup.sh NEW_ABSOLUTE_ROOT EXACT_PUSHED_COMMIT [HOURLY_PRICE]' >&2
    exit 2
fi
pilot_root=$1
pilot_commit=$2
if [[ $(uname -s) != Linux || $pilot_root != /* || -e $pilot_root ]]; then
    echo 'Requires Linux and an absolute root that does not exist.' >&2
    exit 2
fi
pilot_head=$(git rev-parse HEAD)
if [[ $pilot_head != "$pilot_commit" || -n $(git status --porcelain --untracked-files=all) ]]; then
    echo 'Requires a clean checkout of the exact pushed pilot commit.' >&2
    exit 2
fi
mkdir "$pilot_root"
date -u +%Y-%m-%dT%H:%M:%SZ > "$pilot_root/provision-started-utc.txt"
python3.12 -m venv "$pilot_root/venv"
"$pilot_root/venv/bin/python" -m pip install 'pip==25.2'
"$pilot_root/venv/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.8.0+cu128'
"$pilot_root/venv/bin/python" -m pip install 'transformers==4.56.2' 'tokenizers==0.22.1' 'numpy==2.1.3' \
    'safetensors==0.6.2' 'huggingface-hub==0.34.4'
"$pilot_root/venv/bin/python" -m pip freeze --all > "$pilot_root/installed-runtime.freeze.txt"
"$pilot_root/venv/bin/python" -m pip check > "$pilot_root/pip-check.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$pilot_root/provision-completed-utc.txt"
# HF_TOKEN may be supplied in the environment for gated Llama; never echo/store it.
# All Hugging Face downloads and incidental hub/Xet metadata use this fresh root.
export HF_HOME="$pilot_root/hf-home"
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
pilot_price_args=()
if [[ $# == 3 ]]; then
    pilot_price_args=(--hourly-price "$3")
fi
"$pilot_root/venv/bin/python" -B -m src.checkpoint_r2_fresh_pilot run \
    --output "$pilot_root/pilot" --cache "$pilot_root/model-cache" \
    --expected-commit "$pilot_commit" "${pilot_price_args[@]}"
