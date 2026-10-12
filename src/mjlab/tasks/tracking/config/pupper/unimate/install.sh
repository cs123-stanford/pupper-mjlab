#!/usr/bin/env bash
# Install UniMate (text-to-motion) in its own Python 3.10 environment.
#
#   bash install.sh [DIR]      # default DIR: ./UniMate
#
# Clones UniMate at a tested commit into DIR, makes DIR/.venv with uv, and
# downloads the released checkpoint (about 1.2 GB) into DIR/outputs. Safe to
# rerun: finished steps are skipped. UniMate keeps its own environment because
# it pins its own torch and transformers; mjlab talks to it through worker.py.
set -euo pipefail

DIR=${1:-UniMate}
REPO=https://github.com/Friedrich-M/UniMate.git
COMMIT=b78c7807df7cf4231d4e67f8d0ece29c5e11025b
EXP=unimate_uniml3d_f60_v3
STEP=150000

command -v uv >/dev/null || pip install -q uv
if [ ! -d "$DIR/.git" ]; then
  git clone -q "$REPO" "$DIR"
fi
git -C "$DIR" checkout -q "$COMMIT"

export VIRTUAL_ENV="$DIR/.venv"
if [ ! -f "$DIR/.venv/.done" ]; then
  uv venv -q --allow-existing --python 3.10 "$DIR/.venv"
  uv pip install -q "setuptools<81"
  # cu128 wheels run on every current GPU, Blackwell (RTX 5090, RTX PRO 6000) included.
  uv pip install -q torch==2.9.0 --torch-backend cu128
  uv pip install -q numpy torch-geometric==2.7.0 torchdiffeq==0.2.5 einops==0.8.2 \
    accelerate==1.14.0 tensorboard transformers==5.16.1 sentencepiece==0.2.1 \
    "spacy>=3.8,<3.9" \
    "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" \
    num2words==0.5.14 huggingface_hub==1.29.0 scipy matplotlib imageio imageio-ffmpeg pillow tyro==1.0.12 rich loguru tqdm
  uv pip install -q --no-build-isolation "Motion @ git+https://github.com/inbar-2344/Motion.git"
  touch "$DIR/.venv/.done"
fi

CKPT="$DIR/outputs/$EXP/checkpoints/checkpoint_step_$STEP.pt"
if [ ! -f "$CKPT" ]; then
  "$DIR/.venv/bin/hf" download Linzhan/UniMate --repo-type model --local-dir "$DIR/outputs" \
    --include "$EXP/*.json" --include "$EXP/*.npy" \
    --include "$EXP/checkpoints/checkpoint_step_$STEP.pt" >/dev/null || true
  # hf can exit 0 after a failed download: check for the file itself.
  [ -f "$CKPT" ] || { echo "Checkpoint download failed (disk full? offline?): rerun this script." >&2; exit 1; }
fi
echo "UniMate ready: $DIR ($EXP, step $STEP)"
