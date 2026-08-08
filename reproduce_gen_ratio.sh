#!/usr/bin/env bash
# reproduce_gen_ratio.sh
#
# One-shot script to reproduce the prefill-vs-decode ratio analysis
# (gen_ratio_profiler) for both decoder-only and encoder-decoder models.
#
# Usage:
#   ./reproduce_gen_ratio.sh                       # quick CPU-friendly run (default)
#   ./reproduce_gen_ratio.sh --tier 3b --device cuda --dtype float16   # ~3B-param models, GPU
#   ./reproduce_gen_ratio.sh --tier full            # standard preset, larger L sweep
#   ./reproduce_gen_ratio.sh --device cuda --dtype float16 --tier full
#   ./reproduce_gen_ratio.sh --repo-dir /path/to/transformer_latency
#
# Tiers:
#   quick  - gpt2_medium / bart_large    (~350M / ~0.4B params, runs fine on CPU)
#   3b     - gpt3_2p7b / bart_large      (~2.6B / ~0.4B params; the decoder needs
#            ~10.5GB RAM or VRAM at float32, or ~5.2GB at --dtype float16/bfloat16
#            on a GPU. float16 on CPU is auto-upgraded to float32 by this repo,
#            so use --dtype bfloat16 if you want 2-byte weights without a GPU)
#   full   - llama_7b / t5_large         (~7B / ~800M params; needs a real GPU)
#
# What it does:
#   1. Clones the repo (unless --repo-dir points at an existing checkout, or
#      the script is already run from inside one).
#   2. Creates/reuses a virtualenv and installs requirements.txt.
#   3. Runs gen_ratio_profiler/run_and_plot.py for:
#        - decoder-only     (gpt2_medium / gpt3_2p7b / llama_7b depending on --tier)
#        - encoder-decoder  (bart_large / t5_large depending on --tier)
#   4. Prints where the CSVs and figures landed.
#
# All flags are optional; sane defaults reproduce the exact example figures
# shown in this project's write-up.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (override with flags below)
# ---------------------------------------------------------------------------
REPO_URL="https://github.com/JINU98/transformer_latency.git"
REPO_DIR=""
TIER="quick"                 # quick | full
DEVICE="auto"                # auto | cpu | cuda
DTYPE="float32"               # float32 | float16 | bfloat16
REPEATS="3"
WARMUPS="1"
TOKEN_SCENARIOS="1,10,20%,50%,100%"
DECODER_SHAPE=""              # empty => tier default
ENC_DEC_SHAPE=""              # empty => tier default
SEQ_LENS=""                   # empty => tier default (overrides --preset if set)
SKIP_SETUP="0"                # 1 => skip venv/pip install (assume env already ready)

usage() {
  grep '^#' "$0" | sed -e 's/^#//' -e 's/^ //'
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --tier) TIER="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --warmups) WARMUPS="$2"; shift 2 ;;
    --token-scenarios) TOKEN_SCENARIOS="$2"; shift 2 ;;
    --decoder-shape) DECODER_SHAPE="$2"; shift 2 ;;
    --enc-dec-shape) ENC_DEC_SHAPE="$2"; shift 2 ;;
    --seq-lens) SEQ_LENS="$2"; shift 2 ;;
    --skip-setup) SKIP_SETUP="1"; shift 1 ;;
    -h|--help) usage ;;
    *) echo "Unknown flag: $1"; usage ;;
  esac
done

case "$TIER" in
  quick)
    : "${DECODER_SHAPE:=gpt2_medium}"
    : "${ENC_DEC_SHAPE:=bart_large}"
    : "${SEQ_LENS:=128,256,512,1024}"
    ;;
  3b)
    : "${DECODER_SHAPE:=gpt3_2p7b}"
    : "${ENC_DEC_SHAPE:=bart_large}"
    : "${SEQ_LENS:=512,1024,2048}"
    ;;
  full)
    : "${DECODER_SHAPE:=llama_7b}"
    : "${ENC_DEC_SHAPE:=t5_large}"
    : "${SEQ_LENS:=512,1024,2048,4096}"
    ;;
  *)
    echo "Unknown --tier '$TIER' (expected quick|3b|full)"; exit 1 ;;
esac

echo "== gen_ratio_profiler reproduction =="
echo "tier=$TIER device=$DEVICE dtype=$DTYPE repeats=$REPEATS warmups=$WARMUPS"
echo "decoder shape=$DECODER_SHAPE  encoder_decoder shape=$ENC_DEC_SHAPE"
echo "seq-lens=$SEQ_LENS  token-scenarios=$TOKEN_SCENARIOS"
echo

# ---------------------------------------------------------------------------
# 1. Locate or clone the repo
# ---------------------------------------------------------------------------
if [[ -z "$REPO_DIR" ]]; then
  if [[ -f "./gen_ratio_profiler/run_and_plot.py" ]]; then
    REPO_DIR="$(pwd)"
  elif [[ -f "../gen_ratio_profiler/run_and_plot.py" ]]; then
    REPO_DIR="$(cd .. && pwd)"
  else
    REPO_DIR="$(pwd)/transformer_latency"
  fi
fi

if [[ ! -d "$REPO_DIR/.git" && ! -f "$REPO_DIR/gen_ratio_profiler/run_and_plot.py" ]]; then
  echo "Cloning $REPO_URL into $REPO_DIR ..."
  git clone "$REPO_URL" "$REPO_DIR"
fi

if [[ ! -f "$REPO_DIR/gen_ratio_profiler/run_and_plot.py" ]]; then
  echo "ERROR: $REPO_DIR/gen_ratio_profiler/run_and_plot.py not found."
  echo "Copy the gen_ratio_profiler/ folder into $REPO_DIR first, or pass --repo-dir."
  exit 1
fi

cd "$REPO_DIR"
echo "Using repo at: $REPO_DIR"

# ---------------------------------------------------------------------------
# 2. Environment setup
# ---------------------------------------------------------------------------
if [[ "$SKIP_SETUP" != "1" ]]; then
  if [[ ! -d ".venv" ]]; then
    echo "Creating virtualenv (.venv) ..."
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "Installing requirements.txt ..."
  pip install --upgrade pip >/dev/null
  pip install -r requirements.txt
else
  echo "Skipping venv/pip setup (--skip-setup)."
fi

# ---------------------------------------------------------------------------
# 3. Run both architectures
# ---------------------------------------------------------------------------
cd gen_ratio_profiler

echo
echo "== Decoder-only: $DECODER_SHAPE =="
python run_and_plot.py \
  --architecture decoder \
  --shape-name "$DECODER_SHAPE" \
  --seq-lens "$SEQ_LENS" \
  --token-scenarios "$TOKEN_SCENARIOS" \
  --repeats "$REPEATS" \
  --warmups "$WARMUPS" \
  --device "$DEVICE" \
  --dtype "$DTYPE"

echo
echo "== Encoder-decoder: $ENC_DEC_SHAPE =="
python run_and_plot.py \
  --architecture encoder_decoder \
  --shape-name "$ENC_DEC_SHAPE" \
  --seq-lens "$SEQ_LENS" \
  --token-scenarios "$TOKEN_SCENARIOS" \
  --repeats "$REPEATS" \
  --warmups "$WARMUPS" \
  --device "$DEVICE" \
  --dtype "$DTYPE"

cd ..

echo
echo "== Done =="
echo "CSVs:    $REPO_DIR/gen_ratio_profiler/latency_results/{decoder,encoder_decoder}/*/"
echo "Figures: $REPO_DIR/figures/gen_ratio/{decoder,encoder_decoder}/"
