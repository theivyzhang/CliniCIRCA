#!/usr/bin/env bash
# Run all five steps on the synthetic sample patient.
#
#   bash run_pipeline.sh open_source   [agent_name]   # local GPUs with vLLM
#   bash run_pipeline.sh vertex_gemini [agent_name]   # Gemini on Vertex AI
#
# Agent names are the keys in each stage's agent_specs.json.
# Outputs go to outputs/<backend>/<agent_name>/<step>/.
# Optional environment variables:
#   INPUT   input parquet (default: dataset/sample_patient.parquet)
#   PYTHON  python executable (default: python)
set -euo pipefail
cd "$(dirname "$0")"

# ---- Credentials: replace the placeholders, or export these in your shell ----
# Hugging Face token. Only needed for gated models (Llama, Gemma, MedGemma).
HF_TOKEN="${HF_TOKEN:-PLACEHOLDER: ENTER YOUR KEY}"
# Vertex AI: path to your service-account key file (.json). Leave the placeholder
# if you use `gcloud auth application-default login` instead.
GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-PLACEHOLDER: ENTER YOUR KEY}"
# Vertex AI: your Google Cloud project ID.
GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-PLACEHOLDER: ENTER YOUR PROJECT ID}"

# Pass on only the values that were filled in.
for var in HF_TOKEN GOOGLE_APPLICATION_CREDENTIALS GOOGLE_CLOUD_PROJECT; do
  if [[ "${!var}" == PLACEHOLDER* ]]; then unset "$var"; else export "$var"; fi
done

BACKEND="${1:-}"
case "$BACKEND" in
  open_source)   AGENT="${2:-qwen3_30b_a3b_instruct_2507}"; SUFFIX="" ;;
  vertex_gemini) AGENT="${2:-gemini_2_5_pro}";              SUFFIX="_gemini" ;;
  *) echo "usage: bash run_pipeline.sh <open_source|vertex_gemini> [agent_name]" >&2; exit 1 ;;
esac

PYTHON="${PYTHON:-python}"
INPUT="${INPUT:-dataset/sample_patient.parquet}"
OUT="outputs/$BACKEND/$AGENT"

echo "== Stage 1: event extraction"
"$PYTHON" pipeline/stage1_event_extraction/$BACKEND/event_extraction$SUFFIX.py \
    --input_parquet "$INPUT" --output_dir "$OUT/stage1" --agent_name "$AGENT"

echo "== Stage 1 postprocessing"
"$PYTHON" pipeline/stage1_postprocessing/run_stage1_postprocessing.py \
    --input_dir "$OUT/stage1" --output_dir "$OUT/stage1_postprocessing"

echo "== Stage 2: time tagging"
"$PYTHON" pipeline/stage2_time_tagging/$BACKEND/time_tagging$SUFFIX.py \
    --input_dir "$OUT/stage1_postprocessing" --output_dir "$OUT/stage2" --agent_name "$AGENT"

echo "== Stage 2 postprocessing"
"$PYTHON" pipeline/stage2_postprocessing/run_stage2_postprocessing.py \
    --input_dir "$OUT/stage2" --output_dir "$OUT/stage2_postprocessing"

echo "== Stage 3: summarization"
"$PYTHON" pipeline/stage3_summarization/$BACKEND/summarization$SUFFIX.py \
    --input_dir "$OUT/stage2_postprocessing" --output_dir "$OUT/stage3" --agent_name "$AGENT"

echo "Done. Outputs in $OUT/"
