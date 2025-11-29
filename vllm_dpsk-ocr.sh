# uv venv
# source .venv/bin/activate
# # Until v0.11.1 release, you need to install vLLM from nightly build
# uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match


#!/bin/bash
export $(grep -v '^#' .env | xargs)


## https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html#offline-ocr-tasks


# cd ~/shared-dsrs/ai-notebooks/vllm-ocr-server/
# bash vllm_dpsk-ocr.sh
# nvitop --monitor full --graphics

vllm serve deepseek-ai/DeepSeek-OCR \
    --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor --no-enable-prefix-caching --mm-processor-cache-gb 0 \
    --served-model-name "deepseek-ai/DeepSeek-OCR" \
    --download-dir /home/jovyan/shared-dsrs/ai-models/hub \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.50 \
    --port 8000 \
    --allow-credentials \
    --api_key "taco" \
    --uvicorn-log-level info
    
