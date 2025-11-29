# vllm-ocr-server

This environment sets up a vLLM server for DeepSeek-OCR and allows calling it using the OpenAI-compatible API.



## 1. venv

Based on [DeepSeek-OCR Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html)

```
uv venv && \
source .venv/bin/activate && \
uv sync
```


## 3. Install the kernel for Jupyter

To use in jup notebooks

```
python -m ipykernel install --user --name vllm-ocr-server --display-name "vllm-ocr-server"
```




## 5. Run the vLLM server


Then run it:

```
bash vllm_dpsk-ocr.sh
```

This should start a server exposing the OpenAI-compatible API at the configured host/port.


## 6. Use the notebook to call the model

...

