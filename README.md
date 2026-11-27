# ComfyUI-LLM-GGUF

ComfyUI custom nodes for running Gemma 4 (or any llama.cpp-supported) GGUF
language models inside ComfyUI, via `llama-cpp-python`. Text jobs share
ComfyUI's queue and VRAM management with image/audio jobs.

## Nodes

| Node | Purpose |
|---|---|
| **Load Gemma GGUF** (`GemmaGGUFLoader`) | Picks a `.gguf` from `ComfyUI/models/LLM/`; unloads ComfyUI's resident models first to make room. One LLM is cached at a time. |
| **Gemma Generate Text** (`GemmaGenerate`) | Chat completion using the GGUF's embedded chat template. Outputs a `STRING`, and (as an output node) puts the text in `/history/<prompt_id>` under `outputs.<node_id>.text`. |

`GemmaGenerate` details:
- Honors Interrupt / Clear Queue between tokens.
- Streams each token over ComfyUI's websocket as a `llm_gguf.chunk` event
  (`{node, chunk}`) for clients that want live text.
- `unload_after` (default on) frees the LLM's VRAM when done; the next run
  reloads it automatically even though ComfyUI caches the loader node.
- If the chat template has no system role, the system text is folded into the
  user turn.

## Install

1. Clone, link, or copy this folder into `ComfyUI/custom_nodes/`:
   ```
   ln -s /path/to/ComfyUI-LLM-GGUF ComfyUI/custom_nodes/
   ```
2. Install `llama-cpp-python` into **ComfyUI's** Python environment
   (`>=0.3.38` is required for the `gemma4` architecture):
   - NVIDIA (CUDA):
     ```
     CMAKE_ARGS="-DGGML_CUDA=on" pip install "llama-cpp-python>=0.3.38" --no-binary llama-cpp-python
     ```
   - macOS (Metal is on by default): `pip install "llama-cpp-python>=0.3.38"`
   - CPU only: `pip install -r requirements.txt`
3. Put a GGUF in `ComfyUI/models/LLM/` (e.g. `gemma-4-e4b-it-Q4_K_M.gguf`).
4. Restart ComfyUI.

## Test via the API

```
curl -s http://127.0.0.1:8188/object_info/GemmaGenerate
jq '{prompt: .}' example_workflow_api.json | curl -s -X POST -H 'Content-Type: application/json' -d @- http://127.0.0.1:8188/prompt
curl -s http://127.0.0.1:8188/history/<prompt_id> | jq '.[].outputs'
```

Edit `example_workflow_api.json`'s `model` to match your file name first.

## Out of scope (for now)

Vision/audio input (mmproj), tool calling, and Gemma "thinking" output
filtering. A model that emits reasoning tokens will return them in the text.
