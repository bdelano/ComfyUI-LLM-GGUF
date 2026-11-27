# ComfyUI-LLM-GGUF

ComfyUI custom nodes for running any llama.cpp-supported GGUF chat model (Gemma 4,
Llama, Qwen, Mistral, Phi, ...)
language models inside ComfyUI, via `llama-cpp-python`. Text jobs share
ComfyUI's queue and VRAM management with image/audio jobs.

## Nodes

| Node | Purpose |
|---|---|
| **Load LLM GGUF** (`LLMGGUFLoader`) | Picks a `.gguf` from `ComfyUI/models/LLM/`; unloads ComfyUI's resident models first to make room. One LLM is cached at a time. |
| **LLM Generate Text** (`LLMGenerate`) | Chat completion using the GGUF's embedded chat template. Outputs a `STRING`, and (as an output node) puts the text in `/history/<prompt_id>` under `outputs.<node_id>.text`. |

The old `GemmaGGUFLoader` / `GemmaGenerate` names are still registered as
deprecated aliases, so workflows saved before the rename still load and run.
They're hidden from the node search menu. Rebuild those workflows with the
new nodes when convenient. Old and new nodes use different model types
(`GEMMA_MODEL` vs `LLM_GGUF_MODEL`), so they can't be wired to each other.

`LLMGenerate` details:
- Honors Interrupt / Clear Queue between tokens.
- Streams each token over ComfyUI's websocket as a `llm_gguf.chunk` event
  (`{node, chunk}`) for clients that want live text.
- `unload_after` (default on) frees the LLM's VRAM when done; the next run
  reloads it automatically even though ComfyUI caches the loader node.
- If the chat template has no system role, the system text is folded into the
  user turn.
- `strip_thinking` (default on) removes reasoning from the output text, leaving
  only the answer. It recognizes Gemma 4's thought channel
  (`<|channel>thought … <channel|>`), `<think>…</think>` (Qwen3, DeepSeek R1,
  and fine-tunes, including R1-style output that only has the closing tag),
  `<thinking>`, and `<reasoning>`. If a model opens a reasoning block and runs
  out of `max_tokens` before closing it, the node raises an error instead of
  returning half a thought, so raise `max_tokens`. Gemma 4 marks reasoning only with
  special tokens, which llama-cpp-python normally drops from the text, so the
  node renders special tokens for that call, then strips them. With
  `strip_thinking` off, a thinking Gemma 4 model's reasoning runs straight
  into the answer with no markers. The `llm_gguf.chunk` websocket events
  always carry the raw text, reasoning included.

## Install

1. Clone, link, or copy this folder into `ComfyUI/custom_nodes/`:
   ```
   ln -s /path/to/ComfyUI-LLM-GGUF ComfyUI/custom_nodes/
   ```
2. Install `llama-cpp-python` into **ComfyUI's** Python environment
   (`>=0.3.38` is required for the `gemma4` architecture):
   - NVIDIA (CUDA): see [Building with CUDA](#building-with-cuda) below.
   - macOS (Metal is on by default): `pip install "llama-cpp-python>=0.3.38"`
   - CPU only: `pip install -r requirements.txt`
3. Put a GGUF in `ComfyUI/models/LLM/` (e.g. `gemma-4-e4b-it-Q4_K_M.gguf`).
4. Restart ComfyUI.

## Building with CUDA

`llama-cpp-python` compiles llama.cpp from source. Pip shows only
`Building wheel for llama-cpp-python ...` until the build finishes. On macOS
that takes about 3–10 minutes. A CUDA build can take 10–30+ minutes if it
compiles for every GPU architecture. Add `-v` to watch progress.

1. Make sure the CUDA compiler exists. The runtime alone is not enough:
   ```
   /usr/local/cuda/bin/nvcc --version
   ```
   If it's missing, install it, e.g. `sudo apt install cuda-nvcc-13-2` (or
   `cuda-toolkit-13-2`) from NVIDIA's repo, matching your CUDA version.
2. Find your GPU's compute capability (e.g. `86` = RTX 30xx, `89` = RTX 40xx,
   `120` = RTX 50xx):
   ```
   nvidia-smi --query-gpu=compute_cap --format=csv
   ```
3. Build with **ComfyUI's** pip, with `nvcc` on `PATH`. Replace `89` with your
   value from step 2, without the dot:
   ```
   export PATH=/usr/local/cuda/bin:$PATH
   export CUDACXX=/usr/local/cuda/bin/nvcc
   CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89" \
     /path/to/ComfyUI/venv/bin/pip install -v "llama-cpp-python>=0.3.38" \
     --no-binary llama-cpp-python --no-cache-dir
   ```
   Add the two `export` lines to `~/.bashrc` to make them stick.

### Troubleshooting

| Error | Fix |
|---|---|
| `No CMAKE_CUDA_COMPILER could be found` / `The CUDA compiler identification is unknown` | `nvcc` isn't on `PATH`. Set `PATH` and `CUDACXX` as in step 3, or install `nvcc` (step 1). |
| `unsupported GNU version! gcc versions later than N are not supported` | Your GCC is newer than this CUDA supports. Either use an older host compiler (`sudo apt install g++-14`, then `export CUDAHOSTCXX=g++-14`) or add `-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler` to `CMAKE_ARGS`. |
| `cmake: not found` | `sudo apt install cmake` / `brew install cmake` / `pip install cmake` |
| macOS: compiler or SDK errors | `xcode-select --install` |
| Builds fine but ComfyUI says it isn't installed | You installed into a different Python. Use ComfyUI's own `venv/bin/pip` (or `python_embeded\python.exe -m pip` on Windows portable). |
| `Failed to create llama_context` | The weights loaded, but the KV cache didn't fit, which is usually out of VRAM. Fixes, cheapest first: leave `swa_full` **off** (llama.cpp turns it on by default, which allocates a full-length cache on every layer); set `kv_cache_type` to `q8_0` (needs `flash_attn`); lower `n_ctx`; lower `n_gpu_layers` to keep some layers on the CPU. A 31B model at Q6_K is about 25 GB of weights before any cache, so on a 24 GB card use a smaller quant (Q4_K_M) or partial offload. |
| Model fails to load with an unknown-architecture error | `llama-cpp-python` is older than 0.3.38. Rebuild with `--no-cache-dir` so pip doesn't reuse an old wheel. |

## Example workflows

There are two example files, in different formats:

| File | Format | Use it for |
|---|---|---|
| `example_workflow.json` | ComfyUI editor (UI) format | Drag onto the canvas or open from the Workflow menu. Loader → Generate → Preview Any, which shows the text on the canvas. |
| `example_workflow_api.json` | API format | POSTing to `/prompt` from scripts. **The editor can't open it.** It fails with `Loading aborted due to error reloading workflow data ... e is undefined`. |

In both, change the loader's `model` to match your GGUF's file name.

## Test via the API

```
curl -s http://127.0.0.1:8188/object_info/LLMGenerate
jq '{prompt: .}' example_workflow_api.json | curl -s -X POST -H 'Content-Type: application/json' -d @- http://127.0.0.1:8188/prompt
curl -s http://127.0.0.1:8188/history/<prompt_id> | jq '.[].outputs'
```

Edit `example_workflow_api.json`'s `model` to match your file name first.

## Out of scope (for now)

Vision/audio input (mmproj) and tool calling.
