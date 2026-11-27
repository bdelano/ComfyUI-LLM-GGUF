"""Gemma 4 GGUF text generation nodes for ComfyUI, backed by llama-cpp-python.

Loads a GGUF LLM from ComfyUI's ``models/LLM/`` folder and generates text with
the chat template embedded in the GGUF. The model shares ComfyUI's queue (so
text jobs wait behind image jobs instead of fighting them for VRAM) and honors
Interrupt / Clear Queue between tokens.
"""

import gc
import logging
import os

import folder_paths
import comfy.model_management as mm
import comfy.utils

try:
    from server import PromptServer
except Exception:  # running outside the ComfyUI server (e.g. a unit test)
    PromptServer = None

log = logging.getLogger("ComfyUI-LLM-GGUF")

LLM_FOLDER = "LLM"
GGUF_EXT = {".gguf"}

# Register models/LLM/ for .gguf files without clobbering an entry another
# node pack may already have created for the same folder name.
_default_llm_dir = os.path.join(folder_paths.models_dir, LLM_FOLDER)
if LLM_FOLDER in folder_paths.folder_names_and_paths:
    _paths, _exts = folder_paths.folder_names_and_paths[LLM_FOLDER]
    if _default_llm_dir not in _paths:
        _paths.append(_default_llm_dir)
    folder_paths.folder_names_and_paths[LLM_FOLDER] = (_paths, set(_exts) | GGUF_EXT)
else:
    folder_paths.folder_names_and_paths[LLM_FOLDER] = ([_default_llm_dir], set(GGUF_EXT))


class GemmaModel:
    """Handle passed between nodes (custom GEMMA_MODEL type).

    Holds only the load parameters, not the Llama object itself: ComfyUI caches
    the loader's output, so after ``unload_after`` frees VRAM the loader won't
    re-run on the next prompt. ``GemmaGenerate`` calls ``get()``, which reloads
    on demand.
    """

    def __init__(self, path, name, n_gpu_layers, n_ctx, flash_attn):
        self.path = path
        self.name = name
        self.key = (path, n_gpu_layers, n_ctx, flash_attn)
        self.kwargs = dict(n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, flash_attn=flash_attn)

    def get(self):
        if _cache["key"] == self.key and _cache["llm"] is not None:
            return _cache["llm"]

        llama_cpp = _import_llama()
        _free_cached()
        # Make room: evict whatever diffusion/TTS models ComfyUI has resident.
        mm.unload_all_models()
        mm.soft_empty_cache()
        try:
            llm = llama_cpp.Llama(model_path=self.path, verbose=False, **self.kwargs)
        except Exception as e:
            version = getattr(llama_cpp, "__version__", "unknown")
            raise RuntimeError(
                f"Failed to load {self.name} with llama-cpp-python {version}: {e}. "
                "If the architecture is unknown (e.g. gemma4), upgrade llama-cpp-python "
                "to >=0.3.38 or rebuild it against a newer llama.cpp."
            ) from e
        _cache["key"] = self.key
        _cache["llm"] = llm
        return llm


# Single-slot cache: one LLM resident at a time keeps VRAM predictable.
_cache = {"key": None, "llm": None}


def _free_cached():
    llm = _cache["llm"]
    _cache["key"] = None
    _cache["llm"] = None
    if llm is not None:
        try:
            llm.close()
        except Exception:
            pass
        del llm
    gc.collect()
    mm.soft_empty_cache()


def _import_llama():
    try:
        import llama_cpp
    except ImportError as e:
        raise RuntimeError(
            "llama-cpp-python is not installed in ComfyUI's Python environment. "
            "See ComfyUI-LLM-GGUF/README.md for install instructions."
        ) from e
    return llama_cpp


class GemmaGGUFLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (folder_paths.get_filename_list(LLM_FOLDER),),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000,
                                         "tooltip": "-1 offloads every layer to the GPU."}),
                "n_ctx": ("INT", {"default": 8192, "min": 512, "max": 262144, "step": 512}),
                "flash_attn": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("GEMMA_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "text/gemma"

    def load(self, model, n_gpu_layers, n_ctx, flash_attn):
        path = folder_paths.get_full_path_or_raise(LLM_FOLDER, model)
        handle = GemmaModel(path, model, n_gpu_layers, n_ctx, flash_attn)
        handle.get()  # load now so failures surface on the loader node
        return (handle,)


class GemmaGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("GEMMA_MODEL",),
                "system": ("STRING", {"multiline": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 32768}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 64, "min": 0, "max": 1000}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "unload_after": ("BOOLEAN", {"default": True,
                                             "tooltip": "Free the LLM's VRAM when done so the next image job has room."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "text/gemma"

    def generate(self, model, system, prompt, max_tokens, temperature, top_p,
                 top_k, repeat_penalty, seed, unload_after, unique_id=None):
        self._unique_id = unique_id
        llm = model.get()

        params = dict(max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                      top_k=top_k, repeat_penalty=repeat_penalty, seed=seed)
        try:
            text = self._run(llm, self._messages(system, prompt, True), params)
        except _SystemRoleRejected:
            # Some chat templates have no system role; fold it into the user turn.
            text = self._run(llm, self._messages(system, prompt, False), params)
        finally:
            if unload_after:
                _free_cached()

        return {"ui": {"text": [text]}, "result": (text,)}

    @staticmethod
    def _messages(system, prompt, use_system_role):
        system = (system or "").strip()
        if not system:
            return [{"role": "user", "content": prompt}]
        if use_system_role:
            return [{"role": "system", "content": system},
                    {"role": "user", "content": prompt}]
        return [{"role": "user", "content": f"{system}\n\n{prompt}"}]

    def _run(self, llm, messages, params):
        pbar = comfy.utils.ProgressBar(params["max_tokens"])
        node_id = getattr(self, "_unique_id", None)
        parts = []
        try:
            stream = llm.create_chat_completion(messages=messages, stream=True, **params)
            for chunk in stream:
                mm.throw_exception_if_processing_interrupted()
                delta = chunk["choices"][0].get("delta", {}).get("content")
                if not delta:
                    continue
                parts.append(delta)
                pbar.update(1)
                if PromptServer is not None:
                    PromptServer.instance.send_sync(
                        "llm_gguf.chunk", {"node": node_id, "chunk": delta})
        except mm.InterruptProcessingException:
            raise
        except Exception as e:
            if messages[0]["role"] == "system" and "system" in str(e).lower():
                raise _SystemRoleRejected() from e
            raise
        return "".join(parts).strip()


class _SystemRoleRejected(Exception):
    pass


NODE_CLASS_MAPPINGS = {
    "GemmaGGUFLoader": GemmaGGUFLoader,
    "GemmaGenerate": GemmaGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GemmaGGUFLoader": "Load Gemma GGUF",
    "GemmaGenerate": "Gemma Generate Text",
}
