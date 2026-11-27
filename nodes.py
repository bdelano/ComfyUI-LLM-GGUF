"""GGUF LLM text generation nodes for ComfyUI, backed by llama-cpp-python.

Works with any chat model llama.cpp supports (Gemma, Llama, Qwen, Mistral, ...).
Loads a GGUF LLM from ComfyUI's ``models/LLM/`` folder and generates text with
the chat template embedded in the GGUF. The model shares ComfyUI's queue (so
text jobs wait behind image jobs instead of fighting them for VRAM) and honors
Interrupt / Clear Queue between tokens.
"""

import gc
import logging
import os
import re

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


class GGUFModel:
    """Handle passed between nodes (custom LLM_GGUF_MODEL type).

    Holds only the load parameters, not the Llama object itself: ComfyUI caches
    the loader's output, so after ``unload_after`` frees VRAM the loader won't
    re-run on the next prompt. ``LLMGenerate`` calls ``get()``, which reloads
    on demand.
    """

    def __init__(self, path, name, n_gpu_layers, n_ctx, flash_attn, swa_full, kv_cache_type,
                 chat_format="auto"):
        self.path = path
        self.name = name
        self.key = (path, n_gpu_layers, n_ctx, flash_attn, swa_full, kv_cache_type, chat_format)
        self.kwargs = dict(n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, flash_attn=flash_attn,
                           swa_full=swa_full)
        self.kv_cache_type = kv_cache_type
        self.chat_format = chat_format

    def get(self):
        if _cache["key"] == self.key and _cache["llm"] is not None:
            return _cache["llm"]

        llama_cpp = _import_llama()
        _free_cached()
        # Make room: evict whatever diffusion/TTS models ComfyUI has resident.
        mm.unload_all_models()
        mm.soft_empty_cache()
        kv_type = getattr(llama_cpp, KV_CACHE_TYPES[self.kv_cache_type])
        try:
            llm = llama_cpp.Llama(model_path=self.path, verbose=False,
                                  type_k=kv_type, type_v=kv_type, **self.kwargs)
        except Exception as e:
            version = getattr(llama_cpp, "__version__", "unknown")
            msg = str(e)
            if "llama_context" in msg:
                # Weights loaded fine; allocating the KV cache/compute buffers failed.
                hint = ("The model weights loaded, but the context (KV cache) could not be "
                        "allocated, which is usually out of VRAM. Try a smaller n_ctx, "
                        "swa_full off, kv_cache_type q8_0 (needs flash_attn), or lower "
                        "n_gpu_layers to keep some layers on the CPU.")
            else:
                hint = ("If the architecture is unknown (e.g. gemma4), upgrade "
                        "llama-cpp-python to >=0.3.38 or rebuild it against a newer llama.cpp.")
            raise RuntimeError(
                f"Failed to load {self.name} with llama-cpp-python {version}: {msg}. {hint}"
            ) from e
        llm.chat_format = _resolve_chat_format(llm, self.chat_format, self.name)
        _cache["key"] = self.key
        _cache["llm"] = llm
        return llm


def _register_chat_formats():
    """Prompt formats for GGUFs that don't embed a chat template.

    llama-cpp-python's own fallback for a template-less GGUF is the Llama 2
    format ("<s>[INST] ... [/INST]"), which a Gemma model only sees as plain
    text: it echoes "</s>" and degenerates into bracket junk. Its built-in
    "gemma" format also silently drops the system prompt. These follow the
    official templates instead.

    Each prompt spells out its own "<bos>", so the responses set
    added_special: otherwise llama-cpp-python tokenizes with add_bos and
    prepends a second BOS (its "check_double_bos_eos" warning), which
    degrades the output. Its own Jinja (embedded-template) path does the same.
    """
    from llama_cpp import llama_chat_format as lcf

    def turns(messages):
        system = "\n\n".join((m.get("content") or "").strip()
                              for m in messages if m["role"] in ("system", "developer"))
        rest = [m for m in messages if m["role"] not in ("system", "developer")]
        return system, rest

    @lcf.register_chat_format(CHAT_FORMATS["gemma4"])
    def gemma4(messages, **kwargs):
        system, rest = turns(messages)
        thinking = _generation_options["enable_thinking"]
        prompt = "<bos>"
        if thinking or system:
            # The official template's thinking switch opens the system turn.
            prompt += "<|turn>system\n" + ("<|think|>\n" if thinking else "") + f"{system}<turn|>\n"
        for m in rest:
            role = "model" if m["role"] == "assistant" else "user"
            prompt += f"<|turn>{role}\n{(m.get('content') or '').strip()}<turn|>\n"
        prompt += "<|turn>model\n"
        if not thinking:
            # An empty thought channel is how the official template turns
            # thinking off.
            prompt += "<|channel>thought\n<channel|>"
        return lcf.ChatFormatterResponse(prompt=prompt, stop=["<turn|>"], added_special=True)

    @lcf.register_chat_format(CHAT_FORMATS["gemma"])
    def gemma(messages, **kwargs):
        # Gemma 2/3 have no system role; the official template prepends it to
        # the first user turn.
        system, rest = turns(messages)
        prompt = "<bos>"
        for i, m in enumerate(rest):
            role = "model" if m["role"] == "assistant" else "user"
            content = (m.get("content") or "").strip()
            if i == 0 and system and role == "user":
                content = f"{system}\n\n{content}"
            prompt += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
        prompt += "<start_of_turn>model\n"
        return lcf.ChatFormatterResponse(prompt=prompt, stop=["<end_of_turn>"], added_special=True)


# Per-generation options the node's own chat formatters read. llama-cpp-python
# gives a chat formatter no way to receive extra per-call options, so
# LLMGenerate sets these for the duration of one generation (ComfyUI runs one
# node at a time).
_generation_options = {"enable_thinking": False}

# Loader choices -> llama-cpp-python chat_format names. "auto" means the
# GGUF's embedded template, else a guess from its architecture.
CHAT_FORMATS = {
    "auto": None,
    "gemma4": "comfy-gemma4",
    "gemma": "comfy-gemma",
    "chatml": "chatml",
    "llama-3": "llama-3",
}
_formats_registered = False


def _resolve_chat_format(llm, choice, name):
    global _formats_registered
    if not _formats_registered:
        _register_chat_formats()
        _formats_registered = True
    if choice != "auto":
        return CHAT_FORMATS[choice]
    metadata = getattr(llm, "metadata", None) or {}
    if metadata.get("tokenizer.chat_template"):
        return llm.chat_format  # llama-cpp-python already chose from the template
    arch = metadata.get("general.architecture", "")
    if arch == "gemma4":
        fmt = "gemma4"
    elif arch.startswith("gemma"):
        fmt = "gemma"
    elif arch == "llama":
        fmt = "llama-3"
    else:
        fmt = "chatml"
    log.warning("%s has no embedded chat template; using the %s format for its %r "
                "architecture. Set the loader's chat_format if this is wrong.", name, fmt, arch)
    return CHAT_FORMATS[fmt]


# KV cache element types (llama_cpp module constant names). Quantized V cache
# requires flash attention.
KV_CACHE_TYPES = {"f16": "GGML_TYPE_F16", "q8_0": "GGML_TYPE_Q8_0", "q4_0": "GGML_TYPE_Q4_0"}

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


class LLMGGUFLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (folder_paths.get_filename_list(LLM_FOLDER),),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000,
                                         "tooltip": "-1 offloads every layer to the GPU."}),
                "n_ctx": ("INT", {"default": 8192, "min": 512, "max": 262144, "step": 512}),
                "flash_attn": ("BOOLEAN", {"default": True}),
                "swa_full": ("BOOLEAN", {"default": False,
                                         "tooltip": "Full-size sliding-window-attention cache. llama.cpp "
                                                    "defaults this on; off uses far less VRAM on Gemma models."}),
                "kv_cache_type": (list(KV_CACHE_TYPES), {"default": "f16",
                                  "tooltip": "q8_0 roughly halves KV cache VRAM; q8_0/q4_0 need flash_attn."}),
            },
            # Optional so workflows saved before this input existed still validate.
            "optional": {
                "chat_format": (list(CHAT_FORMATS), {"default": "auto",
                                "tooltip": "Prompt format. auto uses the GGUF's embedded chat template, or "
                                           "picks one from the model's architecture if it has none."}),
            },
        }

    RETURN_TYPES = ("LLM_GGUF_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "text/llm"

    def load(self, model, n_gpu_layers, n_ctx, flash_attn, swa_full=False, kv_cache_type="f16",
             chat_format="auto"):
        if kv_cache_type != "f16" and not flash_attn:
            raise ValueError("kv_cache_type q8_0/q4_0 requires flash_attn to be enabled.")
        path = folder_paths.get_full_path_or_raise(LLM_FOLDER, model)
        handle = GGUFModel(path, model, n_gpu_layers, n_ctx, flash_attn, swa_full, kv_cache_type,
                           chat_format)
        handle.get()  # load now so failures surface on the loader node
        return (handle,)


class LLMGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LLM_GGUF_MODEL",),
                "system": ("STRING", {"multiline": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "max_tokens": ("INT", {"default": 32000, "min": 1, "max": 262144,
                                       "tooltip": "Hard cap on generated tokens (reasoning included); output is cut off if reached. Also limited by n_ctx minus the prompt."}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 64, "min": 0, "max": 1000}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": True}),
                "unload_after": ("BOOLEAN", {"default": True,
                                             "tooltip": "Free the LLM's VRAM when done so the next image job has room."}),
            },
            # Optional so API prompts saved before this input existed still validate.
            "optional": {
                "strip_thinking": ("BOOLEAN", {"default": True,
                                               "tooltip": "Remove reasoning blocks (<think>...</think>, Gemma 4's "
                                                          "thought channel, ...) from the output text."}),
                # Last, so canvas workflows saved before it existed keep their
                # positional widget values.
                "enable_thinking": ("BOOLEAN", {"default": False,
                                                "tooltip": "Let the model reason before answering (slower, uses "
                                                           "max_tokens). Applies to the node's own Gemma 4 format, "
                                                           "i.e. Gemma 4 GGUFs without an embedded chat template."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "text/llm"

    def generate(self, model, system, prompt, max_tokens, temperature, top_p,
                 top_k, repeat_penalty, seed, unload_after, strip_thinking=True,
                 enable_thinking=False, unique_id=None):
        self._unique_id = unique_id
        llm = model.get()

        params = dict(max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                      top_k=top_k, repeat_penalty=repeat_penalty, seed=seed)
        # Always render special tokens: the END_OF_TURN stop strings below
        # can only match if end-of-turn tokens appear in the text, and the
        # reasoning markers strip_thinking relies on are special tokens too.
        restore = _render_special_tokens(llm)
        params["stop"] = END_OF_TURN
        _generation_options["enable_thinking"] = bool(enable_thinking)
        try:
            try:
                text = self._run(llm, self._messages(system, prompt, True), params)
            except _SystemRoleRejected:
                # Some chat templates have no system role; fold it into the user turn.
                text = self._run(llm, self._messages(system, prompt, False), params)
        finally:
            _generation_options["enable_thinking"] = False
            if restore:
                restore()
            if unload_after:
                _free_cached()

        if strip_thinking:
            text = strip_reasoning(text)
        else:
            # Raw output: reasoning (and its markers) kept, just trimmed.
            text = text.strip()

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
                choice = chunk["choices"][0]
                if choice.get("finish_reason") == "length":
                    log.warning(
                        "Generation hit max_tokens (%s) and was cut off. If the output "
                        "ends in repeated junk, the model probably never emitted a stop "
                        "token it was trained on.", params["max_tokens"])
                delta = choice.get("delta", {}).get("content")
                if not delta:
                    continue
                parts.append(delta)
                pbar.update(1)
                if PromptServer is not None:
                    # Sent only to the client that queued this prompt (like
                    # ComfyUI's own progress events), tagged with its prompt id.
                    server = PromptServer.instance
                    server.send_sync(
                        "llm_gguf.chunk",
                        {"node": node_id, "prompt_id": getattr(server, "last_prompt_id", None),
                         "chunk": delta},
                        getattr(server, "client_id", None))
        except mm.InterruptProcessingException:
            raise
        except Exception as e:
            if messages[0]["role"] == "system" and "system" in str(e).lower():
                raise _SystemRoleRejected() from e
            raise
        return "".join(parts).strip()


class _SystemRoleRejected(Exception):
    pass


# End-of-turn markers across chat formats, used as stop strings. llama.cpp
# already ends generation on most of these tokens by name; this is a safety
# net for GGUFs whose vocab marks them oddly. Matching them as text works
# because special tokens are rendered. "</s>" is deliberately absent: a model
# prompted in the wrong format can write it as plain text at the very start
# of its output, and stopping there would return nothing.
END_OF_TURN = [
    "<turn|>",          # Gemma 4
    "<end_of_turn>",    # Gemma 2/3
    "<|im_end|>",       # ChatML (Qwen, many fine-tunes)
    "<|eot_id|>",       # Llama 3
    "<|end|>",          # Phi
    "<|endoftext|>",
    "<|end_of_text|>",
]


def _render_special_tokens(llm):
    """Make ``llm`` keep special tokens in generated text until the returned
    restore function is called.

    llama-cpp-python detokenizes completions with ``special=False``, which drops
    control tokens. Some models (Gemma 4) mark reasoning only with control
    tokens (``<|channel>thought ... <channel|>``), so without this the thought
    text would be indistinguishable from the answer.
    """
    original = llm.detokenize

    def detokenize(tokens, prev_tokens=None, special=False):
        return original(tokens, prev_tokens=prev_tokens, special=True)

    llm.detokenize = detokenize

    def restore():
        llm.__dict__.pop("detokenize", None)

    return restore


# Reasoning blocks, as (open, close) regex pairs.
_REASONING_TAGS = [
    (r"<\|channel>thought", r"<channel\|>"),   # Gemma 4
    (r"<think>", r"</think>"),                 # Qwen3, DeepSeek R1, many fine-tunes
    (r"<thinking>", r"</thinking>"),
    (r"<reasoning>", r"</reasoning>"),
]
# Leftover control tokens once special tokens are rendered, e.g. Gemma's
# <|channel>, <channel|>, <turn|>, ChatML's <|im_end|>, or a stray </s> / <eos>.
_CONTROL_TOKEN = re.compile(r"<\|[A-Za-z_\"]*\|?>|<[A-Za-z_]+\|>|</?s>|<(?:bos|eos|pad|unk)>")


def strip_reasoning(text):
    """Remove model reasoning from ``text``, leaving only the answer.

    Raises if the model opened a reasoning block and never closed it (usually
    max_tokens ran out mid-thought), since then there is no answer to return.
    """
    for open_tag, close_tag in _REASONING_TAGS:
        text = re.sub(open_tag + r".*?" + close_tag, "", text, flags=re.DOTALL)
        if re.search(open_tag, text):
            raise RuntimeError(
                "The model's reasoning block was never closed, so there is no answer "
                "(max_tokens probably ran out while it was thinking). Increase "
                "max_tokens, or turn strip_thinking off to see the raw output.")
        # Templates like DeepSeek R1's put the opening tag in the prompt, so the
        # output starts mid-thought and contains only the closing tag.
        m = re.search(close_tag, text)
        if m:
            text = text[m.end():]
    return _CONTROL_TOKEN.sub("", text).strip()


class LegacyGemmaGGUFLoader(LLMGGUFLoader):
    """Old name kept so workflows saved before the rename still load."""
    DEPRECATED = True
    RETURN_TYPES = ("GEMMA_MODEL",)


class LegacyGemmaGenerate(LLMGenerate):
    """Old name kept so workflows saved before the rename still load."""
    DEPRECATED = True

    @classmethod
    def INPUT_TYPES(cls):
        types = LLMGenerate.INPUT_TYPES()
        types["required"]["model"] = ("GEMMA_MODEL",)
        return types


NODE_CLASS_MAPPINGS = {
    "LLMGGUFLoader": LLMGGUFLoader,
    "LLMGenerate": LLMGenerate,
    # Deprecated aliases (hidden from the node search menu).
    "GemmaGGUFLoader": LegacyGemmaGGUFLoader,
    "GemmaGenerate": LegacyGemmaGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMGGUFLoader": "Load LLM GGUF",
    "LLMGenerate": "LLM Generate Text",
    "GemmaGGUFLoader": "Load Gemma GGUF (deprecated)",
    "GemmaGenerate": "Gemma Generate Text (deprecated)",
}
