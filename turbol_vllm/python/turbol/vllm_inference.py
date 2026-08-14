"""
TurboRL — vLLM Python inference backend for DeepSeek small-parameter models.

The C++ ``RolloutEngine`` ships a ``VLLMBackend`` stub that can only speak to the
bundled C++ ``libvllm.a`` (llama/qwen/gemma only).  For DeepSeek — and for the
recommended, fully-featured serving stack — the correct integration point is the
vLLM *Python* API (``vllm.LLM``).  This module provides that bridge with the same
``Generate`` / ``GenerateBatch`` surface as the C++ engine, so the rest of the
rollout loop is unchanged.

The ``vllm`` package is imported lazily: this module is always importable, and
degrades to a deterministic echo backend when ``vllm`` (or a model) is absent.
That keeps TurboRL usable on CPU-only / non-GPU machines while remaining
drop-in ready for the real DeepSeek weights.

Usage::

    from turbol.vllm_inference import VLLMInferenceEngine, VLLMInferenceConfig

    cfg = VLLMInferenceConfig(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        max_new_tokens=256,
    )
    engine = VLLMInferenceEngine(cfg)
    print(engine.generate("Hello, who are you?"))
    print(engine.generate_batch(["a", "b", "c"]))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

__all__ = [
    "DEEPSEEK_MODELS",
    "VLLMInferenceConfig",
    "VLLMInferenceEngine",
    "vllm_available",
]


# --------------------------------------------------------------------------- #
# DeepSeek small-parameter model presets (ascending by parameter count)
# --------------------------------------------------------------------------- #
#
# The "lightweight" candidates for a local RTX 5070 (12 GB) are the Qwen/Llama
# distill series.  DeepSeek-R1-Distill-Qwen-1.5B is the smallest and is the
# default for quick local debugging; the 7B/8B variants give better quality at
# the cost of more VRAM.  DeepSeek-V2-Lite (MoE) is included for reference.
DEEPSEEK_MODELS: dict = {
    "deepseek-r1-distill-qwen-1.5b": {
        "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "params": "1.5B",
        "chat": True,
    },
    "deepseek-r1-distill-qwen-7b": {
        "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "params": "7B",
        "chat": True,
    },
    "deepseek-r1-distill-llama-8b": {
        "repo": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "params": "8B",
        "chat": True,
    },
    "deepseek-v2-lite": {
        "repo": "deepseek-ai/DeepSeek-V2-Lite",
        "params": "15.7B (MoE, ~2.4B active)",
        "chat": True,
    },
    "deepseek-llm-7b-chat": {
        "repo": "deepseek-ai/deepseek-llm-7b-chat",
        "params": "7B",
        "chat": True,
    },
}


def vllm_available() -> bool:
    """Return True if the ``vllm`` Python package can be imported."""
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
# Config — mirrors the C++ ``rollout::RolloutConfig`` fields
# --------------------------------------------------------------------------- #

@dataclass
class VLLMInferenceConfig:
    model: str = "deepseek-r1-distill-qwen-1.5b"
    """Model repo id (Hugging Face) or a key from :data:`DEEPSEEK_MODELS`."""

    tensor_parallel_size: int = 1
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_batch_size: int = 256
    dtype: str = "auto"
    trust_remote_code: bool = True
    use_chat_template: bool = False
    gpu_memory_utilization: float = 0.90
    fallback_to_echo: bool = True
    """When vLLM/model is unavailable, return a deterministic echo instead of raising."""

    def resolve_model(self) -> str:
        """Resolve a preset key to its full Hugging Face repo id."""
        return DEEPSEEK_MODELS.get(self.model, {}).get("repo", self.model)

    def uses_chat_template(self) -> bool:
        return self.use_chat_template or DEEPSEEK_MODELS.get(self.model, {}).get("chat", False)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

class VLLMInferenceEngine:
    """Thin, lazy wrapper around ``vllm.LLM`` for DeepSeek rollout generation.

    The heavy ``vllm`` import and the actual model load both happen on first use
    (``_ensure_loaded``), never at construction time, so instantiating the engine
    on a machine without vLLM is free.
    """

    def __init__(self, config: Optional[VLLMInferenceConfig] = None):
        self.config = config or VLLMInferenceConfig()
        self._llm = None
        self._sampling_params = None
        self._load_error: Optional[Exception] = None

    # -- availability -------------------------------------------------------

    def is_available(self) -> bool:
        """True once the underlying vLLM model has actually been loaded."""
        return self._llm is not None

    def availability_error(self) -> Optional[str]:
        """Human-readable reason the engine is not available, if any."""
        if self._llm is not None:
            return None
        if not vllm_available():
            return "vllm package is not installed (pip install vllm)"
        if self._load_error is not None:
            return f"vllm model load failed: {self._load_error}"
        return "model not loaded yet"

    # -- loading ------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        if self._load_error is not None:
            raise RuntimeError(self.availability_error())

        if not vllm_available():
            raise RuntimeError(
                "vllm is not installed. Install it (pip install vllm) or run with "
                "fallback_to_echo=True to use the echo backend."
            )

        try:
            from vllm import LLM, SamplingParams

            self._llm = LLM(
                model=self.config.resolve_model(),
                tensor_parallel_size=self.config.tensor_parallel_size,
                dtype=self.config.dtype,
                trust_remote_code=self.config.trust_remote_code,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=None,
            )
            self._sampling_params = SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k if self.config.top_k > 0 else -1,
                max_tokens=self.config.max_new_tokens,
            )
        except Exception as e:  # model download / OOM / driver mismatch, etc.
            self._load_error = e
            raise RuntimeError(self.availability_error()) from e

    # -- formatting ----------------------------------------------------------

    def _format_prompt(self, prompt: str) -> str:
        if not self.config.uses_chat_template():
            return prompt
        self._ensure_loaded()
        tokenizer = self._llm.get_tokenizer()
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # -- generation ----------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Generate a completion for a single prompt.

        Raises ``RuntimeError`` when vLLM is unavailable and
        ``fallback_to_echo`` is disabled; otherwise returns an echo.
        """
        if self.is_available():
            return self.generate_batch([prompt])[0]

        if self.config.fallback_to_echo:
            return "[VLLMInferenceEngine] " + prompt

        self._ensure_loaded()  # raises a clear RuntimeError
        raise RuntimeError(self.availability_error())

    def generate_batch(self, prompts: Sequence[str]) -> List[str]:
        """Generate completions for a batch of prompts, preserving order."""
        prompts = list(prompts)
        if not prompts:
            return []

        if self._llm is None and self.config.fallback_to_echo:
            return ["[VLLMInferenceEngine] " + p for p in prompts]

        self._ensure_loaded()

        formatted = [self._format_prompt(p) for p in prompts]
        outputs = self._llm.generate(
            formatted, self._sampling_params, use_tqdm=False
        )
        return [o.outputs[0].text for o in outputs]
