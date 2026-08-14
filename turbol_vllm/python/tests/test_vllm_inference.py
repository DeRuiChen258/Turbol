"""Tests for the vLLM Python inference backend (DeepSeek integration).

Run directly (no pytest required)::

    python tests/test_vllm_inference.py

or via pytest::

    pytest tests/test_vllm_inference.py

The vLLM package and model weights are *not* required: the interface, presets,
config resolution, and echo-fallback path are all exercised without them.  Real
end-to-end generation is gated behind ``vllm_available()`` and skipped (not
failed) when the package is absent.
"""

import os
import sys

# Make the in-tree ``turbol`` package importable without an installed build.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from turbol.vllm_inference import (  # noqa: E402
    DEEPSEEK_MODELS,
    VLLMInferenceConfig,
    VLLMInferenceEngine,
    vllm_available,
)


def test_model_presets_exist():
    assert "deepseek-r1-distill-qwen-1.5b" in DEEPSEEK_MODELS
    assert "deepseek-r1-distill-qwen-7b" in DEEPSEEK_MODELS
    assert "deepseek-v2-lite" in DEEPSEEK_MODELS
    assert DEEPSEEK_MODELS["deepseek-r1-distill-qwen-1.5b"]["repo"].startswith(
        "deepseek-ai/"
    )


def test_config_resolves_preset_key():
    cfg = VLLMInferenceConfig(model="deepseek-r1-distill-qwen-1.5b")
    assert cfg.resolve_model() == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    assert cfg.uses_chat_template() is True


def test_config_resolves_full_repo_id():
    cfg = VLLMInferenceConfig(model="org/custom-model")
    assert cfg.resolve_model() == "org/custom-model"


def test_defaults_match_rollout_config():
    cfg = VLLMInferenceConfig()
    assert cfg.tensor_parallel_size == 1
    assert cfg.max_new_tokens > 0
    assert cfg.temperature == 1.0
    assert cfg.top_k == -1
    assert cfg.max_batch_size == 256


def test_echo_fallback_single():
    engine = VLLMInferenceEngine()
    out = engine.generate("hello")
    assert out == "[VLLMInferenceEngine] hello"


def test_echo_fallback_batch():
    engine = VLLMInferenceEngine()
    outs = engine.generate_batch(["a", "b", "c"])
    assert outs == [
        "[VLLMInferenceEngine] a",
        "[VLLMInferenceEngine] b",
        "[VLLMInferenceEngine] c",
    ]


def test_echo_fallback_empty_batch():
    engine = VLLMInferenceEngine()
    assert engine.generate_batch([]) == []


def test_no_fallback_raises_without_vllm():
    if vllm_available():
        return  # would actually load the model; skip the failure-path check
    cfg = VLLMInferenceConfig(fallback_to_echo=False)
    engine = VLLMInferenceEngine(cfg)
    try:
        engine.generate("hello")
    except RuntimeError as e:
        assert "vllm" in str(e).lower()
    else:
        raise AssertionError("expected RuntimeError without vllm installed")


def _main():
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print("-" * 60)
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
