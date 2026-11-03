from helium2.runtime.llm.base import (
    BaseLLM,
    BatchChatMixin,
    BatchCompleteMixin,
    LLMProfilingInfo,
    LLMServiceConfig,
    LLMServiceInfo,
    UsageInfo,
)
from helium2.runtime.llm.mock import MockLLM
from helium2.runtime.llm.openai import OpenAILLM
from helium2.runtime.llm.registry import LLMRegistry

try:
    from helium2.runtime.llm.vllm_local import VLLMLocalLLM  # type: ignore[import]
except ModuleNotFoundError:
    import sys

    print("[Warning] Failed to import vLLM. Disabling VLLMLocalLLM", file=sys.stderr)

    class VLLMLocalLLM(BaseLLM):  # type: ignore[no-redef]
        """Dummy class to allow import without vLLM."""


from helium2.runtime.llm.vllm_openai import VLLMOpenAILLM

__all__ = [
    "BaseLLM",
    "MockLLM",
    "OpenAILLM",
    "VLLMLocalLLM",
    "VLLMOpenAILLM",
    "LLMRegistry",
    "LLMProfilingInfo",
    "LLMServiceConfig",
    "LLMServiceInfo",
    "UsageInfo",
    "BatchChatMixin",
    "BatchCompleteMixin",
]
