"""Experimental tail routing without changing helium2 or its CAS settings.

Logical workers, graph dependencies, result references and the original CAS
order remain intact.  After a scheduler grant, future HTTP requests may overflow
from their original replica to a less busy newly granted replica.  This does not
move requests already sent to vLLM, transfer KV caches, or replan the full CAS
schedule for the enlarged physical pool.  Extra replicas can have cold caches;
an allocation is an opportunity to accelerate, not a non-regression guarantee.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from copy import copy
from dataclasses import dataclass, replace
from typing import Any

from helium2.common import GenerationConfig
from helium2.runtime.llm import LLMServiceConfig
from helium2.runtime.llm.openai import (
    BenchmarkRequest,
    RequestType,
    ResponseType,
    _AsyncOpenAIClient,
    _handle_client_request,
)
from helium2.runtime.llm.vllm_openai import VLLMOpenAILLM
from helium2.runtime.utils.loop import MPConcurrentEventLoop
from helium2.runtime.utils.queue import AIOQueue
from helium2.runtime.worker.llm_worker import LLMWorker
from helium_new2.runtime.tail_resources import TailResourceState


def _prepared_config(config: GenerationConfig | None) -> GenerationConfig:
    if config is None:
        raise ValueError("Tail HTTP request configuration has not been resolved")
    return config


@dataclass
class _TailClientContext:
    client: _AsyncOpenAIClient
    resolve_config: Callable[[GenerationConfig | None], GenerationConfig]
    resources: TailResourceState
    home_url: str


@asynccontextmanager
async def _tail_client_context(
    default_llm_role: str,
    resolve_config: Callable[[GenerationConfig | None], GenerationConfig],
    resources: TailResourceState,
    home_url: str,
) -> AsyncGenerator[_TailClientContext, None]:
    yield _TailClientContext(
        # Final endpoint selection is done below.  Do not run the original
        # VLLM _get_config again, as that would overwrite the selected URL.
        client=_AsyncOpenAIClient(default_llm_role, _prepared_config),
        resolve_config=resolve_config,
        resources=resources,
        home_url=home_url,
    )


async def _handle_tail_request(
    event: RequestType, context: _TailClientContext | None
) -> ResponseType:
    assert context is not None
    if isinstance(event, BenchmarkRequest):
        return await _handle_client_request(event, context.client)

    # This function runs in the actual HTTP client process.  Shared grants are
    # visible here even though this process was started before tail expansion.
    config = copy(context.resolve_config(copy(event.config)))
    index, (base_url, _, api_key) = context.resources.begin_request(context.home_url)
    try:
        config.base_url = base_url
        config.api_key = api_key
        return await _handle_client_request(
            replace(event, config=config), context.client
        )
    finally:
        context.resources.end_request(index)


class TailVLLMOpenAILLM(VLLMOpenAILLM):
    def __init__(
        self, *, config: LLMServiceConfig, tail_resources: TailResourceState
    ) -> None:
        super().__init__(config=config)
        if self._base_url is None:
            raise ValueError("Tail acceleration requires a fixed initial vLLM endpoint")
        self._tail_resources = tail_resources

    def _get_config(self, config: GenerationConfig | None) -> GenerationConfig:
        # The inherited implementation assigns endpoint fields in-place.  Give
        # it a request-local copy so concurrent calls retain their own config.
        return super()._get_config(None if config is None else copy(config))

    async def start(self) -> None:
        if self._client_loop is not None:
            await self._client_loop.start()
            return
        assert self._base_url is not None
        client_loop = MPConcurrentEventLoop(
            _handle_tail_request,
            result_collector=AIOQueue(),
            context_manager=_tail_client_context(
                self.DEFAULT_LLM_ROLE,
                self._get_config,
                self._tail_resources,
                self._base_url,
            ),
        )
        await client_loop.start()
        self._client_loop = client_loop


class TailLLMWorker(LLMWorker):
    """Keep the original worker/dispatcher and substitute only its HTTP LLM."""

    def __init__(
        self, *args: Any, tail_resources: TailResourceState, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        # The original client has not been started, so replacing it creates no
        # extra services.  Tokenizer, batching support, dispatcher and settings
        # are identical to those of the original vllm-openai implementation.
        self._llm = TailVLLMOpenAILLM(
            config=self.service_config, tail_resources=tail_resources
        )
