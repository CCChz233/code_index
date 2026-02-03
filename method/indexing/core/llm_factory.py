import logging
import os
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def get_llm(
    provider: str,
    model_name: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
):
    """Return a LlamaIndex LLM instance based on provider."""
    if provider == "openai":
        try:
            from llama_index.llms.openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "Missing llama-index-llms-openai. Install with: pip install llama-index-llms-openai"
            ) from exc

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        api_base = api_base or os.environ.get("OPENAI_API_BASE")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for provider=openai.")

        def _should_force_chat() -> bool:
            if api_base:
                parsed = urlparse(api_base)
                host = (parsed.netloc or parsed.path or "").lower()
                if host and "openai.com" not in host:
                    return True
            if model_name:
                name = model_name.lower()
                if name.startswith(("gpt-", "deepseek", "qwen", "claude")):
                    return True
            return False

        force_chat = _should_force_chat()
        if force_chat:
            logger.info("Initializing OpenAI-compatible LLM (forced chat): %s", model_name)
        else:
            logger.info("Initializing OpenAI-compatible LLM: %s", model_name)

        kwargs = {
            "model": model_name,
            "api_key": api_key,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if api_base:
            kwargs["api_base"] = api_base
        if not force_chat:
            return OpenAI(**kwargs)

        for extra in (
            {"is_chat_model": True},
            {"use_chat_completions": True},
            {"model_kwargs": {"is_chat_model": True}},
        ):
            try:
                return OpenAI(**kwargs, **extra)
            except TypeError:
                continue

        try:
            from llama_index.llms.openai_like import OpenAILike
        except ImportError:
            return OpenAI(**kwargs)

        try:
            return OpenAILike(**kwargs, is_chat_model=True)
        except TypeError:
            return OpenAILike(**kwargs)

    if provider == "vllm":
        try:
            from llama_index.llms.vllm import Vllm
        except ImportError as exc:
            raise ImportError(
                "Missing llama-index-llms-vllm. Install with: pip install llama-index-llms-vllm"
            ) from exc

        logger.info("Initializing vLLM local LLM: %s", model_name)
        return Vllm(
            model=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            trust_remote_code=True,
            vllm_kwargs={
                "swap_space": 1,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 4096,
            },
        )

    raise ValueError(f"Unknown provider: {provider}")
