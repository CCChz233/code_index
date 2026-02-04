import logging
import os
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Default context window for custom/unknown models (e.g., vLLM served models)
_DEFAULT_CONTEXT_WINDOW = 131072


def _patch_openai_model_validation():
    """
    Monkey-patch LlamaIndex's model name validation to support custom models.
    
    LlamaIndex's OpenAI class validates model names against a whitelist of official
    OpenAI models. This fails for custom models served via vLLM or other OpenAI-compatible
    APIs. We patch the validation function to return a default context size for unknown models.
    """
    try:
        import llama_index.llms.openai.utils as openai_utils
        
        _original_contextsize = openai_utils.openai_modelname_to_contextsize
        
        def _patched_contextsize(modelname: str) -> int:
            try:
                return _original_contextsize(modelname)
            except ValueError:
                # Unknown model (e.g., GPT-OSS-120B from vLLM) - return default
                logger.debug("Unknown model '%s', using default context window %d", 
                           modelname, _DEFAULT_CONTEXT_WINDOW)
                return _DEFAULT_CONTEXT_WINDOW
        
        openai_utils.openai_modelname_to_contextsize = _patched_contextsize
        
        # Also patch it in the base module if imported there
        try:
            import llama_index.llms.openai.base as openai_base
            openai_base.openai_modelname_to_contextsize = _patched_contextsize
        except (ImportError, AttributeError):
            pass
            
    except ImportError:
        pass  # llama_index not installed


# Apply the patch when this module is loaded
_patch_openai_model_validation()


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

        def _is_non_openai_api() -> bool:
            """Check if api_base points to a non-OpenAI server (e.g., vLLM, local)."""
            if api_base:
                parsed = urlparse(api_base)
                host = (parsed.netloc or parsed.path or "").lower()
                if host and "openai.com" not in host:
                    return True
            return False

        kwargs = {
            "model": model_name,
            "api_key": api_key,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if api_base:
            kwargs["api_base"] = api_base

        if _is_non_openai_api():
            logger.info("Initializing OpenAI-compatible LLM: %s @ %s", model_name, api_base)
        else:
            logger.info("Initializing OpenAI LLM: %s", model_name)

        return OpenAI(**kwargs)

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
