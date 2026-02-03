from method.indexing.chunker.basic import FixedChunker, SlidingChunker, RLFixedChunker, RLMiniChunker
from method.indexing.chunker.function import FunctionLevelChunker, IRFunctionChunker
from method.indexing.chunker.epic import EpicChunker
from method.indexing.chunker.langchain import LangChainFixedChunker, LangChainRecursiveChunker, LangChainTokenChunker
from method.indexing.chunker.llamaindex import (
    LlamaIndexCodeChunker,
    LlamaIndexSentenceChunker,
    LlamaIndexTokenChunker,
    LlamaIndexSemanticChunker,
)
from method.indexing.chunker.summary import SummaryChunker

CHUNKER_REGISTRY = {
    "fixed": FixedChunker,
    "sliding": SlidingChunker,
    "rl_fixed": RLFixedChunker,
    "rl_mini": RLMiniChunker,
    "function_level": FunctionLevelChunker,
    "ir_function": IRFunctionChunker,
    "summary": SummaryChunker,
    "llamaindex_code": LlamaIndexCodeChunker,
    "llamaindex_sentence": LlamaIndexSentenceChunker,
    "llamaindex_token": LlamaIndexTokenChunker,
    "llamaindex_semantic": LlamaIndexSemanticChunker,
    "langchain_fixed": LangChainFixedChunker,
    "langchain_recursive": LangChainRecursiveChunker,
    "langchain_token": LangChainTokenChunker,
    "epic": EpicChunker,
}


def get_chunker(strategy: str, **kwargs):
    chunker_cls = CHUNKER_REGISTRY.get(strategy)
    if chunker_cls is None:
        raise ValueError(f"Unknown strategy: {strategy}")
    return chunker_cls(**kwargs)


__all__ = [
    "FixedChunker",
    "SlidingChunker",
    "RLFixedChunker",
    "RLMiniChunker",
    "FunctionLevelChunker",
    "IRFunctionChunker",
    "SummaryChunker",
    "LlamaIndexCodeChunker",
    "LlamaIndexSentenceChunker",
    "LlamaIndexTokenChunker",
    "LlamaIndexSemanticChunker",
    "LangChainFixedChunker",
    "LangChainRecursiveChunker",
    "LangChainTokenChunker",
    "EpicChunker",
    "CHUNKER_REGISTRY",
    "get_chunker",
]
