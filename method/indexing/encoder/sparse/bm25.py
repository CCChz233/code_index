import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    from scipy.sparse import csr_matrix
except ImportError as exc:  # pragma: no cover
    raise ImportError("scipy is required for sparse indexing") from exc


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def build_bm25_matrix(
    documents: Iterable[str],
    *,
    min_df: int = 1,
    max_df_ratio: float = 1.0,
) -> Tuple[csr_matrix, Dict[str, int], np.ndarray, np.ndarray, float]:
    docs = list(documents)
    num_docs = len(docs)
    if num_docs == 0:
        return csr_matrix((0, 0)), {}, np.array([]), np.array([]), 0.0

    doc_tokens: List[List[str]] = []
    doc_len = np.zeros(num_docs, dtype=np.int32)
    df_counter: Counter[str] = Counter()

    for i, text in enumerate(docs):
        tokens = tokenize(text)
        doc_tokens.append(tokens)
        doc_len[i] = len(tokens)
        df_counter.update(set(tokens))

    max_df = max_df_ratio * num_docs
    filtered_tokens = [
        token for token, df in df_counter.items() if df >= min_df and df <= max_df
    ]
    vocab = {token: idx for idx, token in enumerate(filtered_tokens)}

    rows: List[int] = []
    cols: List[int] = []
    data: List[int] = []

    for row, tokens in enumerate(doc_tokens):
        if not tokens:
            continue
        counts = Counter(tokens)
        for token, tf in counts.items():
            col = vocab.get(token)
            if col is None:
                continue
            rows.append(row)
            cols.append(col)
            data.append(tf)

    if vocab:
        tf_matrix = csr_matrix((data, (rows, cols)), shape=(num_docs, len(vocab)), dtype=np.float32)
    else:
        tf_matrix = csr_matrix((num_docs, 0), dtype=np.float32)

    df = np.zeros(len(vocab), dtype=np.int32)
    for token, idx in vocab.items():
        df[idx] = df_counter.get(token, 0)

    # BM25-style IDF
    idf = np.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
    avgdl = float(doc_len.mean()) if num_docs > 0 else 0.0

    return tf_matrix, vocab, idf, doc_len, avgdl
