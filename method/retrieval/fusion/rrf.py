from typing import Dict, Iterable, List, Tuple


def fuse_rrf(
    ranked_lists: Iterable[List[Tuple[int, float]]],
    *,
    k: int = 60,
) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
