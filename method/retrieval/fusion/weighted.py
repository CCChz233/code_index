from typing import Dict, Iterable, List, Tuple


def fuse_weighted(
    ranked_lists: Iterable[List[Tuple[int, float]]],
    *,
    weights: Iterable[float],
) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = {}
    for ranked, w in zip(ranked_lists, weights):
        for doc_id, score in ranked:
            scores[doc_id] = scores.get(doc_id, 0.0) + w * float(score)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
