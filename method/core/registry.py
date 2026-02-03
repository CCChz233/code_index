from typing import Any, Callable, Dict

_REGISTRY: Dict[str, Dict[str, Any]] = {
    "chunker": {},
    "encoder": {},
    "indexer": {},
    "retriever": {},
    "fusion": {},
}


def register(kind: str, name: str, obj: Any) -> None:
    if kind not in _REGISTRY:
        _REGISTRY[kind] = {}
    _REGISTRY[kind][name] = obj


def get(kind: str, name: str) -> Any:
    return _REGISTRY[kind][name]


def list_registered(kind: str) -> Dict[str, Any]:
    return dict(_REGISTRY.get(kind, {}))


def register_as(kind: str, name: str) -> Callable[[Any], Any]:
    def decorator(obj: Any) -> Any:
        register(kind, name, obj)
        return obj
    return decorator
