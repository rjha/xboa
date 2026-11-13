from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any
import orjson

_ORJSON_OPTIONS = (
    orjson.OPT_NON_STR_KEYS 
    | orjson.OPT_PASSTHROUGH_DATACLASS
    | orjson.OPT_SERIALIZE_NUMPY 
)

def _orjson_default(obj: Any) -> Any:
    """Fallback serializer for nested objects and sets encountered by orjson."""
    if isinstance(obj, set):
        return list(obj)
    return object_to_dict(obj)


def _filter_keys(data: Mapping[str, Any], keys: Sequence[str]) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected a Mapping, got {type(data).__name__}")
    if isinstance(keys, (str, bytes)):
        raise TypeError("keys must be a sequence of strings")
    
    return {k: data[k] for k in keys if k in data} if keys else data


def _dump_json(payload: Any) -> str:
    try:
        return orjson.dumps(
            payload,
            default=_orjson_default,
            option=_ORJSON_OPTIONS,
        ).decode("utf-8")
    except Exception as exc:
        # failsafe for circular references or broken __repr__ etc
        return orjson.dumps({
            "trace_serialization_error": str(exc),
            "fallback_repr": f"<unrepresentable {type(payload).__name__}>"
        }).decode("utf-8")


def object_to_dict(obj: Any) -> dict[str, Any]:
    """
    Convert a dataclass instance or standard object into a shallow dict,
    excluding private attributes. Raises TypeError for non-objects.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: getattr(obj, f.name)
            for f in fields(obj)
            if not f.name.startswith("_")
        }

    if not isinstance(obj, type) and hasattr(obj, "__dict__"):
        return {
            k: v
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }

    raise TypeError(f"Expected a dataclass or object instance, got {type(obj).__name__}")



def format_map(data: Mapping[str, Any], keys: Sequence[str] = ()) -> str:
    """Filter a single mapping by optional top-level keys and serialize as JSON."""
    return _dump_json(_filter_keys(data, keys))


def format_maps(items: Iterable[Mapping[str, Any]], keys: Sequence[str] = ()) -> str:
    """Filter an iterable of mappings by optional top-level keys and serialize as JSON."""
    if isinstance(items, (str, bytes, Mapping)):
        raise TypeError(f"Expected an iterable of mappings, got {type(items).__name__}")
    return _dump_json([_filter_keys(item, keys) for item in items])


def format_record(obj: Any, keys: Sequence[str] = ()) -> str:
    """Convert a single object/dataclass to a dict, filter keys, and serialize as JSON."""
    return _dump_json(_filter_keys(object_to_dict(obj), keys))


def format_records(items: Iterable[Any], keys: Sequence[str] = ()) -> str:
    """Convert an iterable of objects/dataclasses, filter keys, and serialize as JSON."""
    if isinstance(items, (str, bytes, Mapping)):
        raise TypeError(f"Expected an iterable of objects, got {type(items).__name__}")
    return _dump_json([_filter_keys(object_to_dict(item), keys) for item in items])


def format_list(items: Iterable[Any]) -> str:
    """Serialize a list or iterable of arbitrary items (primitives, dicts, objects) as JSON."""
    if isinstance(items, (str, bytes, Mapping)):
        raise TypeError(f"Expected a list or iterable of items, got {type(items).__name__}")
    return _dump_json(list(items))
