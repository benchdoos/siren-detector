"""Stored engine-graph.v1 configuration and validation helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
import math
from typing import Any


CONTRACT = "engine-graph.v1"
EMPTY_GRAPH = {
    "contract": CONTRACT,
    "engine_configs": {},
    "engine_routes": {"default": [], "by_source": {}},
}
_ENGINE_ID = re.compile(r"^[a-z][a-z0-9._-]*$")
_JSON_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}
_SCHEMA_KEYWORDS = {
    "type", "enum", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "multipleOf", "minLength", "maxLength", "pattern", "minItems", "maxItems",
    "uniqueItems", "items", "minProperties", "maxProperties", "required", "properties",
    "additionalProperties", "title", "description", "default", "examples", "deprecated",
    "readOnly", "writeOnly", "$comment", "$id", "$schema",
}


def validate_graph(graph: Any) -> dict[str, Any]:
    """Validate graph structure while preserving unrecognized extension fields."""
    if not isinstance(graph, Mapping):
        raise ValueError("engine graph must be an object")
    if graph.get("contract", CONTRACT) != CONTRACT:
        raise ValueError(f"unsupported engine graph contract: {graph.get('contract')!r}")
    configs = graph.get("engine_configs", {})
    routes = graph.get("engine_routes", {})
    if not isinstance(configs, Mapping) or not isinstance(routes, Mapping):
        raise ValueError("engine_configs and engine_routes must be objects")

    normalized_configs: dict[str, dict[str, Any]] = {}
    for engine_id, config in configs.items():
        if not isinstance(engine_id, str) or not _ENGINE_ID.fullmatch(engine_id):
            raise ValueError(f"invalid engine_id: {engine_id!r}")
        if not isinstance(config, Mapping):
            raise ValueError(f"engine_configs[{engine_id!r}] must be an object")
        selector = config.get("selector")
        enabled = config.get("enabled", True)
        settings = config.get("settings", {})
        if not isinstance(selector, str) or not selector.strip():
            raise ValueError(f"engine_configs[{engine_id!r}].selector must be a non-empty string")
        if not isinstance(enabled, bool):
            raise ValueError(f"engine_configs[{engine_id!r}].enabled must be boolean")
        if not isinstance(settings, Mapping):
            raise ValueError(f"engine_configs[{engine_id!r}].settings must be an object")
        normalized_configs[engine_id] = {**config, "selector": selector.strip(),
                                         "enabled": enabled, "settings": dict(settings)}

    default_route = routes.get("default", [])
    by_source = routes.get("by_source", {})
    if not _valid_route(default_route):
        raise ValueError("engine_routes.default must be an array of unique engine IDs")
    if not isinstance(by_source, Mapping):
        raise ValueError("engine_routes.by_source must be an object")
    normalized_source_routes = {}
    for source_id, route in by_source.items():
        if not str(source_id).isdigit():
            raise ValueError(f"invalid source ID in engine_routes.by_source: {source_id!r}")
        if not _valid_route(route):
            raise ValueError(f"engine_routes.by_source[{source_id!r}] must be an array of unique engine IDs")
        normalized_source_routes[str(source_id)] = list(route)

    for route in [default_route, *normalized_source_routes.values()]:
        for engine_id in route:
            if engine_id not in normalized_configs:
                raise ValueError(f"route references unconfigured engine {engine_id!r}")
            if not normalized_configs[engine_id]["enabled"]:
                raise ValueError(f"route references disabled engine {engine_id!r}")

    return {
        **graph,
        "contract": CONTRACT,
        "engine_configs": normalized_configs,
        "engine_routes": {**routes, "default": list(default_route),
                          "by_source": normalized_source_routes},
    }


def _valid_route(value: Any) -> bool:
    return (isinstance(value, list)
            and all(isinstance(item, str) and _ENGINE_ID.fullmatch(item) for item in value)
            and len(value) == len(set(value)))


def route_for_source(graph: Mapping[str, Any], source_id: int) -> tuple[str, ...]:
    """Resolve the explicit source override or the graph's default route."""
    routes = graph.get("engine_routes", {})
    route = routes.get("by_source", {}).get(str(source_id), routes.get("default", []))
    return tuple(route)


def validate_settings_schema(settings: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Validate the supported JSON Schema subset used by engine manifests."""
    if not isinstance(settings, Mapping):
        raise ValueError("engine settings must be an object")
    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        raise ValueError("engine settings schema must have type 'object'")
    _validate_schema_definition(schema, "engine settings schema")
    _validate_schema_value(settings, schema, "engine settings")


def _validate_schema_definition(schema: Mapping[str, Any], path: str) -> None:
    unsupported = set(schema) - _SCHEMA_KEYWORDS
    if unsupported:
        keyword = sorted(unsupported)[0]
        raise ValueError(f"unsupported JSON Schema keyword {keyword!r} at {path}")

    if "type" in schema:
        declared = schema["type"]
        types = declared if isinstance(declared, list) else [declared]
        if (not types or any(not isinstance(item, str) or item not in _JSON_TYPES for item in types)
                or len(types) != len(set(types))):
            raise ValueError(f"invalid JSON Schema type at {path}")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValueError(f"enum must be a non-empty array at {path}")
    for keyword in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"):
        if keyword in schema:
            value = schema[keyword]
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or (isinstance(value, float) and not math.isfinite(value))):
                raise ValueError(f"{keyword} must be a finite number at {path}")
            if keyword == "multipleOf" and value <= 0:
                raise ValueError(f"multipleOf must be greater than zero at {path}")
    for keyword in ("minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties"):
        if keyword in schema and (not isinstance(schema[keyword], int) or isinstance(schema[keyword], bool)
                                  or schema[keyword] < 0):
            raise ValueError(f"{keyword} must be a non-negative integer at {path}")
    if "pattern" in schema:
        if not isinstance(schema["pattern"], str):
            raise ValueError(f"pattern must be a string at {path}")
        try:
            re.compile(schema["pattern"])
        except re.error as exc:
            raise ValueError(f"invalid pattern at {path}: {exc}") from exc
    if "uniqueItems" in schema and not isinstance(schema["uniqueItems"], bool):
        raise ValueError(f"uniqueItems must be a boolean at {path}")
    if "required" in schema:
        required = schema["required"]
        if (not isinstance(required, list)
                or any(not isinstance(name, str) for name in required)
                or len(required) != len(set(required))):
            raise ValueError(f"required must be an array of unique strings at {path}")
    if "properties" in schema:
        properties = schema["properties"]
        if not isinstance(properties, Mapping) or any(not isinstance(name, str) for name in properties):
            raise ValueError(f"properties must be an object with string keys at {path}")
        for name, child in properties.items():
            if not isinstance(child, Mapping):
                raise ValueError(f"schema for {path}.{name} must be an object")
            _validate_schema_definition(child, f"{path}.{name}")
    if "items" in schema:
        items = schema["items"]
        if not isinstance(items, Mapping):
            raise ValueError(f"items must be a schema object at {path}")
        _validate_schema_definition(items, f"{path}[]")
    if "additionalProperties" in schema:
        additional = schema["additionalProperties"]
        if isinstance(additional, Mapping):
            _validate_schema_definition(additional, f"{path}.*")
        elif not isinstance(additional, bool):
            raise ValueError(f"additionalProperties must be a boolean or schema at {path}")


def _matches_json_type(value: Any, kind: str) -> bool:
    if kind == "object":
        return isinstance(value, Mapping)
    if kind == "array":
        return isinstance(value, list)
    if kind == "string":
        return isinstance(value, str)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    return value is None


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (left.keys() == right.keys()
                and all(_json_equal(left[key], right[key]) for key in left))
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _validate_schema_value(value: Any, schema: Mapping[str, Any], path: str) -> None:
    declared = schema.get("type")
    if declared is not None:
        types = declared if isinstance(declared, list) else [declared]
        if not any(_matches_json_type(value, kind) for kind in types):
            expected = " or ".join(types)
            raise ValueError(f"{path} must be {expected}")
    if "enum" in schema and not any(_json_equal(value, candidate) for candidate in schema["enum"]):
        raise ValueError(f"{path} must be one of {schema['enum']}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{path} must be a finite number")
        for keyword, comparison, message in (
            ("minimum", lambda actual, bound: actual < bound, "at least"),
            ("maximum", lambda actual, bound: actual > bound, "at most"),
            ("exclusiveMinimum", lambda actual, bound: actual <= bound, "greater than"),
            ("exclusiveMaximum", lambda actual, bound: actual >= bound, "less than"),
        ):
            if keyword in schema and comparison(value, schema[keyword]):
                raise ValueError(f"{path} must be {message} {schema[keyword]}")
        if "multipleOf" in schema:
            try:
                remainder = Decimal(str(value)) % Decimal(str(schema["multipleOf"]))
            except (InvalidOperation, ZeroDivisionError) as exc:
                raise ValueError(f"could not validate multipleOf at {path}") from exc
            if remainder != 0:
                raise ValueError(f"{path} must be a multiple of {schema['multipleOf']}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ValueError(f"{path} must contain at least {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValueError(f"{path} must contain at most {schema['maxLength']} characters")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise ValueError(f"{path} must match pattern {schema['pattern']!r}")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ValueError(f"{path} must contain at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValueError(f"{path} must contain at most {schema['maxItems']} items")
        if schema.get("uniqueItems") and any(
            _json_equal(value[index], value[other])
            for index in range(len(value)) for other in range(index + 1, len(value))
        ):
            raise ValueError(f"{path} must contain unique items")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                _validate_schema_value(item, item_schema, f"{path}[{index}]")

    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for name in required:
            if name not in value:
                raise ValueError(f"missing required engine setting {path}.{name}")
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            raise ValueError(f"{path} must contain at least {schema['minProperties']} properties")
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            raise ValueError(f"{path} must contain at most {schema['maxProperties']} properties")
        additional = schema.get("additionalProperties", True)
        unknown = set(value) - set(properties)
        if additional is False and unknown:
            joined = ", ".join(sorted(map(str, unknown)))
            prefix = "unknown engine settings" if path == "engine settings" else f"{path} has unknown properties"
            raise ValueError(f"{prefix}: {joined}")
        for name, item in value.items():
            child_schema = properties.get(name)
            if child_schema is not None:
                _validate_schema_value(item, child_schema, f"{path}.{name}")
            elif isinstance(additional, Mapping):
                _validate_schema_value(item, additional, f"{path}.{name}")
