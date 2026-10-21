"""Stored engine-graph.v1 configuration and validation helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


CONTRACT = "engine-graph.v1"
EMPTY_GRAPH = {
    "contract": CONTRACT,
    "engine_configs": {},
    "engine_routes": {"default": [], "by_source": {}},
}
_ENGINE_ID = re.compile(r"^[a-z][a-z0-9._-]*$")


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
    """Validate the JSON Schema subset used by engine manifests."""
    if schema.get("type") != "object":
        raise ValueError("engine settings schema must have type 'object'")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        raise ValueError("engine settings schema has invalid properties or required fields")
    for name in required:
        if name not in settings:
            raise ValueError(f"missing required engine setting {name!r}")
    if schema.get("additionalProperties", True) is False:
        unknown = set(settings) - set(properties)
        if unknown:
            raise ValueError(f"unknown engine settings: {', '.join(sorted(unknown))}")
    for name, value in settings.items():
        spec = properties.get(name)
        if spec is None:
            continue
        kind = spec.get("type")
        valid_type = {
            "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
            "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
            "string": lambda x: isinstance(x, str),
            "boolean": lambda x: isinstance(x, bool),
            "array": lambda x: isinstance(x, list),
            "object": lambda x: isinstance(x, Mapping),
        }.get(kind)
        if valid_type and not valid_type(value):
            raise ValueError(f"engine setting {name!r} must be {kind}")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"engine setting {name!r} must be one of {spec['enum']}")
        for bound in ("minimum", "maximum"):
            if bound in spec and isinstance(value, (int, float)):
                if bound == "minimum" and value < spec[bound]:
                    raise ValueError(f"engine setting {name!r} must be at least {spec[bound]}")
                if bound == "maximum" and value > spec[bound]:
                    raise ValueError(f"engine setting {name!r} must be at most {spec[bound]}")
