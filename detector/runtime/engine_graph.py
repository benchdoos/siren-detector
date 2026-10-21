"""Stored engine-graph.v1 configuration and validation helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from jsonschema import exceptions as jsonschema_exceptions
from jsonschema.validators import validator_for
from referencing.exceptions import Unresolvable


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
    """Validate settings against the JSON Schema draft declared by the manifest."""
    if not isinstance(settings, Mapping):
        raise ValueError("engine settings must be an object")
    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        raise ValueError("engine settings schema must have type 'object'")
    validator_class = validator_for(schema)
    try:
        validator_class.check_schema(schema)
    except jsonschema_exceptions.SchemaError as exc:
        raise ValueError(f"invalid engine settings schema: {exc.message}") from exc

    validator = validator_class(schema)
    try:
        errors = sorted(
            validator.iter_errors(dict(settings)),
            key=lambda error: list(map(str, error.absolute_path)),
        )
    except Unresolvable as exc:
        raise ValueError(f"cannot resolve JSON Schema reference: {exc}") from exc
    if errors:
        error = errors[0]
        path = "engine settings"
        if error.absolute_path:
            path += "." + ".".join(map(str, error.absolute_path))
        raise ValueError(f"{path}: {error.message}")
