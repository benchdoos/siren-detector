"""Versioned manifest describing a runnable detection engine."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping


@dataclass(frozen=True)
class EngineManifest:
    """Discovery and configuration metadata for one engine implementation."""

    engine_id: str
    name: str
    version: str
    description: str
    input_contracts: tuple[str, ...]
    output_contracts: tuple[str, ...]
    classes: tuple[str, ...]
    settings_schema: Mapping[str, Any]
    resources: Mapping[str, Any]
    launch: Mapping[str, Any]
    contract: Literal["engine-manifest.v1"] = "engine-manifest.v1"

    def __post_init__(self) -> None:
        if self.contract != "engine-manifest.v1":
            raise ValueError("Unsupported engine manifest contract")
        for field_name in ("engine_id", "name", "version", "description"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        if not re.fullmatch(r"[a-z][a-z0-9._-]*", self.engine_id):
            raise ValueError("engine_id must be a stable lowercase identifier")
        if not re.fullmatch(r"\d+\.\d+\.\d+", self.version):
            raise ValueError("version must use MAJOR.MINOR.PATCH format")
        if not self.input_contracts:
            raise ValueError("At least one input contract is required")
        if not self.output_contracts:
            raise ValueError("At least one output contract is required")
        if self.settings_schema.get("type") != "object":
            raise ValueError("settings_schema must be a JSON Schema object")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable manifest data."""
        data = asdict(self)
        data["input_contracts"] = list(self.input_contracts)
        data["output_contracts"] = list(self.output_contracts)
        data["classes"] = list(self.classes)
        return data
