"""Registry and compatibility loader for built-in and local Python engines."""

from __future__ import annotations

import importlib
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

from ..runtime.manifest import EngineManifest


@dataclass(frozen=True)
class EngineRegistration:
    """A named engine selector and its Python class entrypoint."""

    selector: str
    entrypoint: str


class EngineRegistry:
    """Resolve named registrations or legacy dotted class paths."""

    def __init__(self) -> None:
        self._registrations: dict[str, EngineRegistration] = {}

    def register(self, selector: str, entrypoint: str) -> None:
        """Register an engine selector before startup.

        Third-party packages may call this during application bootstrap. A
        process started directly with the built-in CLI can also use an
        unregistered dotted class path as a migration-compatible selector.
        """
        if not re.fullmatch(r"[a-z][a-z0-9._-]*", selector):
            raise ValueError("Registered engine selectors must be stable lowercase IDs")
        if selector in self._registrations:
            raise ValueError(f"Engine selector {selector!r} is already registered")
        if "." not in entrypoint:
            raise ValueError("Engine entrypoint must use package.module.ClassName")
        self._registrations[selector] = EngineRegistration(selector, entrypoint)

    def selectors(self) -> tuple[str, ...]:
        """Return the currently registered short selectors."""
        return tuple(sorted(self._registrations))

    def resolve(self, selector: str) -> tuple[type, EngineManifest | None]:
        """Import and validate an engine class and its optional manifest."""
        registration = self._registrations.get(selector)
        entrypoint = registration.entrypoint if registration else selector
        module_path, separator, class_name = entrypoint.rpartition(".")
        if not separator:
            available = ", ".join(self.selectors())
            raise ValueError(
                f"Unknown engine {selector!r}; registered engines: {available}. "
                "Custom engines may use package.module.ClassName."
            )

        module = importlib.import_module(module_path)
        engine_class = getattr(module, class_name)
        if not callable(getattr(engine_class, "start", None)):
            raise TypeError(f"Engine {entrypoint!r} must implement start(runtime, settings, args)")

        manifest = self._read_manifest(engine_class)
        if manifest is None:
            if registration is not None:
                raise ValueError(
                    f"Registered engine {registration.selector!r} must declare ENGINE_MANIFEST; "
                    "use a dotted class path only for legacy engines"
                )
            warnings.warn(
                f"Engine {entrypoint!r} has no ENGINE_MANIFEST; loading via legacy interface",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            if registration and manifest.engine_id != registration.selector:
                raise ValueError(
                    f"Registered selector {registration.selector!r} does not match "
                    f"manifest engine_id {manifest.engine_id!r}"
                )
            declared_entrypoint = manifest.launch.get("entrypoint")
            if declared_entrypoint and declared_entrypoint != entrypoint:
                raise ValueError(
                    f"Manifest entrypoint {declared_entrypoint!r} does not match "
                    f"registered entrypoint {entrypoint!r}"
                )
        return engine_class, manifest

    @staticmethod
    def _read_manifest(engine_class: type) -> EngineManifest | None:
        manifest = getattr(engine_class, "ENGINE_MANIFEST", None)
        if isinstance(manifest, EngineManifest):
            return manifest
        if isinstance(manifest, Mapping):
            return EngineManifest(**manifest)
        if manifest is not None:
            raise TypeError("ENGINE_MANIFEST must be an EngineManifest or mapping")
        return None


engine_registry = EngineRegistry()
engine_registry.register("yamnet", "detector.engine.yamnet_engine.YAMNetEngine")
