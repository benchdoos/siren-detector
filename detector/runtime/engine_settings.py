"""Resolve per-engine settings while preserving legacy hot-reload behavior."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .. import settings as settings_mod


def settings_for_engine(runtime: Any, supplied: Mapping[str, Any]) -> dict[str, Any]:
    """Return settings scoped to one engine graph entry.

    Legacy mode keeps the shared dictionary so the existing Settings API can
    update running smoothers and inference behavior. Graph mode takes a copy
    and overlays the entry-specific values without mutating process defaults.
    """
    if getattr(runtime, "engine_graph_active", False):
        return deepcopy({**settings_mod.settings, **supplied})

    settings_mod.settings.update(supplied)
    return settings_mod.settings
