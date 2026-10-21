"""Shared engine startup and failure isolation for runtime components."""

from __future__ import annotations

from argparse import Namespace
from typing import Any


def has_explicit_engine_option(argv: list[str]) -> bool:
    """Return whether argv explicitly selects a legacy single engine."""
    return any(argument == "--engine" or argument.startswith("--engine=")
               for argument in argv)


def run_engine_component(
    runtime: Any,
    engine_id: str,
    engine_class: type,
    settings: dict,
    args: Namespace,
) -> None:
    """Run one engine and record its terminal state without affecting peers."""
    try:
        engine_class().start(runtime, settings, args)
    except Exception as exc:
        runtime.set_component_state(engine_id, "error", exc)
        print(f"[engine:{engine_id}] failed: {type(exc).__name__}: {exc}", flush=True)
    else:
        runtime.set_component_state(engine_id, "ready")
