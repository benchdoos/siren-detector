"""Versioned event-policy contract and loader for local implementations."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol

from .contracts import InferenceResult

if TYPE_CHECKING:
    from .hub import RuntimeHub


@dataclass(frozen=True)
class EventPolicyContext:
    """Runtime services provided when an event policy is constructed."""

    runtime: RuntimeHub
    get_source: Callable[[int], Any]
    sources: dict[int, Any]
    dsp: Any = None
    settings: dict[str, Any] | None = None


class EventPolicy(Protocol):
    """Consumer contract for inference results routed by Runtime Hub."""

    policy_id: str
    api_version: str

    def process(self, result: InferenceResult) -> None:
        """Apply policy and publish score/event messages via its context runtime."""
        ...


BUILTIN_EVENT_POLICIES: dict[str, str] = {
    "default": "detector.engine.event_policy.EventPolicy",
}


def load_event_policy(name: str, context: EventPolicyContext) -> EventPolicy:
    """Instantiate a built-in policy name or a locally installed dotted class."""
    dotted_path = BUILTIN_EVENT_POLICIES.get(name, name)
    module_path, separator, class_name = dotted_path.rpartition(".")
    if not separator:
        available = ", ".join(sorted(BUILTIN_EVENT_POLICIES))
        raise ValueError(
            f"Unknown event policy {name!r}; built-ins: {available}. "
            "Custom policies must use package.module.ClassName."
        )
    module = importlib.import_module(module_path)
    policy_class = getattr(module, class_name)
    policy = policy_class(context)
    if not callable(getattr(policy, "process", None)):
        raise TypeError(f"Event policy {dotted_path!r} must implement process(result)")
    if not getattr(policy, "policy_id", None):
        raise TypeError(f"Event policy {dotted_path!r} must define policy_id")
    if getattr(policy, "api_version", None) != "event-policy.v1":
        raise TypeError(
            f"Event policy {dotted_path!r} must declare api_version='event-policy.v1'"
        )
    return policy
