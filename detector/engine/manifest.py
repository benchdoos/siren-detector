"""Manifest for the built-in YAMNet + MLP detection engine."""

from ..runtime.manifest import EngineManifest


YAMNET_ENGINE_MANIFEST = EngineManifest(
    engine_id="yamnet",
    name="YAMNet + MLP",
    version="1.0.0",
    description=(
        "YAMNet audio scores and embeddings with the optional project MLP; "
        "an optional DSP pre-filter can skip background windows."
    ),
    input_contracts=("audio-frame.v1",),
    output_contracts=("inference-result.v1",),
    classes=("siren", "explosion", "uav"),
    settings_schema={
        "type": "object",
        "properties": {
            "siren_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.10},
            "explosion_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.15},
            "uav_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.15},
            "yamnet_min_gate": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.01},
            "inference_interval_seconds": {"type": "number", "minimum": 0.25, "maximum": 2, "default": 0.5},
            "dsp_enabled": {"type": "integer", "enum": [0, 1], "default": 1},
            "dsp_bg_threshold": {"type": "number", "minimum": 0.5, "maximum": 1, "default": 0.92},
            "dsp_bg_release_threshold": {"type": "number", "minimum": 0.5, "maximum": 1, "default": 0.90},
            "dsp_event_gate": {"type": "number", "minimum": 0, "maximum": 0.5, "default": 0.30},
        },
        "additionalProperties": True,
    },
    resources={
        "accelerators": ["cpu"],
        "cpu_cores": None,
        "memory_mb": None,
        "notes": "Measure on the target device; parallel source count increases CPU and memory use.",
    },
    launch={
        "kind": "python-class",
        "entrypoint": "detector.engine.yamnet_engine.YAMNetEngine",
        "selector": "--engine yamnet",
    },
)
