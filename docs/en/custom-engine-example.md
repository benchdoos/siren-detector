# Custom engine example

The repository includes a minimal engine at
`detector/examples/energy_threshold_engine.py`. It demonstrates integration
boundaries, not recognition quality: it calculates frame RMS energy and maps it
to the `siren` score slot so the result appears in the current UI. It is **not a
siren detector**.

## Message path

```text
UDPSourceAdapter → AudioFrame → Runtime Hub → EnergyThresholdEngine
                                      ↑                 │
                                      └── score/event ──┘
                                                ↓
                                   EventBus → WebSocket / MQTT
```

The engine subscribes to normalized `audio-frame.v1` frames. It publishes a
score for each frame through Runtime Hub. On a threshold transition it emits an
event and asks the Hub to save a short clip through the configured storage
adapter, which lets the current UI render the event card. The engine does not
open UDP or publish directly to UI or MQTT.

## Run it

From the repository root, start Detector with the example's dotted class path:

```sh
python -m detector.main \
  --engine detector.examples.energy_threshold_engine.EnergyThresholdEngine \
  --energy-event-threshold 0.08 \
  --energy-release-threshold 0.05
```

Other Detector launch options and audio input configuration remain the same.
The example declares a manifest, which can be inspected with:

```sh
python -m detector.main \
  --engine detector.examples.energy_threshold_engine.EnergyThresholdEngine \
  --show-engine-manifest
```

If `settings.json` contains `energy_event_threshold`,
`energy_release_threshold`, or `energy_clip_seconds`, those values take priority
over the CLI options in legacy mode. In graph mode, these manifest-defined
settings appear in Settings → Engines and apply after the detector restarts.

When the input amplitude crosses the threshold, the `siren` score bar rises and
an event card appears after a WAV clip is saved. The same score/event messages
flow through the existing WebSocket and MQTT outputs. The engine label is
`energy-threshold-example`.

## Starting point for an engine

An engine class implements `start(runtime, settings, args)`. It can declare an
`ENGINE_MANIFEST`, implement `add_arguments(parser)` for CLI options, and
subscribe to audio:

```python
runtime.subscribe_audio(on_audio_frame)
```

Publish a score like this:

```python
from detector.runtime.contracts import RuntimeMessage

runtime.publish(RuntimeMessage.from_legacy({
    "type": "score",
    "source_id": frame.source_id,
    "source": "my microphone",
    "top": "My model",
    "siren_score": 0.82,
    "siren_state": "possible",
    "siren_triggers": 1,
    "siren_possible_needed": 1,
    "siren_confirmed_needed": 3,
}))
```

An event uses `type="event"` with `event`, `state`, `confidence`, `timestamp`,
`source_id`, `source`, `engine`, and `clip_url`. The current UI only renders an
event card when `clip_url` points to a saved clip. For compatibility, publish
class names known to the current UI (`siren`, `explosion`, `uav`); arbitrary
classes and manifest settings are not configurable in the web UI yet.

> Engine imports run in the main Detector process. Only connect local code you
> trust; installing Python plugins through the web UI is not supported.
