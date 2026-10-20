# Siren Detector settings

Detector settings are available from the **Settings** button in the web UI and through `/api/settings`. Changes take effect after saving and are persisted to `/clips/settings.json` (the `clips` Docker volume). Reset restores defaults from `detector/config.py`.

## Processing modes

`detection_mode` controls how multiple active audio sources are processed:

| Value | Behavior | When to use |
|---|---|---|
| `round_robin` (default) | One loop processes sources in turn. | Lower peak CPU use; suitable for a smaller host. |
| `parallel` | Each source has its own inference loop. | Less per-source waiting, with higher CPU and memory use. |

Inference runs on the Detector host, not the ESP32. `inference_interval_seconds` sets the time between passes for a source: `0.25–2.0` seconds, default `0.5`. A longer interval reduces load but can increase latency and miss short sounds.

## Detection thresholds

| Setting | Default | Range | Purpose |
|---|---:|---:|---|
| `siren_threshold` | `0.10` | `0–1` | Minimum siren score to trigger. When the MLP is active, this applies to its output. Lower values increase sensitivity and false positives. |
| `explosion_threshold` | `0.15` | `0–1` | YAMNet threshold for Explosion, Burst/pop, and Gunshot/gunfire classes. |
| `uav_threshold` | `0.15` | `0–1` | Threshold for the Motorcycle class, used as a proxy for UAV engine sounds. |
| `yamnet_min_gate` | `0.01` | `0–1` | Minimum YAMNet score required before trusting the siren MLP prediction. `0` disables this check. |

## DSP pre-filter

An optional DSP model can skip YAMNet when it confidently identifies background audio. Model files must be present in the model directory; availability is shown in Settings and `/api/settings`. Settings can be overridden per source.

| Setting | Default | Range | Purpose |
|---|---:|---:|---|
| `dsp_enabled` | `1` (on) | `0–1` | Enables the pre-filter; it is unavailable when model files are missing. |
| `dsp_bg_threshold` | `0.92` | `0.5–1` | Background confidence required to skip YAMNet. Higher values skip more conservatively. |
| `dsp_bg_release_threshold` | `0.90` | `0.5–1` | YAMNet resumes when background confidence falls below this value. |
| `dsp_bg_confirm_windows` | `2` | `1–10` | Consecutive background windows required before skipping begins. |
| `dsp_max_consecutive_skips` | `20` | `0–200` | Maximum consecutive skips before forcing a YAMNet pass; `0` disables the limit. |
| `dsp_event_gate` | `0.30` | `0–0.5` | If an event class probability exceeds this threshold, YAMNet runs despite the background score. `0` disables this gate. |

`dsp_source_overrides` is an object keyed by source ID. The Sources panel can enable or disable DSP and set per-source thresholds; unspecified values inherit the global settings.

## Smoothing and events

| Setting | Default | Range | Purpose |
|---|---:|---:|---|
| `possible_triggers` | `4` | `1–60` | Above-threshold windows needed for the early “possible” status. |
| `confirmed_triggers` | `8` | `1–60` | Above-threshold windows needed to confirm an event, save a clip, and publish the event. A value greater than `possible_triggers` is recommended. |
| `smoothing_window_seconds` | `30` | `5–120` | Rolling duration over which trigger scores are accumulated. |
| `cooldown_seconds` | `60` | `0–600` | Pause after confirmation to avoid duplicate events. `0` disables the pause. |
| `refractory_seconds` | `15` | `0–120` | After cooldown, wait for the score to drop before re-arming; the detector becomes ready when this timeout expires. `0` disables the wait. |

## Click removal

This filter replaces short impulse spikes with interpolated values before inference. Per-source overrides are stored in `click_removal_source_overrides`.

| Setting | Default | Range | Purpose |
|---|---:|---:|---|
| `click_removal_enabled` | `1` (on) | `0–1` | Enables removal of short clicks. |
| `click_removal_threshold` | `10.0` | `3–50` | Outlier threshold in MADs; lower values filter more aggressively. |
| `click_removal_max_width` | `5` | `1–20` samples | Maximum spike-group length to remove; longer signals are preserved. |

## Sources

| Setting | Type | Purpose |
|---|---|---|
| `disabled_sources` | list of IDs | Disabled sources do not produce events, although audio continues to be received. |
| `source_custom_names` | object `ID → name` | Custom source names shown in the UI. |

Source, DSP, and click-removal controls are available in the Sources panel. Source data is also available from [`GET /api/sources`](api.md).

## Audio clips and labeling

| Setting | Default | Range | Purpose |
|---|---:|---:|---|
| `clip_seconds` | `10` | `5–60` sec | Length of a confirmed-event clip. The in-memory buffer resizes when this changes. |
| `clip_from_possible` | `0` (off) | `0–1` | When enabled, save from the early possible state with a two-second pre-buffer instead of the last N seconds before confirmation. |
| `user_event_clip_seconds` | `15` | `5–60` sec | Clip length for a manually created event. |
| `save_background_segments` | `false` | `true/false` | Save unmarked gaps as background examples when labeling segments. |

## Updating settings through the API

Read current values and defaults with `GET /api/settings`. Update only the fields you need using `POST /api/settings` with a JSON object and admin access. For example:

```json
{"siren_threshold": 0.12, "detection_mode": "round_robin"}
```

The server validates supported ranges. Reset through `POST /api/settings/reset`. See the [API reference](api.md) for route details.
