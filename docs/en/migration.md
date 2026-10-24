# Architecture migration and upgrades

Use this guide to update an existing Detector/Hub installation and adopt the
Runtime Hub, versioned contracts, and multiple engines.

## What changes during an upgrade

- The audio wire format is unchanged: ESP32 and the USB source service continue
  to send SDR1/UDP packets. The ESP32 does not need new firmware.
- Detector data—clips and `settings.json`—remain in the Docker `clips` volume.
  Rebuilding the container does not remove that volume.
- Existing flat settings are preserved and remain the defaults/fallback. An
  empty `engine_configs` keeps the previous single-engine mode; no manual
  settings migration is needed for a routine upgrade.
- The engine graph is activated only after it is saved in Settings → Engines.
  Once active, the graph and source routes select engines; `--engine` cannot be
  supplied at the same time.
- `audio-frame.v1`, `inference-result.v1`, and `detector-event.v1` are internal
  or Detector output contracts. They do not change the SDR1 wire format.

## Before upgrading

1. Keep current copies of `hub/.env`, `audio_source/.env`, the root `.env`,
   Compose files, and model files. Do not overwrite production `hub/.env` with a different local
   copy: `make deploy` copies the local file to the server when it exists.
   For a quick rollback, tag the current Detector image and record its Compose
   image reference:

   ```bash
   OLD_IMAGE_ID="$(docker inspect siren-detector-engine --format '{{.Image}}')"
   docker inspect siren-detector-engine --format '{{.Config.Image}}'
   docker image tag "$OLD_IMAGE_ID" siren-detector:pre-migration
   ```

   Record the image reference printed by the second command; Compose may use a
   name other than `siren-detector`.
2. To save only the current settings, run this on the server from the directory
   where you want the backup:

   ```bash
   docker cp siren-detector-engine:/clips/settings.json ./settings-before-upgrade.json
   ```

   The file includes legacy settings and `engine_configs`/routes if a graph is
   configured. To also preserve clips and sidecar metadata, archive the full
   `clips` volume. Find its name with:

   ```bash
   docker inspect siren-detector-engine \
     --format '{{range .Mounts}}{{if eq .Destination "/clips"}}{{.Name}}{{end}}{{end}}'
   ```

   Substitute the returned name for `CLIPS_VOLUME` and create an archive:

   ```bash
   docker run --rm \
     -v CLIPS_VOLUME:/data:ro \
     -v "$PWD:/backup" \
     alpine tar czf /backup/clips-backup.tgz -C /data .
   ```

   Also copy `hub/.env`, `audio_source/.env`, and the current Compose files
   outside the directories being updated. `make deploy` and `make deploy-audio`
   use `rsync --delete`; verify local env files contain production values first,
   or the remote copies may be replaced or deleted. Do not remove the Docker
   volume during an upgrade.

## Update the code

**From a development computer.** Configure the root `.env` with `RPI`,
`RPI_BASE`, and `CONTAINER`, then run this from the repository root:

```bash
make deploy
```

This syncs `detector/`, copies the Compose file, copies local `hub/.env` if
present, then rebuilds and starts Detector. To update the USB source after
changing `audio_source/`, run this separately:

```bash
make deploy-audio
```

**On a server with an existing repository checkout.** Update the repository
files using your normal method, then run from its `hub/` directory:

```bash
docker compose -f docker-compose.yml -f docker-compose.build.yml up -d --build detector
docker compose logs -f detector
```

The Hub and ESP32 firmware normally do not need rebuilding because their wire
protocol did not change. After startup, check the web UI, source list, Settings
→ Runtime, and Detector logs.

## Check existing settings

`settings.json` remains in the `clips` volume. The existing single `yamnet`
engine and its thresholds, smoothing, calibration, and source names continue to
work after a routine upgrade. Do not create `engine_configs` if you want to keep
the previous mode.

To use multiple engines, open Settings → Engines and configure engines and
routes. The relevant saved JSON has this form:

```json
{
  "engine_graph_contract": "engine-graph.v1",
  "engine_configs": {
    "yamnet": {
      "selector": "yamnet",
      "enabled": true,
      "settings": {"siren_threshold": 0.1}
    }
  },
  "engine_routes": {
    "default": ["yamnet"],
    "by_source": {"1": ["yamnet"]}
  }
}
```

`settings` contains optional per-engine overrides. If omitted, saved legacy
settings remain the shared defaults. Do not edit the file while Detector is
running; the Settings API writes it itself. Restart Detector after saving the
graph. With an active graph, remove an explicit `--engine` from a custom launch
command; the project's Compose file does not set it by default.

To roll back, stop Detector and restore the previous image/code and
`hub/docker-compose.yml`. Retag the saved image to its previous Compose image
reference, then start the service without rebuilding:

```bash
docker tag siren-detector:pre-migration <IMAGE_REF>
docker compose up -d detector
```

An older Detector ignores unknown fields in the new
`settings.json` while retaining legacy keys. To restore the complete pre-upgrade
state, restore the `clips` volume backup too.

## USB microphone and ALSA

USB audio already works through a separate service rather than capture inside
Detector. `audio_source/` uses `sounddevice`/PortAudio, with ALSA as an option,
converts audio to SDR1/UDP, and sends it to the Hub. On Linux, check the device
with `arecord -l`, set `AUDIO_SOURCE_ID`, `AUDIO_TARGET`, and, if needed,
`AUDIO_DEVICE` in `audio_source/.env`, then run `docker compose up -d` from
`audio_source/`. See [Multi-microphone setup](multi-microphone.md).

The new `FileAudioAdapter` accepts mono 16 kHz WAV for local tests and is
registered programmatically with `RuntimeHub.add_input`; it does not replace
the USB service. A direct ALSA input adapter inside Detector has not been
implemented and is not needed for the current USB path. Actual device detection
depends on Linux, Docker access to `/dev/snd`, and the selected backend; this
repository cannot verify a specific microphone on your Raspberry Pi.

## Interfaces available after upgrading

- `GET /api/runtime/status` reports engine, queue, error, and output status.
- `GET /metrics` serves Prometheus metrics (viewer access is required when auth
  is enabled).
- `GET /events` provides an SSE stream for one-way HTTP clients.
- `/ws`, MQTT, and existing retained Home Assistant topics remain compatible;
  events also include source and engine identifiers.
- Webhooks remain disabled unless `WEBHOOK_URLS` is set in Detector's
  environment.
