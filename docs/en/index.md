# Siren Detector

An open-source, non-commercial project for detecting civil-alert sirens, explosions, and nearby uavs used as a proxy for UAV-like sound. An ESP32-S3 audio node streams microphone audio to a Linux host, where the detector analyzes it and exposes events through a web interface and optional MQTT integration.

> [!WARNING]
> This project and all its components are intended exclusively for peaceful purposes. Any military, harmful, or destructive use is prohibited.

## What is in the repository

- **ESP32 firmware** captures INMP441 audio and sends it over UDP.
- **Hub** relays audio packets to the detector and optional recorder.
- **Detector** runs YAMNet and an optional project-trained MLP, maintains per-source event state, and serves the web UI and API.
- **Dataset and ML tools** support recording, annotation, augmentation, training, and evaluation.

The current deployment is a remote inference setup: the ESP32 captures and transports audio; inference runs on a Linux host such as a Raspberry Pi. Standalone ESP32 inference is a future direction, not the current operating mode.

## Start here

- [Getting started](getting-started.md) — configure and run the Docker services.
- [Architecture](architecture.md) — audio flow and service responsibilities.
- [Component API](component-api.md) — target contracts for sources, routing, processing, and outputs.
- [Custom engine example](custom-engine-example.md) — runnable example using Runtime Hub.
- [Operations](operations.md) — deployment, settings, API, and troubleshooting.
- [Data and models](data-and-models.md) — collect data and retrain the classifiers.
- [Multi-microphone setup](multi-microphone.md) — add a USB audio source or understand the current ESP32 dual-mic wiring.
- [Siren Detector settings](settings.md) — detailed runtime options and modes.
- [API](api.md) — REST routes, Swagger UI, OpenAPI, and WebSocket.

## Project facts

- Sample rate: 16 kHz, mono.
- Transport: UDP with the SDR1 packet header.
- Default ports: Hub `9000`, recorder `9001`, detector UDP `9002`, web UI `8080`.
- License: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

The detector is an attention aid, not a certified warning system. Do not rely on it as the only means of safety-critical notification.
