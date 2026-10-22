<h1 align="center">
  <a href="https://github.com/benchdoos/siren-detector"><img src=".github/images/siren-detector-logo.png" alt="Siren Detector" width="200"></a>
  <br>
  Siren Detector
</h1>

<p align="right"><b>English</b> • <a href="README.ru.md">Русский</a></p>

Open-source project for detecting civil-alert sirens, explosions, and nearby UAV sounds. An ESP32-S3 captures audio; a Linux host runs inference and exposes events through a web interface and optional MQTT.

> [!WARNING]
> This project and all its components are intended exclusively for peaceful purposes. Military, harmful, or destructive use is prohibited.

## Documentation

**[Read the English documentation](https://benchdoos.github.io/siren-detector/)** for setup, architecture, operation, data and model training, and multi-microphone configuration. The docs are also available in [Russian](https://benchdoos.github.io/siren-detector/ru/).

## Quick start

Requirements: ESP32-S3 + INMP441 microphone, a Linux host with Docker Compose, and an ESPHome installation.

1. Configure the environment and device secrets:

   ```bash
   cp .env.example .env
   cp hub/.env.example hub/.env
   cp esp32/secrets.yaml.example esp32/secrets.yaml
   ```

   Set the host address and credentials in the relevant files.

2. Copy `esp32/siren-detector.yaml` and `secrets.yaml` to your ESPHome host, then validate, compile, and flash the device.
3. Start the Hub and detector on the Linux host:

   ```bash
   cd hub
   docker compose up -d hub detector
   ```

4. Open `http://<host-ip>:8080`. On first start, YAMNet is downloaded and cached.

For deployment, configuration, MQTT, and retraining details, see the [documentation](docs/en/index.md). To preview the full bilingual documentation locally, install `docs/requirements.txt` and run `mkdocs serve`.

## Project layout

- `esp32/` — ESPHome firmware and microphone capture.
- `hub/` — UDP audio relay and Docker Compose deployment.
- `detector/` — runtime result routing, inference engine, API, web UI (i18n: EN/RU), and integrations.
- `audio_source/`, `recorder/` — USB microphone streaming and audio recording.
- `dataset/`, `ml/` — data preparation, training, and evaluation tools.
- `docs/` — bilingual project documentation.

## License

Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You may copy, share, and adapt the project for non-commercial purposes with appropriate attribution and a link to the [original repository](https://github.com/benchdoos/siren-detector). The software is provided “as is”, without warranty.
