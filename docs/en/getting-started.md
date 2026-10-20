# Getting started

## Requirements

- ESP32-S3 and one INMP441 microphone (the second microphone is optional).
- A Linux host for Docker Compose; Raspberry Pi is the documented deployment target.
- An ESPHome installation to compile and flash the firmware.
- Network access from the ESP32 to the host running the Hub.

## 1. Configure the project

From the repository root:

```bash
cp .env.example .env
cp hub/.env.example hub/.env
cp esp32/secrets.yaml.example esp32/secrets.yaml
```

Set the target host address and UDP port in the ESPHome secrets file. Configure the SSH target and web URL in the root `.env` if you plan to use the Makefile deployment commands. The two environment files serve different purposes: the root file configures Make targets; `hub/.env` configures Docker Compose.

Keep real Wi-Fi credentials, admin tokens, and passwords out of Git.

## 2. Flash the ESP32

From the repository root on a machine with the ESPHome CLI installed, validate and compile the configuration:

```bash
esphome config esp32/siren-detector.yaml
esphome compile esp32/siren-detector.yaml
```

Keep `esp32/secrets.yaml` beside the YAML configuration. It must define `wifi_ssid`, `wifi_password`, `ap_password`, `udp_target_ip`, and `udp_target_port` (see `esp32/secrets.yaml.example`). For the first flash, connect the ESP32-S3 over USB and run:

```bash
esphome run esp32/siren-detector.yaml
```

Select the device's serial port when ESPHome prompts you. Later updates can use OTA when the device is online. If compiling on another host, copy both files and preserve their relative locations. See [Multi-microphone setup](multi-microphone.md) for wiring and the [settings reference](settings.md) for firmware and detector options.

## 3. Start the Hub and detector

On the Linux host, from the project deployment directory:

```bash
cd hub
docker compose up -d hub detector
```

The Hub listens for ESP32 audio on UDP port `9000` and forwards packets to the detector on `9002`. The detector web interface is served on port `8080`.

Open `http://<host-ip>:8080`. On first start, YAMNet model files are downloaded and cached; allow network access and time for the initial startup.

## 4. Check the audio path

Confirm that the ESP32 is online and sending packets, then check the detector's source and audio status in the web UI. For deployment logs, use `make logs` from a configured development checkout or `docker compose logs -f hub detector` on the host.

## Local development

The detector can also run directly with Python. Install dependencies from `detector/requirements.txt`, then start `python -m detector.main`; the default UDP port is `9000`, web port `8080`. If running the Hub separately, set the detector UDP port to `9002` as in `hub/docker-compose.yml`.

## Next steps

- Add a USB microphone using [Multi-microphone setup](multi-microphone.md).
- Review [Operations](operations.md) for settings, MQTT, and deployment commands.
- Read the [detector settings reference](settings.md) and the [HTTP/WebSocket API reference](api.md).
- Use [Data and models](data-and-models.md) to record examples and retrain the classifier.
