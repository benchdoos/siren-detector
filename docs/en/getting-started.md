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

Copy `esp32/siren-detector.yaml` and your `secrets.yaml` to the ESPHome host. Validate, compile, then flash the device with ESPHome. The default microphone wiring is documented in [Multi-microphone setup](multi-microphone.md).

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
- Use [Data and models](data-and-models.md) to record examples and retrain the classifier.
