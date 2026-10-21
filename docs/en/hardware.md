# Hardware setup

## ESP32-S3 and INMP441

The audio node uses an ESP32-S3 development board with one or two INMP441 MEMS microphones. The firmware is built with ESPHome and flashed over USB or OTA.

### Bill of materials

| Part | Quantity | Notes |
|---|---:|---|
| ESP32-S3 dev board (N16R8 or similar) | 1 | Must have PSRAM for ESPHome; most N16R8 boards work. |
| INMP441 breakout | 1–2 | Second mic is optional; improves SNR by ~3 dB when averaged. |
| 100 nF ceramic capacitor | 1 per mic | Between VDD and GND, as close to the mic as possible. Reduces supply noise and impulse clicks. |
| Jumper wires | ~6 per mic | For breadboard prototyping. |
| USB-C cable | 1 | For initial flash; OTA updates work over Wi-Fi afterward. |

### Wiring

Each INMP441 connects to its own I2S bus on the ESP32-S3. The firmware supports two microphones; the second is optional.

**Microphone 1** (I2S bus 0):

| INMP441 pin | ESP32-S3 GPIO | Notes |
|---|---:|---|
| SCK (BCLK) | GPIO4 | Bit clock |
| WS (LRCLK) | GPIO5 | Word select / left-right clock |
| SD (DOUT) | GPIO6 | Data out |
| L/R | GND | Left channel (tie to GND) |
| VDD | 3.3 V | |
| GND | GND | |

**Microphone 2** (I2S bus 1, optional):

| INMP441 pin | ESP32-S3 GPIO | Notes |
|---|---:|---|
| SCK (BCLK) | GPIO15 | Bit clock |
| WS (LRCLK) | GPIO16 | Word select |
| SD (DOUT) | GPIO17 | Data out |
| L/R | GND | Left channel |
| VDD | 3.3 V | |
| GND | GND | |

No MCLK wire is required — the INMP441 generates its clock internally from BCLK.

### Capacitor placement

Place a 100 nF (0.1 uF) ceramic capacitor between VDD and GND on each microphone, directly at the mic breakout pins. Without this capacitor, the INMP441 is more susceptible to supply noise that manifests as impulse clicks in the audio. The detector includes a click removal filter, but clean hardware reduces the problem at the source.

### How the two microphones work

When both microphones are connected, the firmware averages their samples into one SDR1 stream (source ID `0`). This improves signal-to-noise ratio by approximately 3 dB. The second microphone does **not** appear as an independent source in the detector — it is a single mixed stream.

If the second microphone is disconnected or silent (RMS below 0.0001), the firmware sends only the first microphone's data. No configuration change is needed.

### Power and placement notes

- The INMP441 operates on 1.8–3.3 V. Use the ESP32-S3 board's 3.3 V output.
- Keep wires between the mic and ESP32 short to reduce noise pickup on the I2S lines.
- Mount the microphone with the sound port facing the area of interest. Enclosures should leave the port open.
- INMP441 sends 24-bit audio packed in 32-bit I2S slots. The firmware shifts right by 8 bits and normalizes to the 24-bit range before transmitting.

## Flashing the firmware

1. Copy `esp32/secrets.yaml.example` to `esp32/secrets.yaml` and fill in your Wi-Fi credentials and the Hub host IP:

   ```yaml
   wifi_ssid: "YourSSID"
   wifi_password: "YourPassword"
   ap_password: "fallback-password"
   udp_target_ip: "192.168.1.100"
   udp_target_port: "9000"
   ```

2. First flash — connect the ESP32-S3 via USB and run:

   ```bash
   esphome run esp32/siren-detector.yaml
   ```

   Select the serial port when prompted.

3. Subsequent updates use OTA (no USB needed):

   ```bash
   esphome run esp32/siren-detector.yaml --device siren-detector.local
   ```

   Or use `make deploy-firmware` which copies the config to the RPi ESPHome host and compiles/flashes there.

The device starts the `microphone_capture` switch 5 seconds after boot and begins sending SDR1/UDP packets to the configured Hub address.

### Fallback AP

If the configured Wi-Fi network is unavailable, the ESP32 creates a fallback access point named "Siren Detector Fallback" with the password from `secrets.yaml`. Connect to it and use the captive portal at `192.168.4.1` to reconfigure Wi-Fi.

## USB microphone (alternative)

A USB microphone on the Linux host is an alternative or addition to the ESP32. See [Multi-microphone setup](multi-microphone.md) for the `audio_source/` service configuration.
