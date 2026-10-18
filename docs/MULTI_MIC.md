# Multi-Microphone Support

The detector supports multiple audio sources — each with its own inference
pipeline (buffers, smoothers).  YAMNet and MLP classifier are loaded once
and shared across all sources.

## Architecture

```
ESP32 (source_id=0) ──UDP──┐
                            │
USB mic (source_id=1) ─UDP──┼──→ Hub ──→ Detector (per-source inference)
                            │               ↓
ESP32 #2 (source_id=2) ─UDP┘         Web UI / MQTT / clips
```

Every audio source sends SDR1 packets with a unique `source_id` in the
header (bytes 10–11, previously "reserved").  The detector automatically
registers new sources on first packet and creates an independent inference
pipeline for each.

**Round-robin inference**: sources are processed one at a time in a loop,
keeping CPU budget constant regardless of the number of mics.

## SDR1 Protocol

```
Offset  Size  Field
  0      4    magic       = 0x53445231 ("SDR1")
  4      4    sequence    (uint32, network byte order)
  8      2    payload_size (uint16, network byte order)
 10      2    source_id   (uint16, network byte order)  ← was "reserved"
 12      N    PCM payload (int32 LE, I2S format)
```

`source_id = 0` is the default.  Old firmware with `reserved = 0` works
unchanged — the detector treats it as source 0.

## Source Naming

Map `source_id` to human-readable names at detector startup:

```bash
python -m detector.main \
  --source siren-detector-01 \
  --source-name 0=esp32-main \
  --source-name 1=rpi-usb \
  --source-name 2=esp32-aux
```

- `--source` sets the default name for source 0 (backwards compatible)
- `--source-name ID=NAME` explicitly names a source (repeatable)
- Unknown source IDs are auto-named `source-{id}`

In Docker Compose, add `--source-name` args to the detector command:

```yaml
command:
  - --bind
  - 127.0.0.1
  - --port
  - "9002"
  - --source
  - siren-detector-01
  - --source-name
  - 0=esp32-main
  - --source-name
  - 1=rpi-usb
```

---

## Option 1: USB Microphone on Raspberry Pi

### Docker (recommended)

1. Plug a USB microphone into the Raspberry Pi.

2. Verify the device is recognized:
   ```bash
   arecord -l
   ```

3. Deploy and start:
   ```bash
   # From Mac:
   make deploy-audio

   # Or manually on RPi:
   cd ~/Projects/esphome/audio_source
   docker compose up -d
   ```

4. Configure via `audio_source/.env` (optional):
   ```
   AUDIO_SOURCE_ID=1
   AUDIO_TARGET=127.0.0.1:9000
   AUDIO_SOURCE_GAIN=1.0
   ```

5. Add `--source-name 1=rpi-usb` to the detector command in
   `hub/docker-compose.yml` so events show a readable source name.

### Python (without Docker)

1. Install dependencies:
   ```bash
   pip install sounddevice==0.5.1 numpy==1.26.4
   ```

2. List available audio devices:
   ```bash
   python audio_source/audio_source.py --list-devices
   ```

3. Start streaming:
   ```bash
   python audio_source/audio_source.py \
     --device 0 \
     --source-id 1 \
     --target 127.0.0.1:9000 \
     --gain 1.0
   ```

   The `--device` flag accepts a device index (from `--list-devices`) or a
   substring of the device name (e.g. `"USB"` to match "USB Audio Device").

### Troubleshooting

- **No sound**: check `arecord -l`, try a different `--device` index.
- **Audio too quiet/loud**: adjust `--gain` (e.g. `--gain 3.0`).
- **"PortAudio error"**: ensure `libportaudio2` is installed
  (`apt install libportaudio2`).
- **Docker: "No such device"**: make sure `/dev/snd` exists and the user
  has access (`usermod -aG audio $USER`).

---

## Option 2: Second INMP441 on ESP32 (Separate I2S Bus)

Two INMP441 microphones use **separate I2S peripherals** on the ESP32-S3.
Each mic has its own GPIO pins — just dupont wires, no breadboard or
soldering needed.

> **Why not stereo (L/R)?** ESPHome validates GPIO pin uniqueness and
> does not allow two microphone components to share the same `i2s_din_pin`.

### Wiring

| Mic pin   | Mic 1 → ESP32 | Mic 2 → ESP32 (optional) |
|-----------|---------------|--------------------------|
| SCK/BCLK  | GPIO4         | GPIO15                   |
| WS/LRCLK  | GPIO5         | GPIO16                   |
| SD/DOUT   | GPIO6         | GPIO17                   |
| L/R       | GND           | GND                      |
| VDD       | 3.3V          | 3.3V                     |
| GND       | GND           | GND                      |

Each mic connects with dupont wires directly to its own set of GPIO
pins on the ESP32 dev board.  No shared signal lines.

Add a **100 nF ceramic capacitor** between VDD and GND on each mic to
reduce power supply noise.

### Firmware

The base `esp32/siren-detector.yaml` already includes both microphones.
The second mic is **optional** — if not physically connected, the I2S bus
reads silence and the firmware skips sending UDP packets (RMS below noise
floor threshold).  The ESP32 will not crash or reboot.

```yaml
substitutions:
  source_id: "0"           # primary mic (GPIO4/5/6)
  second_source_id: "1"    # secondary mic (GPIO15/16/17, optional)
```

To use different GPIO pins, edit the `i2s_bus_2` and `inmp441_2`
sections in the YAML.

Update the detector to name both sources:
```bash
--source-name 0=esp32-main --source-name 1=esp32-aux
```

> **Note:** If only one INMP441 is connected, the second I2S bus reads
> silence.  The firmware detects this (RMS < 0.0001) and skips sending
> packets, so the detector won't register a phantom source.

---

## Option 3: Any Custom Audio Source

Any device or script that sends valid SDR1 packets can act as an audio
source.  Requirements:

1. **PCM format**: 16 kHz sample rate, mono, 32-bit signed integers (I2S
   format: audio in upper 24 bits, lower byte zero).
2. **SDR1 header**: 12-byte header with magic, sequence, payload_size,
   and a unique source_id.
3. **UDP transport**: send packets to the Hub port (default 9000) or
   directly to the detector port.

See `audio_source/audio_source.py` for a reference implementation.

---

## Hub

The Hub (UDP fan-out relay) requires **no changes**.  It transparently
forwards all packets, including the `source_id` field in the SDR1 header.

If a new audio source runs on the same host as the Hub, it can send
packets directly to `127.0.0.1:9000` (the Hub will relay to all
forwarding targets including the detector).

## MQTT

When events are triggered, the MQTT integration publishes the source name
to `sound_detector/{detector}/source` so Home Assistant can distinguish
which microphone triggered the event.
