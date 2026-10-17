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

## Option 2: Second INMP441 on ESP32 (Stereo)

Two INMP441 microphones can share a single I2S bus by using the left and
right channels.

### Wiring

| Pin       | Mic 1 (Left) | Mic 2 (Right) |
|-----------|-------------|---------------|
| SCK/BCLK  | GPIO4       | GPIO4         |
| WS/LRCLK  | GPIO5       | GPIO5         |
| SD/DOUT   | GPIO6       | GPIO6         |
| L/R       | **GND**     | **VDD**       |
| VDD       | 3.3V        | 3.3V          |
| GND       | GND         | GND           |

Both mics share the same clock and data lines.  The L/R pin determines
which channel each mic responds on: GND = left, VDD = right.

### Firmware

The base `esp32/siren-detector.yaml` captures the **left channel**
(source_id=0).  To add the right channel as a second source:

1. Copy the `microphone:` and related sections or create a second YAML
   that extends the base via `packages:`.

2. Add a second microphone component for the right channel:

   ```yaml
   substitutions:
     source_id: "0"
     second_source_id: "1"

   microphone:
     # Primary mic (left channel, source_id=0)
     - platform: i2s_audio
       id: inmp441_left
       adc_type: external
       i2s_din_pin: GPIO6
       i2s_audio_id: i2s_bus
       channel: left
       sample_rate: 16000
       bits_per_sample: 32bit
       use_apll: true
       on_data:
         - lambda: |-
             // ... (same as base, source_id = ${source_id})

     # Secondary mic (right channel, source_id=1)
     - platform: i2s_audio
       id: inmp441_right
       adc_type: external
       i2s_din_pin: GPIO6
       i2s_audio_id: i2s_bus
       channel: right
       sample_rate: 16000
       bits_per_sample: 32bit
       use_apll: true
       on_data:
         - lambda: |-
             // Same packet-sending code, but with:
             const uint16_t reserved = htons(${second_source_id});
   ```

3. Update the detector to name both sources:
   ```bash
   --source-name 0=esp32-left --source-name 1=esp32-right
   ```

> **Note:** If only one INMP441 is connected, the unused channel will
> produce silence.  The detector will register it but won't generate events
> (scores stay at 0).

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
