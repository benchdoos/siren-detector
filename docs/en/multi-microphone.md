# Multi-microphone setup

The detector can receive independent audio sources over UDP. Each source has its own ID, buffers, inference state, event history, and source name. The Hub forwards source packets without changing their IDs.

## USB microphone on the Linux host

The repository's `audio_source/` service captures a local microphone and sends it to the Hub.

1. Connect the microphone and confirm that Linux recognizes it (`arecord -l`).
2. Set `AUDIO_SOURCE_ID=1` and `AUDIO_TARGET=127.0.0.1:9000` in `audio_source/.env` if needed.
3. Start it on the host with `cd audio_source && docker compose up -d`.
4. Ensure the detector command includes `--source-name 1=rpi-usb`. The provided `hub/docker-compose.yml` already includes this mapping.

To run without Docker, install the requirements from `audio_source/requirements.txt` and start `audio_source/audio_source.py`; use `--list-devices` to find an input device.

## Two INMP441 microphones on one ESP32

The ESP32-S3 firmware has two separate I2S buses:

| Signal | Microphone 1 | Microphone 2 |
|---|---:|---:|
| BCLK / SCK | GPIO4 | GPIO15 |
| LRCLK / WS | GPIO5 | GPIO16 |
| Data | GPIO6 | GPIO17 |
| L/R | GND | GND |
| Power | 3.3 V | 3.3 V |

The firmware currently **mixes (averages) the two microphone signals** and transmits one SDR1 stream with source ID `0`. The second microphone does not appear as an independent source in the detector, and toggling it cannot create a second source entry. If it is disconnected or silent, the first microphone continues to send audio. This is a single mixed source design, not source-level multi-mic inference.

The ESPHome `microphone_capture` switch starts and stops both microphones together. The optional second microphone must be wired to GPIO15/16/17. A 100 nF capacitor between VDD and GND at each mic can help reduce supply noise.

## SDR1 source IDs

The 12-byte packet header stores the source ID as an unsigned 16-bit big-endian field at byte offset 10. ID `0` is the default ESP32 source. The detector creates a source entry after receiving its first valid packet and names unknown IDs `source-<id>` unless configured with repeatable `--source-name ID=NAME` arguments.

For a custom source, use 16 kHz mono audio, SDR1 framing, monotonically increasing packet sequence numbers, and a unique ID. Send to the Hub UDP port (default `9000`) so it reaches all configured destinations.

## Round-robin and parallel inference

Independent sources can be processed in `round_robin` mode (one loop rotates between sources) or `parallel` mode (one inference loop per source). Parallel work may increase CPU and memory use. These modes apply to independent SDR1 source IDs; the two microphones mixed by the current ESP32 firmware remain one source either way.

The dashboard combines independent sources into one score bar: it shows the highest current score and the most urgent state (`confirmed` → `possible` → `idle`). The score and state can therefore come from different microphones; the bar label identifies the source contributing the score. When DSP skips YAMNet for a source, that source's last score fades to zero over three seconds until a new model score is available.
