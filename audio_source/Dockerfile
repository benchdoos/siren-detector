FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends libportaudio2 libpulse0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY audio_source.py .

ENTRYPOINT ["python", "audio_source.py"]
