FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/detector/requirements.txt
RUN pip install --no-cache-dir -r /app/detector/requirements.txt
COPY . /app/detector/

ENTRYPOINT ["python", "-m", "detector.main"]
