FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY detector.py .
COPY classifier_infer.py .
COPY static/ static/

ENTRYPOINT ["python", "detector.py"]
