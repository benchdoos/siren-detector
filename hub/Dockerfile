FROM python:3.11-slim
RUN useradd -r -s /usr/sbin/nologin appuser
WORKDIR /app
COPY hub.py .
USER appuser
ENTRYPOINT ["python", "hub.py"]
