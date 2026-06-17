FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pré-télécharge les modèles (VAD silero, turn-detector) pour éviter de le faire
# au premier appel et réduire la latence de démarrage.
COPY src/ ./src/
RUN python -m src.agent download-files || true

CMD ["python", "-m", "src.agent", "start"]
