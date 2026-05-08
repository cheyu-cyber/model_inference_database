FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Mirror the host layout so `from DB.model_inference_database.X import …`
# resolves the same way it does on the host (DB/ and model_inference_database/
# are namespace packages, no __init__.py needed).
COPY events/    DB/model_inference_database/events/
COPY messaging/ DB/model_inference_database/messaging/
COPY services/  DB/model_inference_database/services/
COPY web/       DB/model_inference_database/web/

ENV PYTHONPATH=/app
WORKDIR /app/DB/model_inference_database
