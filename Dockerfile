FROM python:3.11-slim

WORKDIR /app

# Install deps first for layer caching.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# App code, web UI, sample documents, and eval harness.
COPY backend/ backend/
COPY frontend/ frontend/
COPY samples/ samples/
COPY eval/ eval/

EXPOSE 8000

# Keys are injected at run time via --env-file .env (never baked into the image).
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
