FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; slim image + light Python deps = small footprint.
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# App code and corpus. chroma_db is intentionally NOT copied - the app builds
# its index at startup from data/ (see ensure_indexed in app/main.py).
COPY app ./app
COPY data ./data

# Render provides $PORT at runtime; default to 8000 for local docker runs.
ENV PORT=8000
ENV RETRIEVER=hybrid
ENV EMBED_MODEL=openai

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]