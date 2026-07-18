FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Requirements copied ahead of the source so the dependency layer — which
# includes the slow xgboost/shap wheels — survives ordinary code edits.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Cloud Run injects PORT and expects the container to listen on it; the default
# only covers local `docker run` without an explicit port.
ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
