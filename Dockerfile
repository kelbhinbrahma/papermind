# Render (free web service) / any Docker host. Runtime only: no torch; the trained topic model + prebuilt corpus ship in the image.
FROM python:3.12-slim
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user papermind papermind
COPY --chown=user frontend frontend
COPY --chown=user data/papermind.db data/topic_model.joblib data/
USER user
ENV PYTHONUNBUFFERED=1
EXPOSE 10000
# Render injects $PORT; default for local `docker run -p 10000:10000`
CMD uvicorn papermind.app:app --host 0.0.0.0 --port ${PORT:-10000}
