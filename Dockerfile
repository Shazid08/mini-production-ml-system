# Churn scoring service: FastAPI app (default) and batch CLI in one image.
# Base image pinned to an exact patch release AND its content digest, so every build
# starts from byte-identical base layers.
FROM python:3.12.7-slim-bookworm@sha256:60d9996b6a8a3689d36db740b49f4327be3be09a21122bd02fb8895abb38b50d

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# LightGBM's compiled library links against GCC's OpenMP runtime (libgomp.so.1),
# which the slim image does not include. Without it, "import lightgbm" fails.
# Not version-pinned: Debian removes superseded package versions from its mirrors,
# so an exact pin would eventually break the build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so this layer is cached when only the code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Only what the service and the batch CLI need at runtime.
COPY src/ src/
COPY serving/ serving/
COPY predict_batch.py .
COPY artifacts/churn_model.joblib artifacts/churn_model.joblib
COPY data/sample_input.csv data/sample_input.csv

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
