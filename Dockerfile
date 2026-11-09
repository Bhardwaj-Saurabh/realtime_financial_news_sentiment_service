FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
# The CPU index keeps PyTorch from pulling in CUDA libraries.
COPY requirements-api.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-api.txt

COPY starter/app ./app

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "app/main.py"]
