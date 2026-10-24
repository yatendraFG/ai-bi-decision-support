# One image, two apps:
#   dashboard (default):  docker run -p 8501:8501 ai-bi
#   API:                   docker run -p 8000:8000 ai-bi uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# Easiest: docker compose up   (starts both, see docker-compose.yml)

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# 1) dependencies first -> this layer is cached, rebuilds after code changes are fast
COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

# 2) code + trained models + outputs (produced by run_pipeline.py on the laptop)
COPY src/ src/
COPY app/ app/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/

# run as a non-root user (safer if the container is ever exposed to the internet)
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request,os; urllib.request.urlopen('http://localhost:' + os.environ.get('HEALTH_PORT','8501') + os.environ.get('HEALTH_PATH','/_stcore/health'))"

CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
