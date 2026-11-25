FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN useradd --create-home --uid 10001 appuser
COPY pyproject.toml README.md ./
COPY src/ ./src/
ARG MODEL_EXTRA=cnn
RUN case "$MODEL_EXTRA" in \
      cnn) python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu ;; \
      transfer) python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu ;; \
      classical) ;; \
      *) exit 2 ;; \
    esac \
    && if [ "$MODEL_EXTRA" = classical ]; then pip install .; else pip install ".[${MODEL_EXTRA}]"; fi
COPY app/ ./app/
COPY .streamlit/config.toml ./.streamlit/config.toml
USER appuser
ENV CROP_MODEL_RUN=/models/plantvillage
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=4s CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"]
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
