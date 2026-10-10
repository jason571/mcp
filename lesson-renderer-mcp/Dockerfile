FROM python:3.12.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg tini gcc g++ pkg-config libcairo2-dev libpango1.0-dev \
    fonts-noto-cjk tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.lock /tmp/lesson-requirements.lock
RUN pip install --no-cache-dir -r /tmp/lesson-requirements.lock \
    && useradd --uid 10001 --create-home lesson
ENV PYTHONPATH=/opt/lesson-renderer/src PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 XDG_CACHE_HOME=/tmp/cache MPLCONFIGDIR=/tmp/matplotlib
USER 10001:10001
WORKDIR /tmp
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "lesson_renderer.server"]
