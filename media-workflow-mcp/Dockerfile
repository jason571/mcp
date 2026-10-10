FROM python:3.12.11-slim-bookworm

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /tmp/media-workflow-requirements.lock
RUN pip install --no-cache-dir --requirement /tmp/media-workflow-requirements.lock \
    && rm -f /tmp/media-workflow-requirements.lock

WORKDIR /opt/media-workflow
COPY src ./src
ENV PYTHONPATH=/opt/media-workflow/src \
    MEDIA_MCP_DATA_DIR=/var/lib/media-workflow

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "media_workflow_mcp"]
