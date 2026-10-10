FROM node:24-bookworm-slim@sha256:2fe369e969550cde8e867afc3fe370b260140cab4a23d467074295b42163d553 AS node_runtime
FROM lesson-renderer:0.1.0-manim0.20.1
USER root
COPY --from=node_runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=node_runtime /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx
RUN apt-get update && apt-get install -y --no-install-recommends libreoffice-impress fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /opt/workspace-requirements.txt
RUN pip install --no-cache-dir -r /opt/workspace-requirements.txt \
    && npm install --global pptxgenjs@4.0.1 react@18.3.1 react-dom@18.3.1 react-icons@5.5.0 sharp@0.34.5
RUN npm install --prefix /opt/exoharness-typescript --ignore-scripts --no-audit --no-fund \
    typescript@5.9.3 tsx@4.20.6 @types/node@24.3.0 @types/react@18.3.24 @types/react-dom@18.3.7 \
    && ln -s /opt/exoharness-typescript/node_modules/.bin/tsc /usr/local/bin/tsc \
    && ln -s /opt/exoharness-typescript/node_modules/.bin/tsx /usr/local/bin/tsx \
    && node /opt/exoharness-typescript/node_modules/tsx/dist/cli.mjs --version \
    && tsc --version
COPY typescript-tsconfig.json /opt/exoharness-typescript/tsconfig.json
ENV NODE_PATH=/usr/local/lib/node_modules PYTHONPATH=/skill HOME=/tmp
RUN useradd --uid 65532 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin skillworker
RUN apt-get update && apt-get install -y --no-install-recommends poppler-utils libreoffice-calc \
    && rm -rf /var/lib/apt/lists/*
RUN npm install --global --ignore-scripts --no-audit --no-fund docx@9.7.1 \
    && apt-get update && apt-get install -y --no-install-recommends libreoffice-writer pandoc \
    && rm -rf /var/lib/apt/lists/*
ENTRYPOINT []
CMD ["python", "/runner/sandbox.py"]
