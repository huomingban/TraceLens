# The same Linux image is used by the HTTP API, Alembic migration job, and
# RocketMQ worker.  RocketMQ's Python client does not support Windows, which
# is why the worker belongs in this image rather than the local Windows setup.
ARG IMAGE_REGISTRY=docker.m.daocloud.io
FROM ${IMAGE_REGISTRY}/library/python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/backend

ARG DEBIAN_MIRROR=mirrors.aliyun.com

# FFmpeg is used for media inspection/transcoding. The RocketMQ Python client
# links to the Apache C++ runtime installed below. Retries matter on networks
# where a package CDN occasionally returns 502 during a long first build.
RUN set -eux; \
    if [ "$DEBIAN_MIRROR" != "deb.debian.org" ]; then \
        for source_file in /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources; do \
            if [ -f "$source_file" ]; then \
                sed -i \
                    -e "s|https\?://deb.debian.org/debian-security|https://${DEBIAN_MIRROR}/debian-security|g" \
                    -e "s|https\?://deb.debian.org/debian|https://${DEBIAN_MIRROR}/debian|g" \
                    "$source_file"; \
            fi; \
        done; \
    fi; \
    installed=0; \
    for attempt in 1 2 3; do \
        if apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 update \
            && apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 install -y --no-install-recommends ca-certificates curl ffmpeg; then \
            installed=1; break; \
        fi; \
        rm -rf /var/lib/apt/lists/*; sleep $((attempt * 3)); \
    done; \
    test "$installed" = 1; \
    rm -rf /var/lib/apt/lists/*

ARG ROCKETMQ_CLIENT_VERSION=2.0.0
ARG ROCKETMQ_CLIENT_SHA256=d8a97b5aed30559a6bffe846835f0de39c6cb3f051b9ef665e461e1111ddd785
RUN curl --fail --location --retry 3 --retry-all-errors --connect-timeout 10 --max-time 180 \
        --output /tmp/rocketmq-client.deb \
        "https://github.com/apache/rocketmq-client-cpp/releases/download/${ROCKETMQ_CLIENT_VERSION}/rocketmq-client-cpp-${ROCKETMQ_CLIENT_VERSION}.amd64.deb" \
    && echo "${ROCKETMQ_CLIENT_SHA256}  /tmp/rocketmq-client.deb" | sha256sum --check --strict \
    && dpkg --install /tmp/rocketmq-client.deb \
    && ldconfig \
    && rm -f /tmp/rocketmq-client.deb

COPY requirements.txt requirements-rocketmq.txt ./
RUN pip install -r requirements.txt -r requirements-rocketmq.txt

COPY . ./

RUN mkdir -p /app/data/uploads /app/data/models

EXPOSE 9090

CMD ["uvicorn", "tracelens.main:app", "--host", "0.0.0.0", "--port", "9090"]
