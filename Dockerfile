FROM python:3.8

ARG PACKAGES="ffmpeg build-essential"

ARG DEBIAN_FRONTEND=noninteractive
RUN \
    apt-get update && \
    apt-get install -qq $PACKAGES && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install poetry

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false
RUN poetry install --no-root

COPY . ./

ENV PYTHONPATH "${PYTHONPATH}:/app"

