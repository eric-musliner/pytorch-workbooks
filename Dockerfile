FROM python:3.13 AS base
WORKDIR /opt/torch

COPY pyproject.toml poetry.lock ./

RUN pip install --upgrade pip \
    && pip install --no-cache-dir "poetry==2.*" \
    && poetry config virtualenvs.create false \
    && poetry install -n --no-ansi --no-root
