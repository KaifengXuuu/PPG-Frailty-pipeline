FROM python:3.9-slim-bullseye


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc gfortran \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY pyproject.toml README.md /app/

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install .


COPY . /app

EXPOSE 8050

ENV APP_HOST=0.0.0.0 \
    APP_PORT=8050

CMD ["sh", "-c", "gunicorn", "-w", "2", "-b", "0.0.0.0:8050", "ppg:server"]
