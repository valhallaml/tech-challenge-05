FROM python:3.13.5-slim

WORKDIR /app

RUN addgroup \
        --gid 1000 \
        app \
    && adduser \
        --uid 1000 \
        --gid 1000 \
        --disabled-password \
        --quiet \
        app \
    && chown app:app /app/ \
    && chmod -R 700 /app

USER app

ENV PATH="/home/app/.local/bin:${PATH}"
ENV PYTHONPATH=/app/src

COPY --chown=app:app src/data/ src/data/
COPY --chown=app:app requirements.txt requirements.txt

RUN pip install \
    --user \
    --no-cache-dir \
    --upgrade \
    -r requirements.txt \
    && rm -f requirements.txt

COPY --chown=app:app --chmod=0500 src src

CMD [ "uvicorn", "main:app", "--host=0.0.0.0", "--port", "8000" ]

EXPOSE 8000
