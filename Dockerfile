# Engine Build
FROM rust:1.94-slim-bookworm AS engine
ARG TARGET_CPU=x86-64-v3
WORKDIR /src/engine
COPY engine/Cargo.toml engine/Cargo.lock ./
COPY engine/src src
RUN RUSTFLAGS="-C target-cpu=${TARGET_CPU}" cargo build --release --locked

# Bot Runtime
FROM python:3.13-slim-bookworm
RUN pip install --no-cache-dir numpy==2.5.5 onnxruntime==1.32.0 \
    && useradd --create-home --uid 1000 kestrel
COPY --from=engine /src/engine/target/release/kestrel /usr/local/bin/kestrel
COPY python/kestrel /app/kestrel
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KESTREL_ENGINE=/usr/local/bin/kestrel \
    KESTREL_BOT_CONFIG=/config/bot.toml
USER kestrel
VOLUME /config
STOPSIGNAL SIGINT
ENTRYPOINT ["python", "-m", "kestrel.lichess_bot"]
