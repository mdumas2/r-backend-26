FROM debian:12-slim AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates liburing-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Makefile ./
COPY src ./src
RUN make rinha-c-api rinha-c-api-uring rinha-c-lb

FROM debian:12-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends liburing2 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /src/rinha-c-api /app/rinha-c-api
COPY --from=build /src/rinha-c-api-uring /app/rinha-c-api-uring
COPY refs.bin kdtree_v2.bin /app/
ENV RINHA_REFS_BIN=/app/refs.bin \
    RINHA_KDTREE_V2_BIN=/app/kdtree_v2.bin \
    RINHA_SOCKET=/sockets/api.sock \
    RINHA_C_EPOLL_MODE=1 \
    RINHA_C_EPOLL_IDLE_US=60 \
    RINHA_C_BUSY_POLL_US=100 \
    RINHA_C_WARMUP_QUERIES=50000 \
    RINHA_C_KD_CHROMATIC=1 \
    RINHA_C_KD_PREFETCH_DIST=0 \
    RINHA_C_INSTRUMENT=1 \
    RINHA_C_INSTRUMENT_INTERVAL_SECS=10 \
    MALLOC_ARENA_MAX=1 \
    MALLOC_TRIM_THRESHOLD_=131072
ENTRYPOINT ["/app/rinha-c-api"]
