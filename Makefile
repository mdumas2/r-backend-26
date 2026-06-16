CC ?= gcc
CFLAGS_EXTRA ?=
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -march=haswell -mavx2 -mfma -flto $(CFLAGS_EXTRA)
LDFLAGS ?= -pthread -lm -flto

# ── Epoll/blocking API server ──────────────────────────────────
# Built from a single main.c (original approach, no split needed)
API_BIN := rinha-c-api
API_SRC := src/main.c

# ── io_uring API server ───────────────────────────────────────
# engine.o = main.c compiled without main() (exports engine functions)
# api_uring.o = io_uring event loop + main()
URING_BIN := rinha-c-api-uring
URING_SRC := src/api_uring.c
URING_OBJ := api_uring.o
ENGINE_OBJ := engine.o

# ── Load balancer ─────────────────────────────────────────────
LB_BIN := rinha-c-lb
LB_SRC := src/lb.c

.PHONY: all clean

all: $(API_BIN) $(URING_BIN) $(LB_BIN)

# Original API: single-file compile (all static, includes main())
$(API_BIN): $(API_SRC) src/json_simd.h
	$(CC) $(CFLAGS) -o $@ $(API_SRC) $(LDFLAGS)

# Engine: main.c without main(), functions exported for io_uring
$(ENGINE_OBJ): $(API_SRC) src/json_simd.h
	$(CC) $(CFLAGS) -DRINHA_C_NO_MAIN -c -o $@ $<

# io_uring object
$(URING_OBJ): $(URING_SRC) src/json_simd.h
	$(CC) $(CFLAGS) -c -o $@ $<

# io_uring binary
$(URING_BIN): $(URING_OBJ) $(ENGINE_OBJ)
	$(CC) $(CFLAGS) -o $@ $(URING_OBJ) $(ENGINE_OBJ) $(LDFLAGS) -luring

# Load balancer
$(LB_BIN): $(LB_SRC)
	$(CC) $(CFLAGS) -o $@ $(LB_SRC) $(LDFLAGS)

clean:
	rm -f $(API_BIN) $(URING_BIN) $(LB_BIN) $(ENGINE_OBJ) $(URING_OBJ)
