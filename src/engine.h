/*
 * engine.h — Shared KD-tree fraud detection engine.
 *
 * This header contains ALL code from main.c except:
 *   - main()
 *   - blocking worker loop
 *   - epoll event loop
 *
 * Both main.c (epoll/blocking server) and api_uring.c (io_uring server)
 * include this to get the same engine.
 *
 * When RINHA_C_USE_SIMD_JSON is defined, the SIMD-accelerated JSON
 * parser from json_simd.h is used instead of the scalar SAX parser.
 */

#ifndef ENGINE_H
#define ENGINE_H

/* ═══════════════════════════════════════════════════════════════════
 * Everything below is extracted verbatim from main.c lines 1-2408
 * (constants, data structures, KD-tree, search, parse, vectorize,
 *  decide_frauds, instrumentation, etc.)
 *
 * The original main.c will #include "engine.h" and provide only
 * the serving layer (main, workers, epoll loop).
 *
 * api_uring.c will also #include "engine.h" and provide its
 * io_uring event loop instead.
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * NOTE: This file is a declaration of the extraction strategy.
 * The actual implementation is done by having main.c keep its
 * current code intact, and api_uring.c linking against the same
 * object file for the engine functions.
 *
 * The Makefile compiles main.c once with -DRINHA_C_NO_MAIN to
 * produce engine.o, then both main.o (with main) and api_uring.o
 * link against engine.o.
 */

#endif /* ENGINE_H */
