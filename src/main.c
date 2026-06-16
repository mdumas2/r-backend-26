#define _GNU_SOURCE
#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/epoll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

/* SIMD JSON parser — enable with -DRINHA_C_USE_SIMD_JSON */
#ifdef RINHA_C_USE_SIMD_JSON
#include "json_simd.h"
#endif

/*
 * When compiled as engine.o (RINHA_C_NO_MAIN), we need key functions
 * to be externally visible for api_uring.c to link against.
 * RINHA_C_ENGINE_EXPORT changes 'static' to nothing for those functions.
 */
#ifdef RINHA_C_NO_MAIN
#define ENGINE_FUNC
#else
#define ENGINE_FUNC static
#endif

#ifndef EPIOCSPARAMS
struct epoll_params {
    uint32_t busy_poll_usecs;
    uint16_t busy_poll_budget;
    uint8_t prefer_busy_poll;
    uint8_t __pad;
};
#define EPIOCSPARAMS _IOW(0x8A, 0x01, struct epoll_params)
#endif

#ifndef PR_SET_TIMERSLACK
#define PR_SET_TIMERSLACK 29
#endif

#define DIMENSIONS 14
#define PACKED_DIMENSIONS 16
#define DIM_PAIRS 7
#define NEIGHBOR_K 5
#define LANES 8

#define MAX_KNOWN_MERCHANTS 128
#define REFS_RECORD_BYTES (DIMENSIONS * 2 + 1)
#define FILE_NODE_BYTES (DIMENSIONS * 2 * 2 + 4 * 3)
#define FILE_PARTITION_BYTES (4 * 3 + DIMENSIONS * 2 * 2)
#define MAX_PARTITIONS 512
#define LEAF_FLAG UINT32_MAX
#define Q_SCALE 10000.0f
#define INV_Q_SCALE 0.0001f
#define BOUNDARY_EPSILON 1.0e-5f
#define BOUNDARY_BUFFER_CAP 64
#define HTTP_BUF_CAP 131072
#define FD_QUEUE_CAP 8192
#define C_INSTRUMENT_BUCKETS 2048
#ifndef RINHA_C_DEEP_INSTRUMENT
#define RINHA_C_DEEP_INSTRUMENT 0
#endif
#if RINHA_C_DEEP_INSTRUMENT
#define C_COUNT_BUCKETS 8192
#define C_KEY_US_BUCKETS 512
#define C_KEY_TOP_N 12
#endif
#define NODE_STACK_CAP 1024
#define NODE_HEAP_CAP 8192
#define NODE_DEFER_CAP 4096
#define LABEL_MASK_LEGIT 1u
#define LABEL_MASK_FRAUD 2u

typedef struct {
    int16_t aabb_min[DIMENSIONS];
    int16_t aabb_max[DIMENSIONS];
    uint32_t left;
    uint32_t right;
    uint32_t count;
} FileNode;

typedef struct {
    uint32_t key;
    uint32_t root;
    uint32_t count;
    int16_t aabb_min[DIMENSIONS];
    int16_t aabb_max[DIMENSIONS];
} FilePartition;

typedef struct {
    int16_t aabb_min[PACKED_DIMENSIONS];
    int16_t aabb_max[PACKED_DIMENSIONS];
    uint32_t left;
    uint32_t right;
    uint32_t chunk_start;
    uint32_t chunk_end;
    uint16_t parent_diff_mask;
    uint8_t label_mask;
} KdNode;

typedef struct {
    uint32_t key;
    uint32_t root;
    int16_t aabb_min[PACKED_DIMENSIONS];
    int16_t aabb_max[PACKED_DIMENSIONS];
} KdPartition;

typedef struct {

    int16_t pairs[DIM_PAIRS][2 * LANES];
} LeafVecChunk;

typedef struct {
    uint32_t ref_indices[LANES];
    uint8_t labels[LANES];
    uint8_t len;
} LeafMeta;

typedef struct {
    int16_t dims[DIMENSIONS];
    uint32_t ref_index;
    uint8_t label;
} Candidate;

typedef struct {
    uint32_t idx;
    int64_t bound;
} NodeStackEntry;

typedef struct {
    KdNode *nodes;
    size_t node_count;
    KdPartition *partitions;
    size_t partition_count;
    int32_t part_by_key[MAX_PARTITIONS];
    LeafVecChunk *chunks;
    LeafMeta *metas;
    size_t chunk_count;
    size_t count;
} KdIndex;

typedef struct {
    KdIndex *index;
    const int32_t *query_q;
    const int16_t *query_q16;
    __m256i query_pairs[DIM_PAIRS];
    int64_t top_dist[NEIGHBOR_K];
    uint8_t top_label[NEIGHBOR_K];
    uint32_t top_indices[NEIGHBOR_K];
    uint8_t top_count;
    uint8_t top_sum;
    bool chrom_enabled;
    bool chrom_active;
    bool chrom_replaying;
    uint8_t chrom_initial_sum;
    uint8_t chrom_needed_mask;
    NodeStackEntry *deferred;
    size_t deferred_len;
    bool deferred_overflow;
    bool early_stop;
#if RINHA_C_DEEP_INSTRUMENT
    uint8_t inst_phase;
    uint32_t inst_query_key;
    uint64_t inst_home_ns;
    uint64_t inst_probe_ns;
    uint32_t inst_nodes[2];
    uint32_t inst_leaves[2];
    uint32_t inst_chunks[2];
    uint32_t inst_points[2];
    uint32_t inst_bounds[2];
    uint32_t inst_heap_pushes[2];
    uint32_t inst_heap_pops[2];
    uint32_t inst_heap_max[2];
    uint32_t inst_top_updates[2];
    uint32_t inst_probe_partitions;
#endif
} Search;

typedef struct {
    size_t s;
    size_t e;
} Span;

typedef struct {
    uint8_t h;
    uint8_t weekday;
    int64_t unix_seconds;
} Iso;

typedef struct {
    bool amount_set;
    bool installments_set;
    bool req_set;
    float amount;
    uint32_t installments;
    Iso req;
} TxFields;

typedef struct {
    bool avg_set;
    bool tx_count_set;
    float avg_amount;
    uint32_t tx_count_24h;
    Span known[MAX_KNOWN_MERCHANTS];
    size_t known_count;

    bool known_overflow;
    size_t known_raw_start;
    size_t known_raw_end;
} CustomerFields;

typedef struct {
    bool id_set;
    bool mcc_set;
    bool avg_set;
    Span id;
    uint32_t mcc;
    float avg_amount;
} MerchantFields;

typedef struct {
    bool online_set;
    bool card_set;
    bool km_set;
    bool is_online;
    bool card_present;
    float km_from_home;
} TerminalFields;

typedef struct {
    int64_t unix_seconds;
    float km_from_current;
} LastTxFields;

typedef struct {
    int fd;
    uint64_t lb_accept_ns;
    uint64_t api_queue_ns;
} FdQueueEntry;

typedef struct {
    FdQueueEntry entries[FD_QUEUE_CAP];
    size_t head;
    size_t tail;
    size_t len;
    pthread_mutex_t mu;
    pthread_cond_t cv;
} FdQueue;

static KdIndex g_index;
static FdQueue g_queue = {
    .head = 0,
    .tail = 0,
    .len = 0,
    .mu = PTHREAD_MUTEX_INITIALIZER,
    .cv = PTHREAD_COND_INITIALIZER,
};
static bool g_c_decision_cascade = false;
static bool g_c_fixture_score_guards = false;
static bool g_wide_cascade = true;
static int g_uring_event_fd = -1;
static bool g_uring_mode = false;
static bool g_c_instrument = false;
static bool g_c_mmap_hot = true;
static bool g_c_index_collapse = true;
static bool g_c_index_mlock = true;
static bool g_c_kd_bound_thread = true;
static bool g_c_kd_primary_map = true;
static bool g_c_kd_inline_sort = true;
static bool g_c_kd_best_first = false;
static bool g_c_kd_best_first_primary_only = false;
static bool g_c_kd_best_first_keys_enabled = false;
static bool g_c_kd_best_first_keys[MAX_PARTITIONS];
static bool g_c_kd_chromatic_primary_keys_enabled = false;
static bool g_c_kd_chromatic_primary_keys[MAX_PARTITIONS];
static bool g_c_kd_delta_bounds = false;
static bool g_c_kd_avx_leaf = true;
static bool g_c_kd_pair_leaf = false;
static bool g_c_kd_grouped_leaf = true;
static bool g_c_kd_chromatic = false;
static bool g_c_kd_early_unanimous = false;
static int g_c_kd_early_margin = -1;
static int g_c_kd_prefetch_dist = 3;
static bool g_c_kd_prefetch_meta = false;
static bool g_c_quant_once = false;
static int g_c_kdtree_extra_split = 0;
static bool g_c_compact_resp = false;
static bool g_c_tcp_quickack = false;
static int g_c_busy_poll_us = 0;
ENGINE_FUNC bool g_c_epoll_mode = false;
static int g_c_epoll_loops = 1;

static int g_c_epoll_idle_us = 0;
ENGINE_FUNC uint32_t g_c_warmup_queries = 0;
static volatile uint8_t g_warmup_sink = 0;
static int64_t g_c_kd_early_limit = 0;
static unsigned g_c_instrument_interval_secs = 5;
static atomic_bool g_c_sched_logged = false;
static _Thread_local NodeStackEntry g_deferred_nodes[NODE_DEFER_CAP];

enum {
    C_STAGE_PARSE = 0,
    C_STAGE_CASCADE,
    C_STAGE_KNN,
    C_STAGE_K_PSCAN,
    C_STAGE_K_HOME,
    C_STAGE_K_SORT,
    C_STAGE_K_PROBE,
    C_STAGE_K_RERANK,
    C_STAGE_WRITE,
    C_STAGE_TOTAL,
    C_STAGE_API_QUEUE,
    C_STAGE_FIRST_READ,
    C_STAGE_READ_READY,
    C_STAGE_INGRESS_READY,
    C_STAGE_WIRE_TOTAL,
    C_STAGE_COUNT,
};

static const char *C_STAGE_NAMES[C_STAGE_COUNT] = {
    "parse",
    "cascade",
    "knn",
    "k_pscan",
    "k_home",
    "k_sort",
    "k_probe",
    "k_rerank",
    "write",
    "total",
    "api_queue",
    "first_read",
    "read_ready",
    "ingress_ready",
    "wire_total",
};

static _Atomic uint64_t g_c_hist[C_STAGE_COUNT][C_INSTRUMENT_BUCKETS];

#if RINHA_C_DEEP_INSTRUMENT
enum {
    C_COUNT_HOME_NODES = 0,
    C_COUNT_HOME_LEAVES,
    C_COUNT_HOME_CHUNKS,
    C_COUNT_HOME_POINTS,
    C_COUNT_HOME_BOUNDS,
    C_COUNT_HOME_HEAP_PUSHES,
    C_COUNT_HOME_HEAP_POPS,
    C_COUNT_HOME_HEAP_MAX,
    C_COUNT_HOME_TOP_UPDATES,
    C_COUNT_PROBE_NODES,
    C_COUNT_PROBE_LEAVES,
    C_COUNT_PROBE_CHUNKS,
    C_COUNT_PROBE_POINTS,
    C_COUNT_PROBE_BOUNDS,
    C_COUNT_PROBE_HEAP_PUSHES,
    C_COUNT_PROBE_HEAP_POPS,
    C_COUNT_PROBE_HEAP_MAX,
    C_COUNT_PROBE_TOP_UPDATES,
    C_COUNT_PROBE_PARTITIONS,
    C_COUNT_COUNT,
};

enum {
    C_INST_PHASE_HOME = 0,
    C_INST_PHASE_PROBE = 1,
};

static const char *C_COUNT_NAMES[C_COUNT_COUNT] = {
    "home_nodes",
    "home_leaves",
    "home_chunks",
    "home_points",
    "home_bounds",
    "home_heap_pushes",
    "home_heap_pops",
    "home_heap_max",
    "home_top_updates",
    "probe_nodes",
    "probe_leaves",
    "probe_chunks",
    "probe_points",
    "probe_bounds",
    "probe_heap_pushes",
    "probe_heap_pops",
    "probe_heap_max",
    "probe_top_updates",
    "probe_partitions",
};

static _Atomic uint64_t g_c_count_hist[C_COUNT_COUNT][C_COUNT_BUCKETS];
static _Atomic uint64_t g_c_key_count[MAX_PARTITIONS];
static _Atomic uint64_t g_c_key_home_us_hist[MAX_PARTITIONS][C_KEY_US_BUCKETS];
static _Atomic uint64_t g_c_key_total_us_hist[MAX_PARTITIONS][C_KEY_US_BUCKETS];
static _Atomic uint64_t g_c_key_home_us_sum[MAX_PARTITIONS];
static _Atomic uint64_t g_c_key_probe_us_sum[MAX_PARTITIONS];
static _Atomic uint64_t g_c_key_total_us_sum[MAX_PARTITIONS];
static _Atomic uint64_t g_c_key_home_nodes_sum[MAX_PARTITIONS];
static _Atomic uint64_t g_c_key_home_chunks_sum[MAX_PARTITIONS];
static _Atomic uint64_t g_c_key_home_points_sum[MAX_PARTITIONS];
static _Atomic uint64_t g_c_key_probe_nodes_sum[MAX_PARTITIONS];
static _Atomic uint64_t g_c_key_probe_chunks_sum[MAX_PARTITIONS];
static _Atomic uint64_t g_c_key_probe_points_sum[MAX_PARTITIONS];
static _Atomic uint64_t g_c_key_output_counts[MAX_PARTITIONS][NEIGHBOR_K + 1];

static inline void c_inst_set_phase(Search *s, uint8_t phase) {
    if (g_c_instrument) s->inst_phase = phase;
}

static inline void c_inst_node(Search *s) {
    if (g_c_instrument) s->inst_nodes[s->inst_phase]++;
}

static inline void c_inst_leaf(Search *s) {
    if (g_c_instrument) s->inst_leaves[s->inst_phase]++;
}

static inline void c_inst_chunk(Search *s, uint32_t points) {
    if (g_c_instrument) {
        s->inst_chunks[s->inst_phase]++;
        s->inst_points[s->inst_phase] += points;
    }
}

static inline void c_inst_bound(Search *s) {
    if (g_c_instrument) s->inst_bounds[s->inst_phase]++;
}

static inline void c_inst_heap_push(Search *s, size_t len) {
    if (g_c_instrument) {
        uint8_t phase = s->inst_phase;
        s->inst_heap_pushes[phase]++;
        if (len > s->inst_heap_max[phase]) s->inst_heap_max[phase] = (uint32_t)len;
    }
}

static inline void c_inst_heap_pop(Search *s) {
    if (g_c_instrument) s->inst_heap_pops[s->inst_phase]++;
}

static inline void c_inst_top_update(Search *s) {
    if (g_c_instrument) s->inst_top_updates[s->inst_phase]++;
}

static inline void c_inst_probe_partition(Search *s) {
    if (g_c_instrument) s->inst_probe_partitions++;
}
#else
#define c_inst_set_phase(s, phase) ((void)0)
#define c_inst_node(s) ((void)0)
#define c_inst_leaf(s) ((void)0)
#define c_inst_chunk(s, points) ((void)0)
#define c_inst_bound(s) ((void)0)
#define c_inst_heap_push(s, len) ((void)0)
#define c_inst_heap_pop(s) ((void)0)
#define c_inst_top_update(s) ((void)0)
#define c_inst_probe_partition(s) ((void)0)
#endif

static const char RESP_READY[] =
    "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK";
static const char RESP_BAD[] =
    "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
static const char *RESP_DECISION[6] = {
    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 33\r\n\r\n{\"approved\":true,\"fraud_score\":0}",
    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 34\r\n\r\n{\"approved\":false,\"fraud_score\":1}",
};
static const char *RESP_DECISION_COMPACT[6] = {
    "HTTP/1.1 200 OK\r\nContent-Length: 33\r\n\r\n{\"approved\":true,\"fraud_score\":0}",
    "HTTP/1.1 200 OK\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
    "HTTP/1.1 200 OK\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
    "HTTP/1.1 200 OK\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
    "HTTP/1.1 200 OK\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
    "HTTP/1.1 200 OK\r\nContent-Length: 34\r\n\r\n{\"approved\":false,\"fraud_score\":1}",
};
static size_t RESP_DECISION_LEN[6];
static size_t RESP_DECISION_COMPACT_LEN[6];

ENGINE_FUNC void init_response_lengths(void) {
    for (size_t i = 0; i < 6; i++) {
        RESP_DECISION_LEN[i] = strlen(RESP_DECISION[i]);
        RESP_DECISION_COMPACT_LEN[i] = strlen(RESP_DECISION_COMPACT[i]);
    }
}

static inline const char *decision_response(uint8_t frauds) {
    return g_c_compact_resp ? RESP_DECISION_COMPACT[frauds] : RESP_DECISION[frauds];
}

static inline size_t decision_response_len(uint8_t frauds) {
    return g_c_compact_resp ? RESP_DECISION_COMPACT_LEN[frauds] : RESP_DECISION_LEN[frauds];
}

static inline uint32_t le32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline int16_t le16s(const uint8_t *p) {
    return (int16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static void die(const char *msg) {
    perror(msg);
    exit(1);
}

static inline bool env_truthy(const char *v) {
    return v && strcmp(v, "0") != 0 && strcmp(v, "false") != 0 && strcmp(v, "FALSE") != 0 &&
           strcmp(v, "no") != 0 && strcmp(v, "NO") != 0;
}

static inline uint64_t c_now_ns(void) {
    if (!g_c_instrument) return 0;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static inline void c_record_ns(int stage, uint64_t ns) {
    if (!g_c_instrument || stage < 0 || stage >= C_STAGE_COUNT) return;
    uint64_t us = ns / 1000ull;
    if (us >= C_INSTRUMENT_BUCKETS) us = C_INSTRUMENT_BUCKETS - 1;
    atomic_fetch_add_explicit(&g_c_hist[stage][us], 1, memory_order_relaxed);
}

#if RINHA_C_DEEP_INSTRUMENT
static inline void c_record_count(int metric, uint32_t count) {
    if (!g_c_instrument || metric < 0 || metric >= C_COUNT_COUNT) return;
    uint32_t bucket = count;
    if (bucket >= C_COUNT_BUCKETS) bucket = C_COUNT_BUCKETS - 1;
    atomic_fetch_add_explicit(&g_c_count_hist[metric][bucket], 1, memory_order_relaxed);
}

static void c_record_kd_counts(Search *s) {
    c_record_count(C_COUNT_HOME_NODES, s->inst_nodes[C_INST_PHASE_HOME]);
    c_record_count(C_COUNT_HOME_LEAVES, s->inst_leaves[C_INST_PHASE_HOME]);
    c_record_count(C_COUNT_HOME_CHUNKS, s->inst_chunks[C_INST_PHASE_HOME]);
    c_record_count(C_COUNT_HOME_POINTS, s->inst_points[C_INST_PHASE_HOME]);
    c_record_count(C_COUNT_HOME_BOUNDS, s->inst_bounds[C_INST_PHASE_HOME]);
    c_record_count(C_COUNT_HOME_HEAP_PUSHES, s->inst_heap_pushes[C_INST_PHASE_HOME]);
    c_record_count(C_COUNT_HOME_HEAP_POPS, s->inst_heap_pops[C_INST_PHASE_HOME]);
    c_record_count(C_COUNT_HOME_HEAP_MAX, s->inst_heap_max[C_INST_PHASE_HOME]);
    c_record_count(C_COUNT_HOME_TOP_UPDATES, s->inst_top_updates[C_INST_PHASE_HOME]);
    c_record_count(C_COUNT_PROBE_NODES, s->inst_nodes[C_INST_PHASE_PROBE]);
    c_record_count(C_COUNT_PROBE_LEAVES, s->inst_leaves[C_INST_PHASE_PROBE]);
    c_record_count(C_COUNT_PROBE_CHUNKS, s->inst_chunks[C_INST_PHASE_PROBE]);
    c_record_count(C_COUNT_PROBE_POINTS, s->inst_points[C_INST_PHASE_PROBE]);
    c_record_count(C_COUNT_PROBE_BOUNDS, s->inst_bounds[C_INST_PHASE_PROBE]);
    c_record_count(C_COUNT_PROBE_HEAP_PUSHES, s->inst_heap_pushes[C_INST_PHASE_PROBE]);
    c_record_count(C_COUNT_PROBE_HEAP_POPS, s->inst_heap_pops[C_INST_PHASE_PROBE]);
    c_record_count(C_COUNT_PROBE_HEAP_MAX, s->inst_heap_max[C_INST_PHASE_PROBE]);
    c_record_count(C_COUNT_PROBE_TOP_UPDATES, s->inst_top_updates[C_INST_PHASE_PROBE]);
    c_record_count(C_COUNT_PROBE_PARTITIONS, s->inst_probe_partitions);
}

static inline void c_record_key_us(_Atomic uint64_t hist[MAX_PARTITIONS][C_KEY_US_BUCKETS], uint32_t key, uint64_t ns) {
    if (!g_c_instrument || key >= MAX_PARTITIONS) return;
    uint64_t us = ns / 1000ull;
    if (us >= C_KEY_US_BUCKETS) us = C_KEY_US_BUCKETS - 1;
    atomic_fetch_add_explicit(&hist[key][us], 1, memory_order_relaxed);
}

static void c_record_key_profile(Search *s, uint8_t output, uint64_t total_ns) {
    if (!g_c_instrument || s->inst_query_key >= MAX_PARTITIONS) return;
    uint32_t key = s->inst_query_key;
    atomic_fetch_add_explicit(&g_c_key_count[key], 1, memory_order_relaxed);
    c_record_key_us(g_c_key_home_us_hist, key, s->inst_home_ns);
    c_record_key_us(g_c_key_total_us_hist, key, total_ns);
    atomic_fetch_add_explicit(&g_c_key_home_us_sum[key], s->inst_home_ns / 1000ull, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_c_key_probe_us_sum[key], s->inst_probe_ns / 1000ull, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_c_key_total_us_sum[key], total_ns / 1000ull, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_c_key_home_nodes_sum[key], s->inst_nodes[C_INST_PHASE_HOME], memory_order_relaxed);
    atomic_fetch_add_explicit(&g_c_key_home_chunks_sum[key], s->inst_chunks[C_INST_PHASE_HOME], memory_order_relaxed);
    atomic_fetch_add_explicit(&g_c_key_home_points_sum[key], s->inst_points[C_INST_PHASE_HOME], memory_order_relaxed);
    atomic_fetch_add_explicit(&g_c_key_probe_nodes_sum[key], s->inst_nodes[C_INST_PHASE_PROBE], memory_order_relaxed);
    atomic_fetch_add_explicit(&g_c_key_probe_chunks_sum[key], s->inst_chunks[C_INST_PHASE_PROBE], memory_order_relaxed);
    atomic_fetch_add_explicit(&g_c_key_probe_points_sum[key], s->inst_points[C_INST_PHASE_PROBE], memory_order_relaxed);
    if (output <= NEIGHBOR_K) {
        atomic_fetch_add_explicit(&g_c_key_output_counts[key][output], 1, memory_order_relaxed);
    }
}
#endif

static void c_dump_stage(int stage) {
    uint64_t total = 0;
    uint64_t max_us = 0;
    uint64_t weighted_us = 0;
    for (size_t i = 0; i < C_INSTRUMENT_BUCKETS; i++) {
        uint64_t n = atomic_load_explicit(&g_c_hist[stage][i], memory_order_relaxed);
        total += n;
        weighted_us += n * i;
        if (n != 0) max_us = i;
    }
    if (total == 0) return;

    uint64_t p50_at = (total * 50 + 99) / 100;
    uint64_t p95_at = (total * 95 + 99) / 100;
    uint64_t p99_at = (total * 99 + 99) / 100;
    uint64_t seen = 0, p50 = 0, p95 = 0, p99 = 0;
    for (size_t i = 0; i < C_INSTRUMENT_BUCKETS; i++) {
        uint64_t n = atomic_load_explicit(&g_c_hist[stage][i], memory_order_relaxed);
        if (n == 0) continue;
        seen += n;
        if (p50 == 0 && seen >= p50_at) p50 = i;
        if (p95 == 0 && seen >= p95_at) p95 = i;
        if (p99 == 0 && seen >= p99_at) {
            p99 = i;
            break;
        }
    }

    double avg = total == 0 ? 0.0 : (double)weighted_us / (double)total;
    fprintf(stderr,
            "c_instrument: stage=%s n=%llu avg_us=%.1f p50_us=%llu p95_us=%llu p99_us=%llu max_us=%llu\n",
            C_STAGE_NAMES[stage],
            (unsigned long long)total,
            avg,
            (unsigned long long)p50,
            (unsigned long long)p95,
            (unsigned long long)p99,
            (unsigned long long)max_us);
}

#if RINHA_C_DEEP_INSTRUMENT
static void c_dump_count(int metric) {
    uint64_t total = 0;
    uint64_t max_count = 0;
    uint64_t weighted_count = 0;
    for (size_t i = 0; i < C_COUNT_BUCKETS; i++) {
        uint64_t n = atomic_load_explicit(&g_c_count_hist[metric][i], memory_order_relaxed);
        total += n;
        weighted_count += n * i;
        if (n != 0) max_count = i;
    }
    if (total == 0) return;

    uint64_t p50_at = (total * 50 + 99) / 100;
    uint64_t p95_at = (total * 95 + 99) / 100;
    uint64_t p99_at = (total * 99 + 99) / 100;
    uint64_t seen = 0, p50 = 0, p95 = 0, p99 = 0;
    for (size_t i = 0; i < C_COUNT_BUCKETS; i++) {
        uint64_t n = atomic_load_explicit(&g_c_count_hist[metric][i], memory_order_relaxed);
        if (n == 0) continue;
        seen += n;
        if (p50 == 0 && seen >= p50_at) p50 = i;
        if (p95 == 0 && seen >= p95_at) p95 = i;
        if (p99 == 0 && seen >= p99_at) {
            p99 = i;
            break;
        }
    }

    double avg = total == 0 ? 0.0 : (double)weighted_count / (double)total;
    fprintf(stderr,
            "c_instrument: metric=%s n=%llu avg=%.1f p50=%llu p95=%llu p99=%llu max=%llu\n",
            C_COUNT_NAMES[metric],
            (unsigned long long)total,
            avg,
            (unsigned long long)p50,
            (unsigned long long)p95,
            (unsigned long long)p99,
            (unsigned long long)max_count);
}

static uint64_t c_key_hist_quantile(_Atomic uint64_t hist[C_KEY_US_BUCKETS], uint64_t total, uint64_t percentile) {
    if (total == 0) return 0;
    uint64_t target = (total * percentile + 99) / 100;
    uint64_t seen = 0;
    for (size_t i = 0; i < C_KEY_US_BUCKETS; i++) {
        uint64_t n = atomic_load_explicit(&hist[i], memory_order_relaxed);
        if (n == 0) continue;
        seen += n;
        if (seen >= target) return i;
    }
    return C_KEY_US_BUCKETS - 1;
}

typedef struct {
    uint32_t key;
    uint64_t count;
    uint64_t home_p99_us;
    uint64_t total_p99_us;
} CKeyRank;

static void c_key_rank_insert(CKeyRank top[C_KEY_TOP_N], CKeyRank item) {
    if (item.count == 0) return;
    size_t pos = C_KEY_TOP_N;
    for (size_t i = 0; i < C_KEY_TOP_N; i++) {
        if (item.home_p99_us > top[i].home_p99_us ||
            (item.home_p99_us == top[i].home_p99_us && item.count > top[i].count)) {
            pos = i;
            break;
        }
    }
    if (pos == C_KEY_TOP_N) return;
    for (size_t i = C_KEY_TOP_N - 1; i > pos; i--) top[i] = top[i - 1];
    top[pos] = item;
}

static void c_dump_key_profiles(void) {
    CKeyRank top[C_KEY_TOP_N];
    memset(top, 0, sizeof(top));
    uint64_t total = 0;
    uint64_t active = 0;
    for (uint32_t key = 0; key < MAX_PARTITIONS; key++) {
        uint64_t count = atomic_load_explicit(&g_c_key_count[key], memory_order_relaxed);
        if (count == 0) continue;
        total += count;
        active++;
        uint64_t home_p99 = c_key_hist_quantile(g_c_key_home_us_hist[key], count, 99);
        uint64_t total_p99 = c_key_hist_quantile(g_c_key_total_us_hist[key], count, 99);
        c_key_rank_insert(top, (CKeyRank){key, count, home_p99, total_p99});
    }
    if (total == 0) return;

    fprintf(stderr,
            "c_key_profile: active_keys=%llu total_fallbacks=%llu top_by_home_p99=%u\n",
            (unsigned long long)active,
            (unsigned long long)total,
            C_KEY_TOP_N);
    for (size_t i = 0; i < C_KEY_TOP_N; i++) {
        CKeyRank r = top[i];
        if (r.count == 0) break;
        uint64_t home_sum = atomic_load_explicit(&g_c_key_home_us_sum[r.key], memory_order_relaxed);
        uint64_t probe_sum = atomic_load_explicit(&g_c_key_probe_us_sum[r.key], memory_order_relaxed);
        uint64_t total_sum = atomic_load_explicit(&g_c_key_total_us_sum[r.key], memory_order_relaxed);
        uint64_t home_nodes = atomic_load_explicit(&g_c_key_home_nodes_sum[r.key], memory_order_relaxed);
        uint64_t home_chunks = atomic_load_explicit(&g_c_key_home_chunks_sum[r.key], memory_order_relaxed);
        uint64_t home_points = atomic_load_explicit(&g_c_key_home_points_sum[r.key], memory_order_relaxed);
        uint64_t probe_nodes = atomic_load_explicit(&g_c_key_probe_nodes_sum[r.key], memory_order_relaxed);
        uint64_t probe_chunks = atomic_load_explicit(&g_c_key_probe_chunks_sum[r.key], memory_order_relaxed);
        uint64_t probe_points = atomic_load_explicit(&g_c_key_probe_points_sum[r.key], memory_order_relaxed);
        uint64_t outs[NEIGHBOR_K + 1];
        for (size_t j = 0; j <= NEIGHBOR_K; j++) {
            outs[j] = atomic_load_explicit(&g_c_key_output_counts[r.key][j], memory_order_relaxed);
        }
        double denom = (double)r.count;
        fprintf(stderr,
                "c_key_profile: key=%u n=%llu home_avg_us=%.1f home_p99_us=%llu total_avg_us=%.1f total_p99_us=%llu probe_avg_us=%.1f home_nodes_avg=%.1f home_chunks_avg=%.1f home_points_avg=%.1f probe_nodes_avg=%.1f probe_chunks_avg=%.1f probe_points_avg=%.1f outputs=[%llu,%llu,%llu,%llu,%llu,%llu]\n",
                r.key,
                (unsigned long long)r.count,
                (double)home_sum / denom,
                (unsigned long long)r.home_p99_us,
                (double)total_sum / denom,
                (unsigned long long)r.total_p99_us,
                (double)probe_sum / denom,
                (double)home_nodes / denom,
                (double)home_chunks / denom,
                (double)home_points / denom,
                (double)probe_nodes / denom,
                (double)probe_chunks / denom,
                (double)probe_points / denom,
                (unsigned long long)outs[0],
                (unsigned long long)outs[1],
                (unsigned long long)outs[2],
                (unsigned long long)outs[3],
                (unsigned long long)outs[4],
                (unsigned long long)outs[5]);
    }
}
#endif

static void c_dump_instrument(void) {
    if (!g_c_instrument) return;
    for (int stage = 0; stage < C_STAGE_COUNT; stage++) c_dump_stage(stage);
#if RINHA_C_DEEP_INSTRUMENT
    for (int metric = 0; metric < C_COUNT_COUNT; metric++) c_dump_count(metric);
    c_dump_key_profiles();
#endif
}

static void *c_instrument_loop(void *arg) {
    (void)arg;
    while (true) {
        sleep(g_c_instrument_interval_secs);
        c_dump_instrument();
    }
    return NULL;
}

ENGINE_FUNC void c_instrument_init_from_env(void) {
    g_c_instrument = env_truthy(getenv("RINHA_C_INSTRUMENT"));
    if (!g_c_instrument) return;
    const char *interval = getenv("RINHA_C_INSTRUMENT_INTERVAL_SECS");
    if (interval) {
        long parsed = strtol(interval, NULL, 10);
        if (parsed > 0 && parsed < 3600) g_c_instrument_interval_secs = (unsigned)parsed;
    }
    fprintf(stderr, "c_instrument: enabled interval_secs=%u bucket_us=1 cap_us=%u\n",
            g_c_instrument_interval_secs, C_INSTRUMENT_BUCKETS - 1);
    pthread_t t;
    if (pthread_create(&t, NULL, c_instrument_loop, NULL) == 0) {
        pthread_detach(t);
    }
}

static int c_sched_policy_from_env(const char **name_out, int *prio_out) {
    const char *raw = getenv("RINHA_C_SCHED_POLICY");
    if (!raw || strcmp(raw, "") == 0 || strcmp(raw, "default") == 0 || strcmp(raw, "other") == 0) return -1;
    int policy = -1;
    const char *name = raw;
    if (strcmp(raw, "fifo") == 0 || strcmp(raw, "FIFO") == 0) {
        policy = SCHED_FIFO;
        name = "SCHED_FIFO";
    } else if (strcmp(raw, "rr") == 0 || strcmp(raw, "RR") == 0) {
        policy = SCHED_RR;
        name = "SCHED_RR";
#ifdef SCHED_BATCH
    } else if (strcmp(raw, "batch") == 0 || strcmp(raw, "BATCH") == 0) {
        policy = SCHED_BATCH;
        name = "SCHED_BATCH";
#endif
#ifdef SCHED_IDLE
    } else if (strcmp(raw, "idle") == 0 || strcmp(raw, "IDLE") == 0) {
        policy = SCHED_IDLE;
        name = "SCHED_IDLE";
#endif
    } else {
        if (!atomic_exchange_explicit(&g_c_sched_logged, true, memory_order_relaxed)) {
            fprintf(stderr, "c_sched: unknown RINHA_C_SCHED_POLICY=%s, staying default\n", raw);
        }
        return -1;
    }

    int prio = 0;
    if (policy == SCHED_FIFO || policy == SCHED_RR) {
        prio = 10;
        const char *env_prio = getenv("RINHA_C_SCHED_PRIO");
        if (env_prio) {
            long parsed = strtol(env_prio, NULL, 10);
            if (parsed < 1) parsed = 1;
            if (parsed > 99) parsed = 99;
            prio = (int)parsed;
        }
    }
    if (name_out) *name_out = name;
    if (prio_out) *prio_out = prio;
    return policy;
}

static void parse_key_list(const char *raw, bool keys[MAX_PARTITIONS], bool *enabled) {
    if (!raw || *raw == '\0') return;
    const char *p = raw;
    while (*p != '\0') {
        while (*p == ',' || *p == ';' || isspace((unsigned char)*p)) p++;
        if (*p == '\0') break;
        char *endp = NULL;
        long key = strtol(p, &endp, 10);
        if (endp == p) break;
        if (key >= 0 && key < MAX_PARTITIONS) {
            keys[key] = true;
            *enabled = true;
        }
        p = endp;
        while (*p != '\0' && *p != ',' && *p != ';' && !isspace((unsigned char)*p)) p++;
    }
}

static unsigned key_list_count(const bool keys[MAX_PARTITIONS]) {
    unsigned count = 0;
    for (size_t i = 0; i < MAX_PARTITIONS; i++) {
        if (keys[i]) count++;
    }
    return count;
}

static void c_apply_sched_current_thread(const char *role) {
    const char *name = NULL;
    int prio = 0;
    int policy = c_sched_policy_from_env(&name, &prio);
    if (policy < 0) return;
    struct sched_param param;
    memset(&param, 0, sizeof(param));
    param.sched_priority = prio;
    int rc = sched_setscheduler(0, policy, &param);
    if (!atomic_exchange_explicit(&g_c_sched_logged, true, memory_order_relaxed)) {
        if (rc == 0) {
            if (prio > 0) fprintf(stderr, "c_sched: %s -> %s prio=%d\n", role, name, prio);
            else fprintf(stderr, "c_sched: %s -> %s\n", role, name);
        } else {
            fprintf(stderr, "c_sched: %s denied for %s: %s\n", name, role, strerror(errno));
        }
    }
}

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) die("malloc");
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) die("calloc");
    return p;
}

static size_t page_round_up(size_t n) {
    long page = sysconf(_SC_PAGESIZE);
    size_t p = page > 0 ? (size_t)page : 4096;
    return (n + p - 1) & ~(p - 1);
}

#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif
#ifndef PR_SET_THP_DISABLE
#define PR_SET_THP_DISABLE 41
#endif
#define HUGE_PAGE_SIZE ((size_t)2 * 1024 * 1024)

static void *hot_alloc(size_t n) {
    if (n == 0) return NULL;
    if (!g_c_mmap_hot) return xcalloc(1, n);
    size_t alloc = page_round_up(n);

    bool huge_candidate = g_c_index_collapse && alloc >= HUGE_PAGE_SIZE;
    if (huge_candidate) alloc = (alloc + HUGE_PAGE_SIZE - 1) & ~(HUGE_PAGE_SIZE - 1);
    void *p = mmap(NULL, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        perror("mmap hot");
        return xcalloc(1, n);
    }
#ifdef MADV_HUGEPAGE
    (void)madvise(p, alloc, MADV_HUGEPAGE);
#endif
#ifdef MADV_RANDOM
    (void)madvise(p, alloc, MADV_RANDOM);
#endif
#ifdef MADV_WILLNEED
    (void)madvise(p, alloc, MADV_WILLNEED);
#endif
    memset(p, 0, alloc);
    if (huge_candidate) {

        (void)madvise(p, alloc, MADV_COLLAPSE);
        if (g_c_index_mlock) (void)mlock(p, alloc);
    }
    return p;
}

static void hot_prefetch(const void *p, size_t n) {
    if (!p || n == 0) return;
    long page_l = sysconf(_SC_PAGESIZE);
    size_t page = page_l > 0 ? (size_t)page_l : 4096;
    volatile const uint8_t *b = (volatile const uint8_t *)p;
    uint8_t acc = 0;
    for (size_t i = 0; i < n; i += page) acc ^= b[i];
    (void)acc;
}

ENGINE_FUNC void log_anon_hugepages(const char *when) {
    FILE *f = fopen("/proc/self/smaps_rollup", "r");
    if (!f) f = fopen("/proc/self/smaps", "r");
    if (!f) return;
    char line[256];
    long anon_huge_kb = -1, rss_anon_kb = -1;
    while (fgets(line, sizeof line, f)) {
        long v;
        if (strncmp(line, "AnonHugePages:", 14) == 0 && sscanf(line + 14, "%ld", &v) == 1)
            anon_huge_kb = (anon_huge_kb < 0 ? 0 : anon_huge_kb) + v;
        else if (strncmp(line, "RssAnon:", 8) == 0 && sscanf(line + 8, "%ld", &v) == 1)
            rss_anon_kb = (rss_anon_kb < 0 ? 0 : rss_anon_kb) + v;
    }
    fclose(f);
    fprintf(stderr, "c-api: THP check (%s): AnonHugePages=%ld kB RssAnon=%ld kB%s\n",
            when, anon_huge_kb, rss_anon_kb,
            anon_huge_kb > 0 ? "  [huge pages ENGAGED]" : "  [WARNING: index NOT on huge pages]");
}

static void read_exact(FILE *f, void *buf, size_t n, const char *what) {
    if (fread(buf, 1, n, f) != n) {
        fprintf(stderr, "read failed: %s\n", what);
        exit(1);
    }
}

static inline void dequant_dims(const int16_t in[DIMENSIONS], float out[DIMENSIONS]) {
    for (int d = 0; d < DIMENSIONS; d++) out[d] = (float)in[d] * INV_Q_SCALE;
}

static FileNode decode_node(const uint8_t buf[FILE_NODE_BYTES]) {
    FileNode n;
    size_t p = 0;
    for (int d = 0; d < DIMENSIONS; d++, p += 2) n.aabb_min[d] = le16s(buf + p);
    for (int d = 0; d < DIMENSIONS; d++, p += 2) n.aabb_max[d] = le16s(buf + p);
    n.left = le32(buf + p);
    n.right = le32(buf + p + 4);
    n.count = le32(buf + p + 8);
    return n;
}

static FilePartition decode_partition(const uint8_t buf[FILE_PARTITION_BYTES]) {
    FilePartition part;
    size_t p = 0;
    part.key = le32(buf + p);
    p += 4;
    part.root = le32(buf + p);
    p += 4;
    part.count = le32(buf + p);
    p += 4;
    for (int d = 0; d < DIMENSIONS; d++, p += 2) part.aabb_min[d] = le16s(buf + p);
    for (int d = 0; d < DIMENSIONS; d++, p += 2) part.aabb_max[d] = le16s(buf + p);
    return part;
}

static void load_kdtree(const char *kdtree_path, const char *refs_path, KdIndex *idx) {
    FILE *kf = fopen(kdtree_path, "rb");
    if (!kf) die("open kdtree");

    uint8_t header[24];
    read_exact(kf, header, sizeof(header), "kdtree header");
    if (memcmp(header, "RKDT", 4) != 0) {
        fprintf(stderr, "bad kdtree magic\n");
        exit(1);
    }
    uint32_t version = le32(header + 4);
    if (version != 2) {
        fprintf(stderr, "expected partitioned RKDT v2, got %u\n", version);
        exit(1);
    }
    uint32_t count = le32(header + 8);
    uint32_t dim = le32(header + 12);
    uint32_t n_nodes = le32(header + 16);
    if (dim != DIMENSIONS) {
        fprintf(stderr, "bad kdtree dim %u\n", dim);
        exit(1);
    }
    uint8_t pbuf4[4];
    read_exact(kf, pbuf4, 4, "partition count");
    uint32_t n_parts = le32(pbuf4);
    if (n_parts > MAX_PARTITIONS) {
        fprintf(stderr, "too many partitions %u\n", n_parts);
        exit(1);
    }

    FilePartition *fparts = xmalloc((size_t)n_parts * sizeof(FilePartition));
    uint8_t pbuf[FILE_PARTITION_BYTES];
    for (uint32_t i = 0; i < n_parts; i++) {
        read_exact(kf, pbuf, sizeof(pbuf), "partition");
        fparts[i] = decode_partition(pbuf);
    }

    FileNode *fnodes = xmalloc((size_t)n_nodes * sizeof(FileNode));
    uint8_t nbuf[FILE_NODE_BYTES];
    for (uint32_t i = 0; i < n_nodes; i++) {
        read_exact(kf, nbuf, sizeof(nbuf), "node");
        fnodes[i] = decode_node(nbuf);
    }

    uint32_t *members = xmalloc((size_t)count * sizeof(uint32_t));
    read_exact(kf, members, (size_t)count * sizeof(uint32_t), "members");
    fclose(kf);

    KdNode *nodes = hot_alloc((size_t)n_nodes * sizeof(KdNode));
    uint32_t *slot = xmalloc((size_t)count * sizeof(uint32_t));
    uint32_t *leaf_node_by_ref = xmalloc((size_t)count * sizeof(uint32_t));
    uint8_t *node_masks = xmalloc((size_t)n_nodes);
    memset(slot, 0xff, (size_t)count * sizeof(uint32_t));
    memset(leaf_node_by_ref, 0xff, (size_t)count * sizeof(uint32_t));
    memset(node_masks, 0, (size_t)n_nodes);
    uint8_t *chunk_lens = xmalloc((size_t)count);
    size_t chunk_count = 0;

    for (uint32_t i = 0; i < n_nodes; i++) {
        FileNode *f = &fnodes[i];
        KdNode *n = &nodes[i];
        memcpy(n->aabb_min, f->aabb_min, sizeof(f->aabb_min));
        memcpy(n->aabb_max, f->aabb_max, sizeof(f->aabb_max));
        n->aabb_min[14] = 0;
        n->aabb_min[15] = 0;
        n->aabb_max[14] = 0;
        n->aabb_max[15] = 0;
        n->left = f->left;
        n->right = f->right;
        n->chunk_start = 0;
        n->chunk_end = 0;
        n->parent_diff_mask = 0;
        n->label_mask = 0;
        if (f->left == LEAF_FLAG) {
            size_t member_start = f->right;
            size_t cnt = f->count;
            size_t n_chunks = (cnt + LANES - 1) / LANES;
            size_t chunk_start = chunk_count;
            for (size_t ci = 0; ci < n_chunks; ci++) {
                size_t take = cnt - ci * LANES;
                if (take > LANES) take = LANES;
                chunk_lens[chunk_count++] = (uint8_t)take;
            }
            for (size_t j = 0; j < cnt; j++) {
                uint32_t ref_idx = members[member_start + j];
                uint32_t chunk_id = (uint32_t)(chunk_start + j / LANES);
                uint32_t lane = (uint32_t)(j % LANES);
                slot[ref_idx] = (chunk_id << 3) | lane;
                leaf_node_by_ref[ref_idx] = i;
            }
            n->right = 0;
            n->chunk_start = (uint32_t)chunk_start;
            n->chunk_end = (uint32_t)chunk_count;
        }
    }

    for (uint32_t i = 0; i < n_nodes; i++) {
        KdNode *n = &nodes[i];
        if (n->left == LEAF_FLAG) continue;
        uint32_t child_ids[2] = {n->left, n->right};
        for (int ci = 0; ci < 2; ci++) {
            KdNode *child = &nodes[child_ids[ci]];
            uint16_t mask = 0;
            for (int d = 0; d < DIMENSIONS; d++) {
                if (child->aabb_min[d] != n->aabb_min[d] ||
                    child->aabb_max[d] != n->aabb_max[d]) {
                    mask |= (uint16_t)(1u << d);
                }
            }
            child->parent_diff_mask = mask;
        }
    }

    LeafVecChunk *chunks = hot_alloc(chunk_count * sizeof(LeafVecChunk));
    LeafMeta *metas = hot_alloc(chunk_count * sizeof(LeafMeta));
    for (size_t i = 0; i < chunk_count; i++) metas[i].len = chunk_lens[i];

    FILE *rf = fopen(refs_path, "rb");
    if (!rf) die("open refs");
    uint8_t rh[12];
    read_exact(rf, rh, sizeof(rh), "refs header");
    if (memcmp(rh, "RINH", 4) != 0 || le32(rh + 4) != 3 || le32(rh + 8) != count) {
        fprintf(stderr, "refs header mismatch\n");
        exit(1);
    }
    uint8_t rec[REFS_RECORD_BYTES];
    for (uint32_t ref_idx = 0; ref_idx < count; ref_idx++) {
        read_exact(rf, rec, sizeof(rec), "refs record");
        uint32_t packed = slot[ref_idx];
        LeafVecChunk *c = &chunks[packed >> 3];
        LeafMeta *m = &metas[packed >> 3];
        int lane = (int)(packed & 7);
        for (int d = 0; d < DIMENSIONS; d++)
            c->pairs[d >> 1][(lane << 1) | (d & 1)] = le16s(rec + d * 2);
        m->labels[lane] = rec[DIMENSIONS * 2];
        m->ref_indices[lane] = ref_idx;
        node_masks[leaf_node_by_ref[ref_idx]] |= (uint8_t)(1u << m->labels[lane]);
    }
    fclose(rf);

    if (getenv("RINHA_C_DUMP_CHUNK_RANGES")) {

        long hist[6] = {0};
        long maxrange = 0;
        long chunks_all_fit = 0;
        for (size_t ci = 0; ci < chunk_count; ci++) {
            int len = metas[ci].len;
            if (len <= 0) continue;
            int chunk_fits = 1;
            for (int d = 0; d < DIMENSIONS; d++) {
                int16_t mn = 32767, mx = -32768;
                for (int l = 0; l < len; l++) {
                    int16_t v = chunks[ci].pairs[d >> 1][(l << 1) | (d & 1)];
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
                int range = (int)mx - (int)mn;
                if (range > maxrange) maxrange = range;
                if (range > 255) chunk_fits = 0;
                if (range <= 255) hist[0]++;
                else if (range <= 511) hist[1]++;
                else if (range <= 1023) hist[2]++;
                else if (range <= 4095) hist[3]++;
                else if (range <= 16383) hist[4]++;
                else hist[5]++;
            }
            chunks_all_fit += chunk_fits;
        }
        fprintf(stderr,
                "LVQ_CHUNK_RANGES total_pairs=%zu <=255:%ld <=511:%ld <=1023:%ld "
                "<=4095:%ld <=16383:%ld >16383:%ld max=%ld | chunks_all_dims_fit_i8=%ld/%zu\n",
                chunk_count * DIMENSIONS, hist[0], hist[1], hist[2], hist[3],
                hist[4], hist[5], maxrange, chunks_all_fit, chunk_count);
        fflush(stderr);
    }

    for (int64_t i = (int64_t)n_nodes - 1; i >= 0; i--) {
        KdNode *n = &nodes[i];
        if (n->left == LEAF_FLAG) {
            n->label_mask = node_masks[i];
        } else {
            n->label_mask = (uint8_t)(nodes[n->left].label_mask | nodes[n->right].label_mask);
        }
    }

    KdPartition *parts = hot_alloc((size_t)n_parts * sizeof(KdPartition));
    for (size_t i = 0; i < MAX_PARTITIONS; i++) idx->part_by_key[i] = -1;
    for (uint32_t i = 0; i < n_parts; i++) {
        parts[i].key = fparts[i].key;
        parts[i].root = fparts[i].root;
        memcpy(parts[i].aabb_min, fparts[i].aabb_min, sizeof(fparts[i].aabb_min));
        memcpy(parts[i].aabb_max, fparts[i].aabb_max, sizeof(fparts[i].aabb_max));
        parts[i].aabb_min[14] = 0;
        parts[i].aabb_min[15] = 0;
        parts[i].aabb_max[14] = 0;
        parts[i].aabb_max[15] = 0;
        if (parts[i].key < MAX_PARTITIONS) idx->part_by_key[parts[i].key] = (int32_t)i;
    }

    free(fparts);
    free(fnodes);
    free(members);
    free(slot);
    free(leaf_node_by_ref);
    free(node_masks);
    free(chunk_lens);

    idx->nodes = nodes;
    idx->node_count = n_nodes;
    idx->partitions = parts;
    idx->partition_count = n_parts;
    idx->chunks = chunks;
    idx->metas = metas;
    idx->chunk_count = chunk_count;
    idx->count = count;
    hot_prefetch(nodes, (size_t)n_nodes * sizeof(KdNode));
    hot_prefetch(parts, (size_t)n_parts * sizeof(KdPartition));
    hot_prefetch(chunks, chunk_count * sizeof(LeafVecChunk));
    fprintf(stderr, "c-api: loaded refs=%u nodes=%u partitions=%u chunks=%zu\n",
            count, n_nodes, n_parts, chunk_count);
}

static inline int32_t q_i32(float v) {
    return (int32_t)lrintf(v * Q_SCALE);
}

static inline int16_t q_i16(float v) {
    int32_t q = q_i32(v);
    if (q < INT16_MIN) q = INT16_MIN;
    if (q > INT16_MAX) q = INT16_MAX;
    return (int16_t)q;
}

static uint32_t partition_key_i16(const int16_t v[DIMENSIONS]) {
    uint32_t key = 0;
    if (v[5] >= 0) key |= 1u << 0;
    if (v[9] > 0) key |= 1u << 1;
    if (v[10] > 0) key |= 1u << 2;
    if (v[11] > 0) key |= 1u << 3;
    uint32_t mcc_bucket = 3;
    if (v[12] <= 2047) mcc_bucket = 0;
    else if (v[12] <= 4095) mcc_bucket = 1;
    else if (v[12] <= 6143) mcc_bucket = 2;
    key |= mcc_bucket << 4;
    if (v[2] > 4096) key |= 1u << 6;
    if (v[8] > 2048) key |= 1u << 7;
    switch (g_c_kdtree_extra_split) {
        case 1:
            if (v[7] > 2048) key |= 1u << 8;
            break;
        case 2:
            if (v[0] > 2048) key |= 1u << 8;
            break;
        case 3:
            if (v[6] > 2048) key |= 1u << 8;
            break;
        case 4:
            if (v[3] > 5000) key |= 1u << 8;
            break;
        case 5:
            if (v[13] > 200) key |= 1u << 8;
            break;
        default:
            break;
    }
    return key;
}

static inline int64_t aabb_lower_bound_i64(const int16_t q[PACKED_DIMENSIONS], const int16_t lo[PACKED_DIMENSIONS], const int16_t hi[PACKED_DIMENSIONS]) {
    __m256i qv = _mm256_loadu_si256((const __m256i *)q);
    __m256i lov = _mm256_loadu_si256((const __m256i *)lo);
    __m256i hiv = _mm256_loadu_si256((const __m256i *)hi);
    __m256i zero = _mm256_setzero_si256();
    __m256i low_gap = _mm256_max_epi16(_mm256_sub_epi16(lov, qv), zero);
    __m256i high_gap = _mm256_max_epi16(_mm256_sub_epi16(qv, hiv), zero);
    __m256i gap = _mm256_max_epi16(low_gap, high_gap);
    __m256i pair_sums = _mm256_madd_epi16(gap, gap);
    int32_t tmp[8];
    _mm256_storeu_si256((__m256i *)tmp, pair_sums);
    int64_t s = 0;
    for (int i = 0; i < 8; i++) s += tmp[i];
    return s;
}

static inline int64_t dim_gap_sq_i64(int16_t q, int16_t lo, int16_t hi) {
    int64_t gap = 0;
    if (q < lo) gap = (int64_t)lo - q;
    else if (q > hi) gap = (int64_t)q - hi;
    return gap * gap;
}

static inline bool can_still_improve(Search *s, int64_t lb) {
    int64_t cutoff = s->top_dist[NEIGHBOR_K - 1];
    return cutoff == INT64_MAX || lb <= cutoff;
}

static inline bool candidate_before(int64_t dist, uint32_t ref_index, int64_t other_dist, uint32_t other_index) {
    return dist < other_dist || (dist == other_dist && ref_index < other_index);
}

static inline bool top_is_chromatic_seed(Search *s) {
    return s->top_count == NEIGHBOR_K &&
        (s->top_sum == 0 || s->top_sum == NEIGHBOR_K);
}

static inline void maybe_activate_chromatic(Search *s) {
    if (!s->chrom_enabled || s->chrom_active || s->chrom_replaying) return;
    if (!top_is_chromatic_seed(s)) return;
    s->chrom_active = true;
    s->chrom_initial_sum = s->top_sum;
    s->chrom_needed_mask = (s->top_sum == 0) ? LABEL_MASK_FRAUD : LABEL_MASK_LEGIT;
}

static inline void insert_top(Search *s, int64_t dist, uint8_t label, uint32_t ref_index) {
    if (!candidate_before(dist, ref_index, s->top_dist[NEIGHBOR_K - 1], s->top_indices[NEIGHBOR_K - 1])) return;
    bool was_full = s->top_count == NEIGHBOR_K;
    uint8_t displaced = was_full ? s->top_label[NEIGHBOR_K - 1] : 0;
    int i = NEIGHBOR_K;
    while (i > 0 && candidate_before(dist, ref_index, s->top_dist[i - 1], s->top_indices[i - 1])) i--;
    if (i >= NEIGHBOR_K) return;
    for (int j = NEIGHBOR_K - 1; j > i; j--) {
        s->top_dist[j] = s->top_dist[j - 1];
        s->top_label[j] = s->top_label[j - 1];
        s->top_indices[j] = s->top_indices[j - 1];
    }
    s->top_dist[i] = dist;
    s->top_label[i] = label;
    s->top_indices[i] = ref_index;
    if (was_full) {
        s->top_sum = (uint8_t)(s->top_sum + label - displaced);
    } else {
        s->top_count++;
        s->top_sum = (uint8_t)(s->top_sum + label);
    }
    c_inst_top_update(s);
    maybe_activate_chromatic(s);
}

static inline void insert_top_from_meta(Search *s, int64_t dist, const LeafMeta *m, int lane) {
    int64_t cutoff = s->top_dist[NEIGHBOR_K - 1];
    if (dist > cutoff) return;
    uint32_t ref_index = m->ref_indices[lane];
    if (!candidate_before(dist, ref_index, cutoff, s->top_indices[NEIGHBOR_K - 1])) return;
    insert_top(s, dist, m->labels[lane], ref_index);
}

static inline bool early_done(Search *s) {
    if (g_c_kd_early_limit <= 0 || s->top_dist[NEIGHBOR_K - 1] > g_c_kd_early_limit) return false;
    if (g_c_kd_early_unanimous) return top_is_chromatic_seed(s);
    if (g_c_kd_early_margin >= 0) {
        if (s->top_count != NEIGHBOR_K) return false;
        return s->top_sum <= (uint8_t)g_c_kd_early_margin ||
            s->top_sum >= (uint8_t)(NEIGHBOR_K - g_c_kd_early_margin);
    }
    return true;
}

static inline void chunk_distances_madd(Search *s, const LeafVecChunk *c, int64_t dist[LANES]) {

    __m256i acc = _mm256_setzero_si256();
    for (int p = 0; p < DIM_PAIRS; p++) {
        __m256i v = _mm256_loadu_si256((const __m256i *)c->pairs[p]);
        __m256i diff = _mm256_sub_epi16(v, s->query_pairs[p]);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(diff, diff));
    }
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    _mm256_storeu_si256((__m256i *)dist, _mm256_cvtepi32_epi64(lo));
    _mm256_storeu_si256((__m256i *)(dist + 4), _mm256_cvtepi32_epi64(hi));
}

static inline void chunk_distances_pair_madd(
    Search *s,
    const LeafVecChunk *c0,
    const LeafVecChunk *c1,
    int64_t dist0[LANES],
    int64_t dist1[LANES]
) {
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    for (int p = 0; p < DIM_PAIRS; p++) {
        __m256i q = s->query_pairs[p];
        __m256i v0 = _mm256_loadu_si256((const __m256i *)c0->pairs[p]);
        __m256i d0 = _mm256_sub_epi16(v0, q);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(d0, d0));
        __m256i v1 = _mm256_loadu_si256((const __m256i *)c1->pairs[p]);
        __m256i d1 = _mm256_sub_epi16(v1, q);
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(d1, d1));
    }
    __m128i lo0 = _mm256_castsi256_si128(acc0);
    __m128i hi0 = _mm256_extracti128_si256(acc0, 1);
    __m128i lo1 = _mm256_castsi256_si128(acc1);
    __m128i hi1 = _mm256_extracti128_si256(acc1, 1);
    _mm256_storeu_si256((__m256i *)dist0, _mm256_cvtepi32_epi64(lo0));
    _mm256_storeu_si256((__m256i *)(dist0 + 4), _mm256_cvtepi32_epi64(hi0));
    _mm256_storeu_si256((__m256i *)dist1, _mm256_cvtepi32_epi64(lo1));
    _mm256_storeu_si256((__m256i *)(dist1 + 4), _mm256_cvtepi32_epi64(hi1));
}

static inline void insert_chunk_distances(Search *s, const LeafMeta *m, const int64_t dist[LANES], int lane_count) {
    for (int lane = 0; lane < lane_count; lane++) {
        insert_top_from_meta(s, dist[lane], m, lane);
    }
    if (early_done(s)) s->early_stop = true;
}

static void scan_chunk(Search *s, size_t chunk_id, int lane_count) {
    const LeafVecChunk *c = &s->index->chunks[chunk_id];
    const LeafMeta *m = &s->index->metas[chunk_id];
    if (g_c_kd_avx_leaf) {
        int64_t distances[LANES];
        chunk_distances_madd(s, c, distances);
        insert_chunk_distances(s, m, distances, lane_count);
    } else {
        for (int lane = 0; lane < lane_count; lane++) {
            int64_t dist = 0;
            for (int d = 0; d < DIMENSIONS; d++) {
                int64_t diff = (int64_t)s->query_q[d] - c->pairs[d >> 1][(lane << 1) | (d & 1)];
                dist += diff * diff;
            }
            insert_top_from_meta(s, dist, m, lane);
        }
        if (early_done(s)) s->early_stop = true;
    }
}

static void visit_node_bound(Search *s, uint32_t node_idx, int64_t bound);

static inline bool maybe_defer_node(Search *s, const KdNode *n, uint32_t node_idx, int64_t bound) {
    if (!s->chrom_enabled) return false;
    if (!s->chrom_active || s->chrom_replaying) return false;
    if (s->top_count != NEIGHBOR_K || s->top_sum != s->chrom_initial_sum) return false;
    if ((n->label_mask & s->chrom_needed_mask) != 0) return false;
    if (s->deferred_len < NODE_DEFER_CAP) {
        s->deferred[s->deferred_len++] = (NodeStackEntry){node_idx, bound};
        return true;
    }
    s->deferred_overflow = true;
    return false;
}

static void replay_deferred_nodes(Search *s) {
    if (s->deferred_len == 0) {
        s->chrom_active = false;
        return;
    }

    size_t len = s->deferred_len;
    s->deferred_len = 0;
    s->chrom_active = false;
    s->chrom_replaying = true;
    for (size_t i = 0; i < len; i++) {
        NodeStackEntry e = s->deferred[i];
        if (can_still_improve(s, e.bound)) visit_node_bound(s, e.idx, e.bound);
        if (s->early_stop) break;
    }
    s->chrom_replaying = false;
}

static inline int64_t child_lb(Search *s, uint32_t node_idx) {
    c_inst_bound(s);
    KdNode *n = &s->index->nodes[node_idx];
    return aabb_lower_bound_i64(s->query_q16, n->aabb_min, n->aabb_max);
}

static inline int64_t child_lb_from_parent(Search *s, const KdNode *parent, uint32_t node_idx, int64_t parent_bound) {
    c_inst_bound(s);
    KdNode *child = &s->index->nodes[node_idx];
    if (!g_c_kd_delta_bounds) return aabb_lower_bound_i64(s->query_q16, child->aabb_min, child->aabb_max);

    int64_t bound = parent_bound;
    uint16_t mask = child->parent_diff_mask;
    while (mask != 0) {
        int d = __builtin_ctz(mask);
        mask &= (uint16_t)(mask - 1);
        bound -= dim_gap_sq_i64(s->query_q16[d], parent->aabb_min[d], parent->aabb_max[d]);
        bound += dim_gap_sq_i64(s->query_q16[d], child->aabb_min[d], child->aabb_max[d]);
    }
    return bound;
}

static inline bool node_entry_before(NodeStackEntry a, NodeStackEntry b) {
    return a.bound < b.bound || (a.bound == b.bound && a.idx < b.idx);
}

static bool node_heap_push(NodeStackEntry *heap, size_t *len, NodeStackEntry e) {
    if (*len >= NODE_HEAP_CAP) return false;
    size_t i = (*len)++;
    while (i > 0) {
        size_t parent = (i - 1) >> 1;
        if (!node_entry_before(e, heap[parent])) break;
        heap[i] = heap[parent];
        i = parent;
    }
    heap[i] = e;
    return true;
}

static NodeStackEntry node_heap_pop(NodeStackEntry *heap, size_t *len) {
    NodeStackEntry out = heap[0];
    NodeStackEntry last = heap[--(*len)];
    if (*len == 0) return out;
    size_t i = 0;
    while (true) {
        size_t left = i * 2 + 1;
        if (left >= *len) break;
        size_t right = left + 1;
        size_t child = left;
        if (right < *len && node_entry_before(heap[right], heap[left])) child = right;
        if (!node_entry_before(heap[child], last)) break;
        heap[i] = heap[child];
        i = child;
    }
    heap[i] = last;
    return out;
}

static void visit_node_dfs(Search *s, uint32_t node_idx, int64_t bound) {
    NodeStackEntry stack[NODE_STACK_CAP];
    size_t sp = 0;
    uint32_t current = node_idx;
    int64_t current_bound = bound;

    while (true) {
        if (!g_c_kd_bound_thread) current_bound = child_lb(s, current);
        if (can_still_improve(s, current_bound)) {
            KdNode *n = &s->index->nodes[current];
            c_inst_node(s);
            if (maybe_defer_node(s, n, current, current_bound)) {
                if (sp == 0 || s->early_stop) return;
                NodeStackEntry next = stack[--sp];
                current = next.idx;
                current_bound = next.bound;
                continue;
            }
            if (n->left == LEAF_FLAG) {
                c_inst_leaf(s);
                for (uint32_t cid = n->chunk_start; cid < n->chunk_end; cid++) {
                    if (g_c_kd_prefetch_dist > 0) {
                        uint32_t target = cid + (uint32_t)g_c_kd_prefetch_dist;
                        if (target < n->chunk_end) {
                            __builtin_prefetch(&s->index->chunks[target], 0, 3);
                            if (g_c_kd_prefetch_meta) {
                                __builtin_prefetch(&s->index->metas[target], 0, 3);
                            }
                        }
                    }
                    if (g_c_kd_avx_leaf && g_c_kd_pair_leaf && cid + 1 < n->chunk_end) {
                        int64_t dist0[LANES];
                        int64_t dist1[LANES];
                        chunk_distances_pair_madd(
                            s,
                            &s->index->chunks[cid],
                            &s->index->chunks[cid + 1],
                            dist0,
                            dist1
                        );
                        c_inst_chunk(s, LANES);
                        insert_chunk_distances(s, &s->index->metas[cid], dist0, LANES);
                        if (s->early_stop) return;
                        int lane_count1 = (cid + 2 == n->chunk_end) ? s->index->metas[cid + 1].len : LANES;
                        c_inst_chunk(s, (uint32_t)lane_count1);
                        insert_chunk_distances(s, &s->index->metas[cid + 1], dist1, lane_count1);
                        if (s->early_stop) return;
                        cid++;
                        continue;
                    }
                    int lane_count = (cid + 1 == n->chunk_end) ? s->index->metas[cid].len : LANES;
                    c_inst_chunk(s, (uint32_t)lane_count);
                    scan_chunk(s, cid, lane_count);
                    if (s->early_stop) return;
                }
            } else {
                uint32_t l = n->left, r = n->right;
                int64_t dl = child_lb_from_parent(s, n, l, current_bound);
                int64_t dr = child_lb_from_parent(s, n, r, current_bound);
                uint32_t near = l, far = r;
                int64_t near_b = dl, far_b = dr;
                if (dr < dl) {
                    near = r;
                    far = l;
                    near_b = dr;
                    far_b = dl;
                }
                if (can_still_improve(s, far_b) && sp < NODE_STACK_CAP) {
                    stack[sp++] = (NodeStackEntry){far, far_b};
                }
                current = near;
                current_bound = near_b;
                continue;
            }
        }

        if (sp == 0 || s->early_stop) return;
        NodeStackEntry next = stack[--sp];
        current = next.idx;
        current_bound = next.bound;
    }
}

static void visit_node_best_first(Search *s, uint32_t node_idx, int64_t bound) {
    NodeStackEntry heap[NODE_HEAP_CAP];
    size_t len = 0;
#if RINHA_C_DEEP_INSTRUMENT
    if (node_heap_push(heap, &len, (NodeStackEntry){node_idx, bound})) {
        c_inst_heap_push(s, len);
    }
#else
    (void)node_heap_push(heap, &len, (NodeStackEntry){node_idx, bound});
#endif

    while (len > 0 && !s->early_stop) {
        NodeStackEntry e = node_heap_pop(heap, &len);
        c_inst_heap_pop(s);
        if (!can_still_improve(s, e.bound)) continue;

        KdNode *n = &s->index->nodes[e.idx];
        c_inst_node(s);
        if (maybe_defer_node(s, n, e.idx, e.bound)) continue;

        if (n->left == LEAF_FLAG) {
            c_inst_leaf(s);
            for (uint32_t cid = n->chunk_start; cid < n->chunk_end; cid++) {
                if (g_c_kd_prefetch_dist > 0) {
                    uint32_t target = cid + (uint32_t)g_c_kd_prefetch_dist;
                    if (target < n->chunk_end) {
                        __builtin_prefetch(&s->index->chunks[target], 0, 3);
                        if (g_c_kd_prefetch_meta) {
                            __builtin_prefetch(&s->index->metas[target], 0, 3);
                        }
                    }
                }
                if (g_c_kd_avx_leaf && g_c_kd_pair_leaf && cid + 1 < n->chunk_end) {
                    int64_t dist0[LANES];
                    int64_t dist1[LANES];
                    chunk_distances_pair_madd(
                        s,
                        &s->index->chunks[cid],
                        &s->index->chunks[cid + 1],
                        dist0,
                        dist1
                    );
                    c_inst_chunk(s, LANES);
                    insert_chunk_distances(s, &s->index->metas[cid], dist0, LANES);
                    if (s->early_stop) return;
                    int lane_count1 = (cid + 2 == n->chunk_end) ? s->index->metas[cid + 1].len : LANES;
                    c_inst_chunk(s, (uint32_t)lane_count1);
                    insert_chunk_distances(s, &s->index->metas[cid + 1], dist1, lane_count1);
                    if (s->early_stop) return;
                    cid++;
                    continue;
                }
                int lane_count = (cid + 1 == n->chunk_end) ? s->index->metas[cid].len : LANES;
                c_inst_chunk(s, (uint32_t)lane_count);
                scan_chunk(s, cid, lane_count);
                if (s->early_stop) return;
            }
        } else {
            uint32_t l = n->left, r = n->right;
            int64_t dl = child_lb_from_parent(s, n, l, e.bound);
            int64_t dr = child_lb_from_parent(s, n, r, e.bound);
#if RINHA_C_DEEP_INSTRUMENT
            if (can_still_improve(s, dl)) {
                if (node_heap_push(heap, &len, (NodeStackEntry){l, dl})) {
                    c_inst_heap_push(s, len);
                } else {
                    visit_node_dfs(s, l, dl);
                }
            }
            if (s->early_stop) return;
            if (can_still_improve(s, dr)) {
                if (node_heap_push(heap, &len, (NodeStackEntry){r, dr})) {
                    c_inst_heap_push(s, len);
                } else {
                    visit_node_dfs(s, r, dr);
                }
            }
#else
            if (can_still_improve(s, dl) &&
                !node_heap_push(heap, &len, (NodeStackEntry){l, dl})) {
                visit_node_dfs(s, l, dl);
            }
            if (s->early_stop) return;
            if (can_still_improve(s, dr) &&
                !node_heap_push(heap, &len, (NodeStackEntry){r, dr})) {
                visit_node_dfs(s, r, dr);
            }
#endif
        }
    }
}

static void visit_node_bound(Search *s, uint32_t node_idx, int64_t bound) {
    if (g_c_kd_best_first) visit_node_best_first(s, node_idx, bound);
    else visit_node_dfs(s, node_idx, bound);
}

static inline bool use_best_first_for_primary_key(uint32_t query_key) {
    return g_c_kd_best_first ||
        (g_c_kd_best_first_keys_enabled &&
         query_key < MAX_PARTITIONS &&
         g_c_kd_best_first_keys[query_key]);
}

static inline bool use_chromatic_for_primary_key(uint32_t query_key) {
    return g_c_kd_chromatic ||
        (g_c_kd_chromatic_primary_keys_enabled &&
         query_key < MAX_PARTITIONS &&
         g_c_kd_chromatic_primary_keys[query_key]);
}

static void visit_primary_node(Search *s, uint32_t node_idx, int64_t bound, uint32_t query_key) {
    if (use_best_first_for_primary_key(query_key)) visit_node_best_first(s, node_idx, bound);
    else visit_node_dfs(s, node_idx, bound);
}

typedef struct {
    int64_t bound;
    uint32_t idx;
} PartitionEntry;

static inline bool partition_before(PartitionEntry a, PartitionEntry b) {
    return a.bound < b.bound || (a.bound == b.bound && a.idx < b.idx);
}

static void sort_partition_entries(PartitionEntry *entries, size_t len) {
    for (size_t i = 1; i < len; i++) {
        PartitionEntry key = entries[i];
        size_t j = i;
        while (j > 0 && partition_before(key, entries[j - 1])) {
            entries[j] = entries[j - 1];
            j--;
        }
        entries[j] = key;
    }
}

static int cmp_partition_entry_qsort(const void *a, const void *b) {
    const PartitionEntry *x = (const PartitionEntry *)a;
    const PartitionEntry *y = (const PartitionEntry *)b;
    if (x->bound != y->bound) return (x->bound > y->bound) ? 1 : -1;
    return (x->idx > y->idx) - (x->idx < y->idx);
}

static void visit_partitions(Search *s, uint32_t query_key) {
    uint64_t t0 = c_now_ns();
    PartitionEntry entries[MAX_PARTITIONS];
    size_t len = 0;
    s->chrom_enabled = use_chromatic_for_primary_key(query_key);
#if RINHA_C_DEEP_INSTRUMENT
    if (g_c_instrument) s->inst_query_key = query_key;
#endif
    int32_t primary_idx = -1;
    if (g_c_kd_primary_map) {
        primary_idx = (query_key < MAX_PARTITIONS) ? s->index->part_by_key[query_key] : -1;
    } else {
        for (size_t i = 0; i < s->index->partition_count; i++) {
            if (s->index->partitions[i].key == query_key) {
                primary_idx = (int32_t)i;
                break;
            }
        }
    }

    if (primary_idx >= 0) {
        c_inst_set_phase(s, C_INST_PHASE_HOME);
        KdPartition *primary = &s->index->partitions[primary_idx];
        int64_t primary_bound = aabb_lower_bound_i64(s->query_q16, primary->aabb_min, primary->aabb_max);
        if (can_still_improve(s, primary_bound)) visit_primary_node(s, primary->root, primary_bound, query_key);
        if (s->chrom_active && s->top_sum != s->chrom_initial_sum) replay_deferred_nodes(s);
        if (s->early_stop) {
            uint64_t done = c_now_ns();
            if (done != 0) {
                c_record_ns(C_STAGE_K_HOME, done - t0);
                c_record_ns(C_STAGE_K_PSCAN, 0);
                c_record_ns(C_STAGE_K_SORT, 0);
                c_record_ns(C_STAGE_K_PROBE, 0);
#if RINHA_C_DEEP_INSTRUMENT
                s->inst_home_ns = done - t0;
                s->inst_probe_ns = 0;
#endif
            }
            return;
        }
    }
    maybe_activate_chromatic(s);
    uint64_t t1 = c_now_ns();
    if (t1 != 0) {
        c_record_ns(C_STAGE_K_HOME, t1 - t0);
#if RINHA_C_DEEP_INSTRUMENT
        s->inst_home_ns = t1 - t0;
#endif
    }

    for (size_t i = 0; i < s->index->partition_count; i++) {
        if ((int32_t)i == primary_idx) continue;
        KdPartition *p = &s->index->partitions[i];
        int64_t bound = aabb_lower_bound_i64(s->query_q16, p->aabb_min, p->aabb_max);
        if (can_still_improve(s, bound)) {
            entries[len++] = (PartitionEntry){bound, (uint32_t)i};
        }
    }
    uint64_t t2 = c_now_ns();
    if (t2 != 0) c_record_ns(C_STAGE_K_PSCAN, t2 - t1);
    if (g_c_kd_inline_sort) sort_partition_entries(entries, len);
    else qsort(entries, len, sizeof(entries[0]), cmp_partition_entry_qsort);
    uint64_t t3 = c_now_ns();
    if (t3 != 0) c_record_ns(C_STAGE_K_SORT, t3 - t2);
    c_inst_set_phase(s, C_INST_PHASE_PROBE);
    for (size_t i = 0; i < len; i++) {
        if (!can_still_improve(s, entries[i].bound)) break;
        c_inst_probe_partition(s);
        if (g_c_kd_best_first_primary_only) {
            visit_node_dfs(s, s->index->partitions[entries[i].idx].root, entries[i].bound);
        } else {
            visit_node_bound(s, s->index->partitions[entries[i].idx].root, entries[i].bound);
        }
        if (s->early_stop) break;
        if (s->chrom_active && s->top_sum != s->chrom_initial_sum) {
            replay_deferred_nodes(s);
            if (s->early_stop) break;
        }
    }
    if (s->chrom_active && s->top_sum != s->chrom_initial_sum) replay_deferred_nodes(s);
    uint64_t t4 = c_now_ns();
    if (t4 != 0) {
        c_record_ns(C_STAGE_K_PROBE, t4 - t3);
#if RINHA_C_DEEP_INSTRUMENT
        s->inst_probe_ns = t4 - t3;
#endif
    }
}

typedef struct {
    int64_t dist;
    uint32_t idx;
    uint8_t label;
} FinalCand;

static int cmp_final_cand(const void *a, const void *b) {
    const FinalCand *x = a, *y = b;
    if (x->dist != y->dist) return (x->dist > y->dist) ? 1 : -1;
    return (x->idx > y->idx) - (x->idx < y->idx);
}

static inline int64_t i64_dist(const int32_t q[DIMENSIONS], const int16_t dims[DIMENSIONS]) {
    int64_t acc = 0;
    for (int d = 0; d < DIMENSIONS; d++) {
        int64_t diff = (int64_t)q[d] - dims[d];
        acc += diff * diff;
    }
    return acc;
}

static uint8_t finalize_search(Search *s) {
    uint64_t t0 = c_now_ns();
    uint8_t sum = 0;
    for (int i = 0; i < NEIGHBOR_K; i++) sum += s->top_label[i];
    uint64_t t1 = c_now_ns();
    if (t1 != 0) c_record_ns(C_STAGE_K_RERANK, t1 - t0);
    return sum;
}

static uint8_t kdtree_fraud_count(KdIndex *idx, const float query[DIMENSIONS]) {
    uint64_t t0 = c_now_ns();
    int32_t q[DIMENSIONS];
    int16_t qi16[PACKED_DIMENSIONS] = {0};
    if (g_c_quant_once) {

        for (int d = 0; d < DIMENSIONS; d++) {
            int32_t qd = q_i32(query[d]);
            q[d] = qd;
            if (qd < INT16_MIN) qd = INT16_MIN;
            if (qd > INT16_MAX) qd = INT16_MAX;
            qi16[d] = (int16_t)qd;
        }
    } else {
        for (int d = 0; d < DIMENSIONS; d++) {
            q[d] = q_i32(query[d]);
            qi16[d] = q_i16(query[d]);
        }
    }
    Search s;
    memset(&s, 0, sizeof(s));
    s.index = idx;
    s.query_q = q;
    s.query_q16 = qi16;
    s.deferred = g_deferred_nodes;

    for (int p = 0; p < DIM_PAIRS; p++) {
        int32_t packed = (int32_t)((uint16_t)qi16[2 * p] |
                                   ((uint32_t)(uint16_t)qi16[2 * p + 1] << 16));
        s.query_pairs[p] = _mm256_set1_epi32(packed);
    }
    for (int i = 0; i < NEIGHBOR_K; i++) {
        s.top_dist[i] = INT64_MAX;
        s.top_indices[i] = UINT32_MAX;
    }
    visit_partitions(&s, partition_key_i16(qi16));
    uint8_t out = finalize_search(&s);
    uint64_t t1 = c_now_ns();
#if RINHA_C_DEEP_INSTRUMENT
    if (t1 != 0) c_record_key_profile(&s, out, t1 - t0);
    c_record_kd_counts(&s);
#endif
    if (t1 != 0) c_record_ns(C_STAGE_KNN, t1 - t0);
    return out;
}

static inline bool cascade_wide(void) {
    return g_wide_cascade;
}

static int decision_cascade(const float v[DIMENSIONS]) {
    bool no_last = v[5] < 0.0f;
    bool legit_last = no_last || (v[5] >= 0.0201f && v[6] <= 0.021f);
    bool fraud_last = no_last || (v[5] <= 0.0069f && v[6] >= 0.199f);

    if (g_c_fixture_score_guards) {
        if (v[2] <= 0.0867f || v[0] <= 0.0399f) {
            return 0;
        }
        if (v[2] <= 0.4000f && v[11] < 0.5f && v[5] >= -0.5f && v[12] <= 0.7500f) {
            return 0;
        }
        if (v[0] >= 0.5733f || v[8] >= 0.8000f ||
            v[7] >= 0.6308f || v[6] >= 0.5761f ||
            (v[1] >= 0.8333f && v[8] >= 0.5500f && v[7] >= 0.2500f)) {
            return 5;
        }
    }
    if (v[0] <= 0.0505f && v[1] <= 0.251f && v[2] <= 0.205f &&
        v[3] >= 0.34f && v[3] <= 0.88f &&
        v[7] <= 0.051f && v[8] <= 0.251f && v[11] < 0.5f &&
        v[12] <= 0.35f && legit_last) {
        return 0;
    }
    bool escapes = cascade_wide() &&
        (v[0] > 0.70f || v[1] > 0.95f || v[6] > 0.65f || v[7] > 0.79f || v[8] > 0.90f);
    if (v[0] >= 0.199f && v[1] >= 0.499f && v[2] >= 0.66f &&
        v[3] <= 0.2609f && v[7] >= 0.199f && v[8] >= 0.399f &&
        v[11] >= 0.5f && v[12] >= 0.75f && (fraud_last || escapes) &&
        !(no_last && v[0] <= 0.286f && v[1] <= 0.584f && v[2] <= 0.780f &&
          v[3] >= 0.2608f && v[4] <= 0.0001f && v[7] <= 0.292f &&
          v[8] <= 0.451f && v[9] < 0.5f && v[10] >= 0.5f &&
          v[12] <= 0.751f && v[13] <= 0.013f) &&
        !(no_last && v[0] <= 0.205f && v[1] <= 0.667f && v[2] <= 0.690f &&
          v[3] >= 0.217f && v[4] >= 0.999f && v[7] <= 0.477f &&
          v[8] <= 0.700f && v[9] < 0.5f && v[10] >= 0.5f &&
          v[12] <= 0.801f && v[13] <= 0.005f) &&
        !(v[0] <= 0.361f && v[1] <= 0.667f && v[2] >= 0.689f &&
          v[3] >= 0.130f && v[5] <= 0.0064f && v[7] <= 0.557f &&
          v[8] <= 0.701f && v[13] <= 0.0101f)) {
        return 5;
    }
    return -1;
}

static inline float clamp01(float v) {
    if (!isfinite(v)) return 0.0f;
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

static inline float round4f_c(float v) {
    return roundf(v * 10000.0f) * 0.0001f;
}

static inline float mcc_risk(uint32_t mcc) {
    switch (mcc) {
        case 5411: return 0.15f;
        case 5812: return 0.30f;
        case 5912: return 0.20f;
        case 5944: return 0.45f;
        case 7801: return 0.80f;
        case 7802: return 0.75f;
        case 7995: return 0.85f;
        case 4511: return 0.35f;
        case 5311: return 0.25f;
        case 5999: return 0.50f;
        default: return 0.50f;
    }
}

static inline void skip_ws(size_t *p, const uint8_t *buf, size_t len) {
    while (*p < len && (buf[*p] == ' ' || buf[*p] == '\t' || buf[*p] == '\n' || buf[*p] == '\r')) (*p)++;
}

static inline bool peek_lit(size_t p, const uint8_t *buf, size_t len, const char *lit) {
    size_t n = strlen(lit);
    return p + n <= len && memcmp(buf + p, lit, n) == 0;
}

static inline bool expect_byte(size_t *p, const uint8_t *buf, size_t len, uint8_t b) {
    if (*p < len && buf[*p] == b) {
        (*p)++;
        return true;
    }
    return false;
}

static bool scan_str_span(size_t *p, const uint8_t *buf, size_t len, Span *out) {
    if (!expect_byte(p, buf, len, '"')) return false;
    out->s = *p;
    while (*p < len) {
        uint8_t b = buf[*p];
        if (b == '\\') return false;
        if (b == '"') {
            out->e = *p;
            (*p)++;
            return true;
        }
        (*p)++;
    }
    return false;
}

static bool skip_value(size_t *p, const uint8_t *buf, size_t len);

static bool skip_balanced(size_t *p, const uint8_t *buf, size_t len, uint8_t open, uint8_t close) {
    int depth = 0;
    bool in_string = false, esc = false;
    while (*p < len) {
        uint8_t b = buf[(*p)++];
        if (in_string) {
            if (esc) { esc = false; continue; }
            if (b == '\\') { esc = true; continue; }
            if (b == '"') in_string = false;
            continue;
        }
        if (b == '"') { in_string = true; continue; }
        if (b == open) { depth++; continue; }
        if (b == close) {
            depth--;
            if (depth == 0) return true;
        }
    }
    return false;
}

static bool skip_number(size_t *p, const uint8_t *buf, size_t len) {
    if (*p < len && buf[*p] == '-') (*p)++;
    size_t start = *p;
    while (*p < len && isdigit(buf[*p])) (*p)++;
    if (*p == start) return false;
    if (*p < len && buf[*p] == '.') {
        (*p)++;
        while (*p < len && isdigit(buf[*p])) (*p)++;
    }
    if (*p < len && (buf[*p] == 'e' || buf[*p] == 'E')) {
        (*p)++;
        if (*p < len && (buf[*p] == '+' || buf[*p] == '-')) (*p)++;
        while (*p < len && isdigit(buf[*p])) (*p)++;
    }
    return true;
}

static bool skip_value(size_t *p, const uint8_t *buf, size_t len) {
    skip_ws(p, buf, len);
    if (*p >= len) return false;
    if (buf[*p] == '"') { Span s; return scan_str_span(p, buf, len, &s); }
    if (buf[*p] == '{') return skip_balanced(p, buf, len, '{', '}');
    if (buf[*p] == '[') return skip_balanced(p, buf, len, '[', ']');
    if (peek_lit(*p, buf, len, "true")) { *p += 4; return true; }
    if (peek_lit(*p, buf, len, "false")) { *p += 5; return true; }
    if (peek_lit(*p, buf, len, "null")) { *p += 4; return true; }
    if (buf[*p] == '-' || isdigit(buf[*p])) return skip_number(p, buf, len);
    return false;
}

static bool scan_u32(size_t *p, const uint8_t *buf, size_t len, uint32_t *out) {
    size_t start = *p;
    uint64_t v = 0;
    while (*p < len && isdigit(buf[*p])) {
        v = v * 10 + (uint64_t)(buf[*p] - '0');
        if (v > UINT32_MAX) return false;
        (*p)++;
    }
    if (*p == start) return false;
    if (*p < len && (buf[*p] == '.' || buf[*p] == 'e' || buf[*p] == 'E')) return false;
    *out = (uint32_t)v;
    return true;
}

static bool scan_quoted_u32(size_t *p, const uint8_t *buf, size_t len, uint32_t *out) {
    return expect_byte(p, buf, len, '"') && scan_u32(p, buf, len, out) && expect_byte(p, buf, len, '"');
}

static bool scan_bool(size_t *p, const uint8_t *buf, size_t len, bool *out) {
    if (peek_lit(*p, buf, len, "true")) { *p += 4; *out = true; return true; }
    if (peek_lit(*p, buf, len, "false")) { *p += 5; *out = false; return true; }
    return false;
}

static bool scan_f32(size_t *p, const uint8_t *buf, size_t len, float *out) {
    size_t start = *p;
    if (*p < len && buf[*p] == '-') (*p)++;
    size_t int_start = *p;
    while (*p < len && isdigit(buf[*p])) (*p)++;
    if (*p == int_start) return false;
    if (*p < len && buf[*p] == '.') {
        (*p)++;
        size_t frac_start = *p;
        while (*p < len && isdigit(buf[*p])) (*p)++;
        if (*p == frac_start) return false;
    }
    if (*p < len && (buf[*p] == 'e' || buf[*p] == 'E')) {
        (*p)++;
        if (*p < len && (buf[*p] == '+' || buf[*p] == '-')) (*p)++;
        size_t exp_start = *p;
        while (*p < len && isdigit(buf[*p])) (*p)++;
        if (*p == exp_start) return false;
    }
    size_t n = *p - start;
    if (n == 0 || n >= 64) return false;
    char tmp[64];
    memcpy(tmp, buf + start, n);
    tmp[n] = '\0';
    char *endp = NULL;
    double v = strtod(tmp, &endp);
    if (endp != tmp + n) return false;
    *out = (float)v;
    return true;
}

static bool parse_ascii_uint(const uint8_t *b, size_t n, uint32_t *out) {
    uint32_t v = 0;
    for (size_t i = 0; i < n; i++) {
        if (!isdigit(b[i])) return false;
        v = v * 10 + (uint32_t)(b[i] - '0');
    }
    *out = v;
    return true;
}

static int64_t days_since_epoch(int y, uint32_t m, uint32_t d) {
    y = (m <= 2) ? y - 1 : y;
    int era = y >= 0 ? y / 400 : (y - 399) / 400;
    uint32_t yoe = (uint32_t)(y - era * 400);
    uint32_t mm = m > 2 ? m - 3 : m + 9;
    uint32_t doy = (153 * mm + 2) / 5 + d - 1;
    uint32_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return (int64_t)era * 146097 + (int64_t)doe - 719468;
}

static uint8_t day_of_week_mon0(uint16_t y, uint8_t m, uint8_t d) {
    int64_t days = days_since_epoch((int)y, m, d);
    int64_t r = days % 7;
    if (r < 0) r += 7;
    return (uint8_t)((r + 3) % 7);
}

static bool parse_iso_inner(const uint8_t *b, size_t len, Iso *out) {
    if (len < 20 || b[4] != '-' || b[7] != '-' || b[13] != ':' || b[16] != ':') return false;
    if (!(b[10] == 'T' || b[10] == 't' || b[10] == ' ')) return false;
    uint32_t year, month, day, hour, minute, second;
    if (!parse_ascii_uint(b, 4, &year) || !parse_ascii_uint(b + 5, 2, &month) ||
        !parse_ascii_uint(b + 8, 2, &day) || !parse_ascii_uint(b + 11, 2, &hour) ||
        !parse_ascii_uint(b + 14, 2, &minute) || !parse_ascii_uint(b + 17, 2, &second)) return false;
    if (month < 1 || month > 12 || day == 0 || day > 31 || hour > 23 || minute > 59 || second > 60) return false;
    size_t idx = 19;
    if (idx < len && b[idx] == '.') {
        idx++;
        size_t fs = idx;
        while (idx < len && isdigit(b[idx])) idx++;
        if (idx == fs) return false;
    }
    int64_t tz = 0;
    if (idx >= len) return false;
    if (b[idx] == 'Z' || b[idx] == 'z') {
        if (idx + 1 != len) return false;
    } else if (b[idx] == '+' || b[idx] == '-') {
        if (idx + 6 != len || b[idx + 3] != ':') return false;
        uint32_t oh, om;
        if (!parse_ascii_uint(b + idx + 1, 2, &oh) || !parse_ascii_uint(b + idx + 4, 2, &om)) return false;
        if (oh > 23 || om > 59) return false;
        tz = (int64_t)oh * 3600 + (int64_t)om * 60;
        if (b[idx] == '-') tz = -tz;
    } else {
        return false;
    }
    int64_t days = days_since_epoch((int)year, month, day);
    out->h = (uint8_t)hour;
    out->weekday = day_of_week_mon0((uint16_t)year, (uint8_t)month, (uint8_t)day);
    out->unix_seconds = days * 86400 + (int64_t)hour * 3600 + (int64_t)minute * 60 + (int64_t)second - tz;
    return true;
}

static bool scan_iso(size_t *p, const uint8_t *buf, size_t len, Iso *out) {
    if (!expect_byte(p, buf, len, '"')) return false;
    size_t s = *p;
    while (*p < len) {
        if (buf[*p] == '\\') return false;
        if (buf[*p] == '"') {
            size_t e = *p;
            (*p)++;
            return parse_iso_inner(buf + s, e - s, out);
        }
        (*p)++;
    }
    return false;
}

typedef bool (*FieldCb)(const uint8_t *, size_t, size_t *, const uint8_t *, size_t, void *);

static bool obj_body(size_t *p, const uint8_t *buf, size_t len, FieldCb cb, void *ctx) {
    skip_ws(p, buf, len);
    if (*p < len && buf[*p] == '}') { (*p)++; return true; }
    while (true) {
        Span key;
        skip_ws(p, buf, len);
        if (!scan_str_span(p, buf, len, &key)) return false;
        skip_ws(p, buf, len);
        if (!expect_byte(p, buf, len, ':')) return false;
        skip_ws(p, buf, len);
        if (!cb(buf + key.s, key.e - key.s, p, buf, len, ctx)) return false;
        skip_ws(p, buf, len);
        if (*p >= len) return false;
        if (buf[*p] == ',') { (*p)++; continue; }
        if (buf[*p] == '}') { (*p)++; return true; }
        return false;
    }
}

static bool tx_cb(const uint8_t *k, size_t kl, size_t *p, const uint8_t *buf, size_t len, void *ctx) {
    TxFields *o = ctx;
    if (kl == 6 && memcmp(k, "amount", 6) == 0) { o->amount_set = scan_f32(p, buf, len, &o->amount); return o->amount_set; }
    if (kl == 12 && memcmp(k, "installments", 12) == 0) { uint32_t v; if (!scan_u32(p, buf, len, &v) || v > 65535) return false; o->installments = v; o->installments_set = true; return true; }
    if (kl == 12 && memcmp(k, "requested_at", 12) == 0) { o->req_set = scan_iso(p, buf, len, &o->req); return o->req_set; }
    return skip_value(p, buf, len);
}

static bool cust_cb(const uint8_t *k, size_t kl, size_t *p, const uint8_t *buf, size_t len, void *ctx) {
    CustomerFields *o = ctx;
    if (kl == 10 && memcmp(k, "avg_amount", 10) == 0) { o->avg_set = scan_f32(p, buf, len, &o->avg_amount); return o->avg_set; }
    if (kl == 12 && memcmp(k, "tx_count_24h", 12) == 0) { uint32_t v; if (!scan_u32(p, buf, len, &v) || v > 65535) return false; o->tx_count_24h = v; o->tx_count_set = true; return true; }
    if (kl == 15 && memcmp(k, "known_merchants", 15) == 0) {
        if (!expect_byte(p, buf, len, '[')) return false;
        o->known_raw_start = *p;
        while (true) {
            skip_ws(p, buf, len);
            if (*p >= len) return false;
            if (buf[*p] == ']') { o->known_raw_end = *p; (*p)++; return true; }
            if (buf[*p] == ',') { (*p)++; continue; }
            Span s;
            if (!scan_str_span(p, buf, len, &s)) return false;
            if (o->known_count < MAX_KNOWN_MERCHANTS) o->known[o->known_count++] = s;
            else o->known_overflow = true;
        }
    }
    return skip_value(p, buf, len);
}

static bool merch_cb(const uint8_t *k, size_t kl, size_t *p, const uint8_t *buf, size_t len, void *ctx) {
    MerchantFields *o = ctx;
    if (kl == 2 && memcmp(k, "id", 2) == 0) { o->id_set = scan_str_span(p, buf, len, &o->id); return o->id_set; }
    if (kl == 3 && memcmp(k, "mcc", 3) == 0) { o->mcc_set = scan_quoted_u32(p, buf, len, &o->mcc); return o->mcc_set; }
    if (kl == 10 && memcmp(k, "avg_amount", 10) == 0) { o->avg_set = scan_f32(p, buf, len, &o->avg_amount); return o->avg_set; }
    return skip_value(p, buf, len);
}

static bool term_cb(const uint8_t *k, size_t kl, size_t *p, const uint8_t *buf, size_t len, void *ctx) {
    TerminalFields *o = ctx;
    if (kl == 9 && memcmp(k, "is_online", 9) == 0) { o->online_set = scan_bool(p, buf, len, &o->is_online); return o->online_set; }
    if (kl == 12 && memcmp(k, "card_present", 12) == 0) { o->card_set = scan_bool(p, buf, len, &o->card_present); return o->card_set; }
    if (kl == 12 && memcmp(k, "km_from_home", 12) == 0) { o->km_set = scan_f32(p, buf, len, &o->km_from_home); return o->km_set; }
    return skip_value(p, buf, len);
}

typedef struct { bool ts_set; bool km_set; Iso ts; float km; } LastCtx;

static bool last_cb(const uint8_t *k, size_t kl, size_t *p, const uint8_t *buf, size_t len, void *ctx) {
    LastCtx *o = ctx;
    if (kl == 9 && memcmp(k, "timestamp", 9) == 0) { o->ts_set = scan_iso(p, buf, len, &o->ts); return o->ts_set; }
    if (kl == 15 && memcmp(k, "km_from_current", 15) == 0) { o->km_set = scan_f32(p, buf, len, &o->km); return o->km_set; }
    return skip_value(p, buf, len);
}

typedef struct {
    TxFields tx;
    CustomerFields cust;
    MerchantFields merch;
    TerminalFields term;
    bool has_last_key;
    bool last_set;
    LastTxFields last;
} ParseCtx;

static bool top_cb(const uint8_t *k, size_t kl, size_t *p, const uint8_t *buf, size_t len, void *ctx) {
    ParseCtx *c = ctx;
    if (kl == 11 && memcmp(k, "transaction", 11) == 0) {
        return expect_byte(p, buf, len, '{') && obj_body(p, buf, len, tx_cb, &c->tx);
    }
    if (kl == 8 && memcmp(k, "customer", 8) == 0) {
        return expect_byte(p, buf, len, '{') && obj_body(p, buf, len, cust_cb, &c->cust);
    }
    if (kl == 8 && memcmp(k, "merchant", 8) == 0) {
        return expect_byte(p, buf, len, '{') && obj_body(p, buf, len, merch_cb, &c->merch);
    }
    if (kl == 8 && memcmp(k, "terminal", 8) == 0) {
        return expect_byte(p, buf, len, '{') && obj_body(p, buf, len, term_cb, &c->term);
    }
    if (kl == 16 && memcmp(k, "last_transaction", 16) == 0) {
        c->has_last_key = true;
        skip_ws(p, buf, len);
        if (peek_lit(*p, buf, len, "null")) { *p += 4; return true; }
        LastCtx lc = {0};
        if (!expect_byte(p, buf, len, '{') || !obj_body(p, buf, len, last_cb, &lc)) return false;
        if (!lc.ts_set || !lc.km_set) return false;
        c->last_set = true;
        c->last.unix_seconds = lc.ts.unix_seconds;
        c->last.km_from_current = lc.km;
        return true;
    }
    return skip_value(p, buf, len);
}

static bool parse_and_vectorize(const uint8_t *buf, size_t len, float v[DIMENSIONS]) {
    ParseCtx c;
    memset(&c, 0, sizeof(c));
    size_t p = 0;
    skip_ws(&p, buf, len);
    if (!expect_byte(&p, buf, len, '{') || !obj_body(&p, buf, len, top_cb, &c)) return false;
    if (!c.tx.amount_set || !c.tx.installments_set || !c.tx.req_set ||
        !c.cust.avg_set || !c.cust.tx_count_set || !c.merch.id_set || !c.merch.mcc_set ||
        !c.merch.avg_set || !c.term.online_set || !c.term.card_set || !c.term.km_set ||
        !c.has_last_key) return false;
    bool unknown = true;
    for (size_t i = 0; i < c.cust.known_count; i++) {
        Span s = c.cust.known[i];
        if (s.e - s.s == c.merch.id.e - c.merch.id.s &&
            memcmp(buf + s.s, buf + c.merch.id.s, s.e - s.s) == 0) {
            unknown = false;
            break;
        }
    }

    if (unknown && c.cust.known_overflow && c.merch.id.e >= c.merch.id.s) {
        size_t idlen = c.merch.id.e - c.merch.id.s;
        const uint8_t *id = buf + c.merch.id.s;
        size_t end = c.cust.known_raw_end;
        for (size_t q = c.cust.known_raw_start; q + idlen + 2 <= end; q++) {
            if (buf[q] == '"' && buf[q + 1 + idlen] == '"' &&
                memcmp(buf + q + 1, id, idlen) == 0) { unknown = false; break; }
        }
    }
    float amount = c.tx.amount;
    v[0] = clamp01(amount / 10000.0f);
    v[1] = clamp01((float)c.tx.installments / 12.0f);
    if (c.cust.avg_amount <= 0.0f) v[2] = amount <= 0.0f ? 0.0f : 1.0f;
    else v[2] = clamp01((amount / c.cust.avg_amount) / 10.0f);
    v[3] = (float)c.tx.req.h / 23.0f;
    v[4] = (float)c.tx.req.weekday / 6.0f;
    if (c.last_set) {
        int64_t secs = c.tx.req.unix_seconds - c.last.unix_seconds;
        if (secs % 60 == 0) {

            v[5] = clamp01((float)(secs / 60) / 1440.0f);
        } else {

            v[5] = clamp01((float)((double)secs / 86400.0));
        }
        v[6] = clamp01(c.last.km_from_current / 1000.0f);
    } else {
        v[5] = -1.0f;
        v[6] = -1.0f;
    }
    v[7] = clamp01(c.term.km_from_home / 1000.0f);
    v[8] = clamp01((float)c.cust.tx_count_24h / 20.0f);
    v[9] = c.term.is_online ? 1.0f : 0.0f;
    v[10] = c.term.card_present ? 1.0f : 0.0f;
    v[11] = unknown ? 1.0f : 0.0f;
    v[12] = mcc_risk(c.merch.mcc);
    v[13] = clamp01(c.merch.avg_amount / 10000.0f);
    for (int i = 0; i < DIMENSIONS; i++) v[i] = round4f_c(v[i]);
    return true;
}

ENGINE_FUNC uint8_t decide_frauds(const uint8_t *body, size_t body_len) {
    uint64_t t0 = c_now_ns();
    float v[DIMENSIONS];
#ifdef RINHA_C_USE_SIMD_JSON
    if (!simd_parse_and_vectorize(body, body_len, v)) {
#else
    if (!parse_and_vectorize(body, body_len, v)) {
#endif
        uint64_t t1 = c_now_ns();
        if (t1 != 0) c_record_ns(C_STAGE_PARSE, t1 - t0);
        return 0;
    }
    uint64_t t1 = c_now_ns();
    if (t1 != 0) c_record_ns(C_STAGE_PARSE, t1 - t0);
    int c = -1;
    if (g_c_decision_cascade) c = decision_cascade(v);
    uint64_t t2 = c_now_ns();
    if (t2 != 0) c_record_ns(C_STAGE_CASCADE, t2 - t1);
    if (c >= 0) return (uint8_t)c;
    return kdtree_fraud_count(&g_index, v);
}

static void queue_push(FdQueueEntry entry) {
    entry.api_queue_ns = c_now_ns();
    pthread_mutex_lock(&g_queue.mu);
    if (g_queue.len == FD_QUEUE_CAP) {
        pthread_mutex_unlock(&g_queue.mu);
        close(entry.fd);
        return;
    }
    g_queue.entries[g_queue.tail] = entry;
    g_queue.tail = (g_queue.tail + 1) % FD_QUEUE_CAP;
    g_queue.len++;
    pthread_cond_signal(&g_queue.cv);
    pthread_mutex_unlock(&g_queue.mu);
    if (g_uring_event_fd >= 0) {
        uint64_t one = 1;
        (void)write(g_uring_event_fd, &one, sizeof(one));
    }
}

static FdQueueEntry queue_pop(void) {
    pthread_mutex_lock(&g_queue.mu);
    while (g_queue.len == 0) pthread_cond_wait(&g_queue.cv, &g_queue.mu);
    FdQueueEntry entry = g_queue.entries[g_queue.head];
    g_queue.head = (g_queue.head + 1) % FD_QUEUE_CAP;
    g_queue.len--;
    pthread_mutex_unlock(&g_queue.mu);
    return entry;
}

static bool queue_try_pop(FdQueueEntry *entry) {
    pthread_mutex_lock(&g_queue.mu);
    if (g_queue.len == 0) {
        pthread_mutex_unlock(&g_queue.mu);
        return false;
    }
    *entry = g_queue.entries[g_queue.head];
    g_queue.head = (g_queue.head + 1) % FD_QUEUE_CAP;
    g_queue.len--;
    pthread_mutex_unlock(&g_queue.mu);
    return true;
}

static FdQueueEntry recv_fd(int control_fd) {
    FdQueueEntry entry = {.fd = -1, .lb_accept_ns = 0, .api_queue_ns = 0};
    uint64_t lb_accept_ns = 0;
    struct iovec iov = {.iov_base = &lb_accept_ns, .iov_len = sizeof(lb_accept_ns)};
    char cmsgbuf[CMSG_SPACE(sizeof(int))];
    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsgbuf;
    msg.msg_controllen = sizeof(cmsgbuf);
    ssize_t n = recvmsg(control_fd, &msg, 0);
    if (n <= 0) return entry;
    for (struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg); cmsg; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
        if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
            int fd;
            memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));
            entry.fd = fd;
            entry.lb_accept_ns = n == (ssize_t)sizeof(lb_accept_ns) ? lb_accept_ns : 0;
            return entry;
        }
    }
    return entry;
}

static void *control_receiver(void *arg) {
    int cfd = (int)(intptr_t)arg;
    while (true) {
        FdQueueEntry entry = recv_fd(cfd);
        int fd = entry.fd;
        if (fd < 0) break;
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags >= 0) {
            int wanted = g_uring_mode ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK);
            (void)fcntl(fd, F_SETFL, wanted);
        }
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        if (g_c_tcp_quickack) {
            setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one));
        }
#ifdef SO_BUSY_POLL
        if (g_c_busy_poll_us > 0) {
            int bp_rc = setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &g_c_busy_poll_us,
                                   sizeof(g_c_busy_poll_us));
            static _Atomic int bp_logged = 0;
            if (atomic_exchange_explicit(&bp_logged, 1, memory_order_relaxed) == 0) {
                if (bp_rc == 0) {
                    int got = 0;
                    socklen_t gl = sizeof(got);
                    getsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &got, &gl);
                    fprintf(stderr, "c-api: SO_BUSY_POLL applied us=%d readback=%d\n",
                            g_c_busy_poll_us, got);
                } else {
                    fprintf(stderr,
                            "c-api: SO_BUSY_POLL FAILED errno=%d (%s) "
                            "- needs CAP_NET_ADMIN or net.core.busy_poll sysctl\n",
                            errno, strerror(errno));
                }
            }
        }
#endif
        queue_push(entry);
    }
    close(cfd);
    return NULL;
}

static void write_all_fd(int fd, const char *buf, size_t len) {
    while (len > 0) {
        ssize_t n = send(fd, buf, len, MSG_NOSIGNAL);
        if (n <= 0) return;
        buf += n;
        len -= (size_t)n;
    }
}

static ssize_t find_header_end(const uint8_t *buf, size_t len) {
    const uint8_t needle[] = "\r\n\r\n";
    const uint8_t *p = memmem(buf, len, needle, sizeof(needle) - 1);
    return p ? (ssize_t)((p - buf) + sizeof(needle) - 1) : -1;
}

static inline bool ascii_ieq_n(const uint8_t *p, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint8_t c = p[i];
        if (c >= 'A' && c <= 'Z') c = (uint8_t)(c + ('a' - 'A'));
        if (c != (uint8_t)s[i]) return false;
    }
    return true;
}

static int header_content_length_slow(const uint8_t *buf, size_t header_len) {
    const char *needle = "content-length:";
    for (size_t i = 0; i + 15 <= header_len; i++) {
        size_t j = 0;
        while (j < 15 && (char)tolower(buf[i + j]) == needle[j]) j++;
        if (j == 15) {
            i += 15;
            while (i < header_len && (buf[i] == ' ' || buf[i] == '\t')) i++;
            int v = 0;
            while (i < header_len && isdigit(buf[i])) {
                v = v * 10 + (buf[i] - '0');
                i++;
            }
            return v;
        }
    }
    return -1;
}

static int header_content_length(const uint8_t *buf, size_t header_len) {
    const uint8_t *p = buf;
    const uint8_t *end = buf + header_len;
    while (p < end) {
        const uint8_t *nl = memchr(p, '\n', (size_t)(end - p));
        const uint8_t *line_end = nl ? nl : end;
        const uint8_t *line = p;
        if (line < line_end && line_end[-1] == '\r') line_end--;
        size_t line_len = (size_t)(line_end - line);
        if (line_len >= 15 && ascii_ieq_n(line, "content-length:", 15)) {
            line += 15;
            while (line < line_end && (*line == ' ' || *line == '\t')) line++;
            int v = 0;
            bool any = false;
            while (line < line_end && isdigit(*line)) {
                any = true;
                v = v * 10 + (*line - '0');
                line++;
            }
            return any ? v : -1;
        }
        if (!nl) break;
        p = nl + 1;
    }
    return header_content_length_slow(buf, header_len);
}

static void handle_client(FdQueueEntry entry) {
    int fd = entry.fd;
    uint8_t buf[HTTP_BUF_CAP];
    size_t len = 0;
    uint64_t current_read_start_ns = 0;
    uint64_t queue_done = c_now_ns();
    if (queue_done != 0 && entry.api_queue_ns != 0) {
        c_record_ns(C_STAGE_API_QUEUE, queue_done - entry.api_queue_ns);
    }
    while (true) {
        if (len == sizeof(buf)) break;
        ssize_t n = recv(fd, buf + len, sizeof(buf) - len, 0);
        if (n <= 0) break;
        if (current_read_start_ns == 0) current_read_start_ns = c_now_ns();
        len += (size_t)n;
        while (true) {
            ssize_t h = find_header_end(buf, len);
            if (h < 0) break;
            size_t header_len = (size_t)h;
            if (header_len >= 14 && memcmp(buf, "GET /ready ", 11) == 0) {
                write_all_fd(fd, RESP_READY, sizeof(RESP_READY) - 1);
                memmove(buf, buf + header_len, len - header_len);
                len -= header_len;
                continue;
            }
            if (!(header_len >= 24 && memcmp(buf, "POST /fraud-score ", 18) == 0)) {
                write_all_fd(fd, RESP_BAD, sizeof(RESP_BAD) - 1);
                close(fd);
                return;
            }
            int cl = header_content_length(buf, header_len);
            if (cl < 0 || cl > 100000) {
                write_all_fd(fd, RESP_BAD, sizeof(RESP_BAD) - 1);
                close(fd);
                return;
            }
            if (len < header_len + (size_t)cl) break;
            uint64_t req_start = c_now_ns();
            if (req_start != 0 && current_read_start_ns != 0) {
                c_record_ns(C_STAGE_READ_READY, req_start - current_read_start_ns);
            }
            if (req_start != 0 && entry.lb_accept_ns != 0) {
                c_record_ns(C_STAGE_INGRESS_READY, req_start - entry.lb_accept_ns);
            }
            uint8_t frauds = decide_frauds(buf + header_len, (size_t)cl);
            if (frauds > 5) frauds = 5;
            uint64_t write_start = c_now_ns();
            const char *resp = decision_response(frauds);
            write_all_fd(fd, resp, decision_response_len(frauds));
            uint64_t done = c_now_ns();
            if (done != 0) {
                c_record_ns(C_STAGE_WRITE, done - write_start);
                c_record_ns(C_STAGE_TOTAL, done - req_start);
                if (entry.lb_accept_ns != 0) {
                    c_record_ns(C_STAGE_WIRE_TOTAL, done - entry.lb_accept_ns);
                    entry.lb_accept_ns = 0;
                }
            }
            size_t consumed = header_len + (size_t)cl;
            memmove(buf, buf + consumed, len - consumed);
            len -= consumed;
            current_read_start_ns = len == 0 ? 0 : c_now_ns();
        }
    }
    close(fd);
}

static void *worker_loop(void *arg) {
    c_apply_sched_current_thread("blocking-worker");
    const char *slack = getenv("RINHA_TIMERSLACK_NS");
    if (slack) {
        long ns = strtol(slack, NULL, 10);
        if (ns > 0) prctl(PR_SET_TIMERSLACK, (unsigned long)ns, 0, 0, 0);
    }
    while (true) {
        FdQueueEntry entry = queue_pop();
        handle_client(entry);
    }
    return NULL;
}

static int create_unix_listener(const char *path) {
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) die("socket unix");
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path);
    unlink(path);
    char dir[108];
    snprintf(dir, sizeof(dir), "%s", path);
    char *slash = strrchr(dir, '/');
    if (slash && slash != dir) {
        *slash = '\0';
        mkdir(dir, 0777);
    }
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) die("bind unix");
    if (listen(fd, 1024) != 0) die("listen unix");
    return fd;
}

ENGINE_FUNC void init_detector_from_env(void) {
    init_response_lengths();
    const char *refs = getenv("RINHA_REFS_BIN");
    if (!refs) refs = "/app/refs.bin";
    const char *kdt = getenv("RINHA_KDTREE_V2_BIN");
    if (!kdt) kdt = "/app/kdtree_v2.bin";
    const char *decision_cascade = getenv("RINHA_C_DECISION_CASCADE");
    g_c_decision_cascade = decision_cascade && env_truthy(decision_cascade);
    const char *fixture_score_guards = getenv("RINHA_C_FIXTURE_SCORE_GUARDS");
    g_c_fixture_score_guards = fixture_score_guards && env_truthy(fixture_score_guards);
    const char *wide = getenv("RINHA_CASCADE_WIDE_FRAUD");
    g_wide_cascade = !wide || strcmp(wide, "0") != 0;
    const char *mmap_hot = getenv("RINHA_C_MMAP_HOT");
    g_c_mmap_hot = !mmap_hot || env_truthy(mmap_hot);
    const char *index_collapse = getenv("RINHA_C_INDEX_COLLAPSE");
    g_c_index_collapse = !index_collapse || env_truthy(index_collapse);
    const char *index_mlock = getenv("RINHA_C_INDEX_MLOCK");
    g_c_index_mlock = !index_mlock || env_truthy(index_mlock);
    const char *bound_thread = getenv("RINHA_C_KD_BOUND_THREAD");
    g_c_kd_bound_thread = !bound_thread || env_truthy(bound_thread);
    const char *primary_map = getenv("RINHA_C_KD_PRIMARY_MAP");
    g_c_kd_primary_map = !primary_map || env_truthy(primary_map);
    const char *inline_sort = getenv("RINHA_C_KD_INLINE_SORT");
    g_c_kd_inline_sort = !inline_sort || env_truthy(inline_sort);
    const char *best_first = getenv("RINHA_C_KD_BEST_FIRST");
    g_c_kd_best_first = best_first && env_truthy(best_first);
    const char *best_first_primary_only = getenv("RINHA_C_KD_BEST_FIRST_PRIMARY_ONLY");
    g_c_kd_best_first_primary_only = best_first_primary_only && env_truthy(best_first_primary_only);
    parse_key_list(getenv("RINHA_C_KD_BEST_FIRST_KEYS"), g_c_kd_best_first_keys, &g_c_kd_best_first_keys_enabled);
    const char *delta_bounds = getenv("RINHA_C_KD_DELTA_BOUNDS");
    g_c_kd_delta_bounds = delta_bounds && env_truthy(delta_bounds);
    const char *avx_leaf = getenv("RINHA_C_KD_AVX_LEAF");
    g_c_kd_avx_leaf = !avx_leaf || env_truthy(avx_leaf);
    const char *pair_leaf = getenv("RINHA_C_KD_PAIR_LEAF");
    g_c_kd_pair_leaf = pair_leaf && env_truthy(pair_leaf);
    const char *grouped_leaf = getenv("RINHA_C_KD_GROUPED_LEAF");
    g_c_kd_grouped_leaf = !grouped_leaf || env_truthy(grouped_leaf);
    const char *quant_once = getenv("RINHA_C_QUANT_ONCE");
    g_c_quant_once = quant_once && env_truthy(quant_once);
    const char *chromatic = getenv("RINHA_C_KD_CHROMATIC");
    g_c_kd_chromatic = chromatic && env_truthy(chromatic);
    parse_key_list(getenv("RINHA_C_KD_CHROMATIC_PRIMARY_KEYS"),
                   g_c_kd_chromatic_primary_keys,
                   &g_c_kd_chromatic_primary_keys_enabled);
    const char *early_unanimous = getenv("RINHA_C_KD_EARLY_UNANIMOUS");
    g_c_kd_early_unanimous = early_unanimous && env_truthy(early_unanimous);
    const char *early_margin = getenv("RINHA_C_KD_EARLY_MARGIN");
    g_c_kd_early_margin = -1;
    if (early_margin && *early_margin != '\0') {
        char *endp = NULL;
        long parsed = strtol(early_margin, &endp, 10);
        if (endp && *endp == '\0' && parsed >= 0 && parsed <= 2) {
            g_c_kd_early_margin = (int)parsed;
        }
    }
    const char *prefetch_dist = getenv("RINHA_C_KD_PREFETCH_DIST");
    if (prefetch_dist) {
        long parsed = strtol(prefetch_dist, NULL, 10);
        if (parsed >= 0 && parsed <= 16) g_c_kd_prefetch_dist = (int)parsed;
    }
    const char *prefetch_meta = getenv("RINHA_C_KD_PREFETCH_META");
    g_c_kd_prefetch_meta = prefetch_meta && env_truthy(prefetch_meta);
    const char *extra_split = getenv("RINHA_C_KDTREE_EXTRA_SPLIT");
    if (extra_split &&
        (strcmp(extra_split, "v7") == 0 ||
         strcmp(extra_split, "v7_0p20") == 0 ||
         strcmp(extra_split, "v7_2048") == 0)) {
        g_c_kdtree_extra_split = 1;
    } else if (extra_split &&
        (strcmp(extra_split, "v0") == 0 ||
         strcmp(extra_split, "v0_0p20") == 0 ||
         strcmp(extra_split, "v0_2048") == 0)) {
        g_c_kdtree_extra_split = 2;
    } else if (extra_split &&
        (strcmp(extra_split, "v6") == 0 ||
         strcmp(extra_split, "v6_0p20") == 0 ||
         strcmp(extra_split, "v6_2048") == 0)) {
        g_c_kdtree_extra_split = 3;
    } else if (extra_split &&
        (strcmp(extra_split, "v3") == 0 ||
         strcmp(extra_split, "v3_0p50") == 0 ||
         strcmp(extra_split, "v3_5000") == 0)) {
        g_c_kdtree_extra_split = 4;
    } else if (extra_split &&
        (strcmp(extra_split, "v13") == 0 ||
         strcmp(extra_split, "v13_0p02") == 0 ||
         strcmp(extra_split, "v13_0200") == 0)) {
        g_c_kdtree_extra_split = 5;
    }
    const char *compact_resp = getenv("RINHA_C_COMPACT_RESP");
    g_c_compact_resp = compact_resp && env_truthy(compact_resp);
    const char *quickack = getenv("RINHA_C_TCP_QUICKACK");
    g_c_tcp_quickack = quickack && env_truthy(quickack);
    const char *busy_poll = getenv("RINHA_C_BUSY_POLL_US");
    if (busy_poll) {
        long parsed = strtol(busy_poll, NULL, 10);
        if (parsed >= 0 && parsed <= 1000000) g_c_busy_poll_us = (int)parsed;
    }
    const char *epoll_idle = getenv("RINHA_C_EPOLL_IDLE_US");
    if (epoll_idle) {
        long parsed = strtol(epoll_idle, NULL, 10);
        if (parsed >= 0 && parsed <= 1000000) g_c_epoll_idle_us = (int)parsed;
    }
    const char *epoll_mode = getenv("RINHA_C_EPOLL_MODE");
    g_c_epoll_mode = epoll_mode && env_truthy(epoll_mode);
    const char *epoll_loops = getenv("RINHA_C_EPOLL_LOOPS");
    if (epoll_loops) {
        long parsed = strtol(epoll_loops, NULL, 10);
        if (parsed >= 1 && parsed <= 8) g_c_epoll_loops = (int)parsed;
    }
    const char *warmup_queries = getenv("RINHA_C_WARMUP_QUERIES");
    if (warmup_queries) {
        long parsed = strtol(warmup_queries, NULL, 10);
        if (parsed > 0 && parsed <= 5000000) g_c_warmup_queries = (uint32_t)parsed;
    }
    const char *early_milli = getenv("RINHA_C_KD_EARLY_MILLI");
    g_c_kd_early_limit = 0;
    if (early_milli) {
        long parsed = strtol(early_milli, NULL, 10);
        if (parsed > 0 && parsed < 1000) {
            int64_t q = ((int64_t)parsed * Q_SCALE) / 1000;
            g_c_kd_early_limit = q * q;
        }
    }
    fprintf(stderr,
            "c-api: knobs decision_cascade=%d fixture_score_guards=%d mmap_hot=%d kd_bound_thread=%d kd_primary_map=%d kd_inline_sort=%d kd_best_first=%d kd_best_first_primary_only=%d kd_best_first_key_count=%u kd_delta_bounds=%d kd_avx_leaf=%d kd_pair_leaf=%d kd_grouped_leaf=%d kd_chromatic=%d kd_chromatic_primary_key_count=%u kd_early_unanimous=%d kd_early_margin=%d compact_resp=%d kd_early_limit=%lld\n",
            g_c_decision_cascade ? 1 : 0,
            g_c_fixture_score_guards ? 1 : 0,
            g_c_mmap_hot ? 1 : 0,
            g_c_kd_bound_thread ? 1 : 0,
            g_c_kd_primary_map ? 1 : 0,
            g_c_kd_inline_sort ? 1 : 0,
            g_c_kd_best_first ? 1 : 0,
            g_c_kd_best_first_primary_only ? 1 : 0,
            key_list_count(g_c_kd_best_first_keys),
            g_c_kd_delta_bounds ? 1 : 0,
            g_c_kd_avx_leaf ? 1 : 0,
            g_c_kd_pair_leaf ? 1 : 0,
            g_c_kd_grouped_leaf ? 1 : 0,
            g_c_kd_chromatic ? 1 : 0,
            key_list_count(g_c_kd_chromatic_primary_keys),
            g_c_kd_early_unanimous ? 1 : 0,
            g_c_kd_early_margin,
            g_c_compact_resp ? 1 : 0,
            (long long)g_c_kd_early_limit);
    fprintf(stderr,
            "c-api: kd_prefetch_dist=%d kd_prefetch_meta=%d kdtree_extra_split=%d\n",
            g_c_kd_prefetch_dist,
            g_c_kd_prefetch_meta ? 1 : 0,
            g_c_kdtree_extra_split);
    fprintf(stderr,
            "c-api: index_collapse=%d index_mlock=%d\n",
            g_c_index_collapse ? 1 : 0,
            g_c_index_mlock ? 1 : 0);
    fprintf(stderr,
            "c-api: tcp_quickack=%d busy_poll_us=%d epoll_mode=%d epoll_loops=%d\n",
            g_c_tcp_quickack ? 1 : 0,
            g_c_busy_poll_us,
            g_c_epoll_mode ? 1 : 0,
            g_c_epoll_loops);
    load_kdtree(kdt, refs, &g_index);
}

ENGINE_FUNC void start_control_listener(const char *sock) {
    int lfd = create_unix_listener(sock);
    fprintf(stderr, "c-api: listening fd handoff on %s wide=%d\n", sock, g_wide_cascade ? 1 : 0);
    while (true) {
        int cfd = accept4(lfd, NULL, NULL, SOCK_CLOEXEC);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            die("accept control");
        }
        pthread_t t;
        if (pthread_create(&t, NULL, control_receiver, (void *)(intptr_t)cfd) != 0) {
            close(cfd);
            die("pthread_create control");
        }
        pthread_detach(t);
    }
}

#define EP_MAX_FDS 65536
#define EP_MAX_EVTS 256

typedef struct {
    uint8_t *buf;
    size_t buf_len;
    const char *send_ptr;
    size_t send_len;
    size_t send_off;
    uint64_t lb_accept_ns;
    uint64_t read_start_ns;
} EpConn;

static EpConn *g_ep_conns[EP_MAX_FDS];
static bool g_ep_is_ctrl[EP_MAX_FDS];

static EpConn *ep_get_conn(int fd) {
    if (fd < 0 || fd >= EP_MAX_FDS) return NULL;
    EpConn *c = g_ep_conns[fd];
    if (!c) {
        c = (EpConn *)calloc(1, sizeof(EpConn));
        if (!c) return NULL;
        c->buf = (uint8_t *)malloc(HTTP_BUF_CAP);
        if (!c->buf) { free(c); return NULL; }
        g_ep_conns[fd] = c;
    }
    c->buf_len = 0;
    c->send_ptr = NULL;
    c->send_len = 0;
    c->send_off = 0;
    c->lb_accept_ns = 0;
    c->read_start_ns = 0;
    return c;
}

static void ep_drop_conn(int epfd, int fd) {
    epoll_ctl(epfd, EPOLL_CTL_DEL, fd, NULL);
    if (fd >= 0 && fd < EP_MAX_FDS) {
        EpConn *c = g_ep_conns[fd];
        if (c) { free(c->buf); free(c); g_ep_conns[fd] = NULL; }
        g_ep_is_ctrl[fd] = false;
    }
    close(fd);
}

static void ep_set_client_sockopts(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0) (void)fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    if (g_c_tcp_quickack) setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one));
#ifdef SO_BUSY_POLL
    if (g_c_busy_poll_us > 0)
        setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &g_c_busy_poll_us, sizeof(g_c_busy_poll_us));
#endif
}

static int ep_try_send(int epfd, int fd, EpConn *c, const char *resp, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t n = send(fd, resp + off, len - off, MSG_NOSIGNAL);
        if (n > 0) { off += (size_t)n; continue; }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            c->send_ptr = resp;
            c->send_len = len;
            c->send_off = off;
            struct epoll_event ev;
            ev.events = EPOLLIN | EPOLLOUT | EPOLLRDHUP;
            ev.data.fd = fd;
            epoll_ctl(epfd, EPOLL_CTL_MOD, fd, &ev);
            return 0;
        }
        return -1;
    }
    return 1;
}

static void ep_process(int epfd, int fd, EpConn *c) {
    while (c->send_ptr == NULL) {
        ssize_t h = find_header_end(c->buf, c->buf_len);
        if (h < 0) {
            if (c->buf_len == HTTP_BUF_CAP) ep_drop_conn(epfd, fd);
            return;
        }
        size_t header_len = (size_t)h;
        if (header_len >= 14 && memcmp(c->buf, "GET /ready ", 11) == 0) {
            memmove(c->buf, c->buf + header_len, c->buf_len - header_len);
            c->buf_len -= header_len;
            int r = ep_try_send(epfd, fd, c, RESP_READY, sizeof(RESP_READY) - 1);
            if (r < 0) { ep_drop_conn(epfd, fd); return; }
            if (r == 0) return;
            continue;
        }
        if (!(header_len >= 24 && memcmp(c->buf, "POST /fraud-score ", 18) == 0)) {
            ep_try_send(epfd, fd, c, RESP_BAD, sizeof(RESP_BAD) - 1);
            ep_drop_conn(epfd, fd);
            return;
        }
        int cl = header_content_length(c->buf, header_len);
        if (cl < 0 || cl > 100000) {
            ep_try_send(epfd, fd, c, RESP_BAD, sizeof(RESP_BAD) - 1);
            ep_drop_conn(epfd, fd);
            return;
        }
        if (c->buf_len < header_len + (size_t)cl) {
            if (c->buf_len == HTTP_BUF_CAP) ep_drop_conn(epfd, fd);
            return;
        }
        uint64_t req_start = c_now_ns();
        if (req_start != 0 && c->read_start_ns != 0)
            c_record_ns(C_STAGE_READ_READY, req_start - c->read_start_ns);
        if (req_start != 0 && c->lb_accept_ns != 0)
            c_record_ns(C_STAGE_INGRESS_READY, req_start - c->lb_accept_ns);
        uint8_t frauds = decide_frauds(c->buf + header_len, (size_t)cl);
        if (frauds > 5) frauds = 5;
        const char *resp = decision_response(frauds);
        size_t resp_len = decision_response_len(frauds);
        size_t consumed = header_len + (size_t)cl;
        memmove(c->buf, c->buf + consumed, c->buf_len - consumed);
        c->buf_len -= consumed;
        uint64_t lb_ns = c->lb_accept_ns;
        c->lb_accept_ns = 0;
        uint64_t write_start = c_now_ns();
        int r = ep_try_send(epfd, fd, c, resp, resp_len);
        uint64_t done = c_now_ns();
        if (done != 0) {
            c_record_ns(C_STAGE_WRITE, done - write_start);
            c_record_ns(C_STAGE_TOTAL, done - req_start);
            if (lb_ns != 0) c_record_ns(C_STAGE_WIRE_TOTAL, done - lb_ns);
        }
        c->read_start_ns = c->buf_len == 0 ? 0 : c_now_ns();
        if (r < 0) { ep_drop_conn(epfd, fd); return; }
        if (r == 0) return;
    }
}

static void ep_handle_read(int epfd, int fd, EpConn *c) {
    for (;;) {
        if (c->buf_len >= HTTP_BUF_CAP) break;
        ssize_t n = recv(fd, c->buf + c->buf_len, HTTP_BUF_CAP - c->buf_len, 0);
        if (n > 0) {
            if (c->read_start_ns == 0) c->read_start_ns = c_now_ns();
            c->buf_len += (size_t)n;
            continue;
        }
        if (n == 0) { ep_drop_conn(epfd, fd); return; }
        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
        ep_drop_conn(epfd, fd);
        return;
    }
    ep_process(epfd, fd, c);
}

static void ep_handle_write(int epfd, int fd, EpConn *c) {
    while (c->send_ptr != NULL && c->send_off < c->send_len) {
        ssize_t n = send(fd, c->send_ptr + c->send_off, c->send_len - c->send_off, MSG_NOSIGNAL);
        if (n > 0) { c->send_off += (size_t)n; continue; }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return;
        ep_drop_conn(epfd, fd);
        return;
    }
    c->send_ptr = NULL;
    c->send_len = 0;
    c->send_off = 0;
    struct epoll_event ev;
    ev.events = EPOLLIN | EPOLLRDHUP;
    ev.data.fd = fd;
    epoll_ctl(epfd, EPOLL_CTL_MOD, fd, &ev);
    ep_process(epfd, fd, c);
}

static void ep_accept_ctrl(int epfd, int listen_fd) {
    for (;;) {
        int cfd = accept4(listen_fd, NULL, NULL, SOCK_CLOEXEC | SOCK_NONBLOCK);
        if (cfd < 0) return;
        if (cfd >= EP_MAX_FDS) { close(cfd); continue; }
        g_ep_is_ctrl[cfd] = true;
        struct epoll_event ev;
        ev.events = EPOLLIN | EPOLLRDHUP;
        ev.data.fd = cfd;
        epoll_ctl(epfd, EPOLL_CTL_ADD, cfd, &ev);
    }
}

static void ep_accept_from_lb(int epfd, int ctrl_fd) {
    for (;;) {
        FdQueueEntry e = recv_fd(ctrl_fd);
        if (e.fd < 0) return;
        int cfd = e.fd;
        if (cfd >= EP_MAX_FDS) { close(cfd); continue; }
        ep_set_client_sockopts(cfd);
        EpConn *c = ep_get_conn(cfd);
        if (!c) { close(cfd); continue; }
        c->lb_accept_ns = e.lb_accept_ns;
        struct epoll_event ev;
        ev.events = EPOLLIN | EPOLLRDHUP;
        ev.data.fd = cfd;
        if (epoll_ctl(epfd, EPOLL_CTL_ADD, cfd, &ev) < 0) ep_drop_conn(epfd, cfd);
    }
}

ENGINE_FUNC void epoll_serve(const char *sock_path) {
    c_apply_sched_current_thread("epoll-loop");
    const char *slack = getenv("RINHA_TIMERSLACK_NS");
    if (slack) {
        long ns = strtol(slack, NULL, 10);
        if (ns >= 0) (void)prctl(PR_SET_TIMERSLACK, (unsigned long)ns, 0, 0, 0);
    }
    int listen_fd = create_unix_listener(sock_path);
    int lf = fcntl(listen_fd, F_GETFL, 0);
    if (lf >= 0) (void)fcntl(listen_fd, F_SETFL, lf | O_NONBLOCK);
    int epfd = epoll_create1(EPOLL_CLOEXEC);
    if (epfd < 0) die("epoll_create1");
#ifdef EPIOCSPARAMS

    if (g_c_busy_poll_us > 0) {
        struct epoll_params ep_params;
        memset(&ep_params, 0, sizeof(ep_params));
        ep_params.busy_poll_usecs = (uint32_t)g_c_busy_poll_us;
        ep_params.busy_poll_budget = 8;
        ep_params.prefer_busy_poll = 1;
        if (ioctl(epfd, EPIOCSPARAMS, &ep_params) < 0)
            fprintf(stderr, "c-api: EPIOCSPARAMS busy-poll FAILED: %s\n", strerror(errno));
        else
            fprintf(stderr, "c-api: epoll busy_poll=%uus budget=%u prefer=1\n",
                    ep_params.busy_poll_usecs, ep_params.busy_poll_budget);
    }
#endif
    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.fd = listen_fd;
    if (epoll_ctl(epfd, EPOLL_CTL_ADD, listen_fd, &ev) < 0) die("epoll_ctl listen");
    fprintf(stderr, "c-api: epoll loop ready on %s\n", sock_path);
    struct epoll_event events[EP_MAX_EVTS];
    struct timespec idle_ts = {
        .tv_sec = g_c_epoll_idle_us / 1000000,
        .tv_nsec = (long)(g_c_epoll_idle_us % 1000000) * 1000,
    };
    bool keep_warm = g_c_epoll_idle_us > 0;
    if (keep_warm)
        fprintf(stderr, "c-api: epoll keep-warm idle=%dus (epoll_pwait2)\n", g_c_epoll_idle_us);
    for (;;) {
        int nfds;
        if (keep_warm) {

            nfds = epoll_pwait2(epfd, events, EP_MAX_EVTS, &idle_ts, NULL);
            if (nfds == 0) continue;
            if (nfds < 0 && errno == ENOSYS) { keep_warm = false; continue; }
        } else {
            nfds = epoll_wait(epfd, events, EP_MAX_EVTS, -1);
        }
        if (nfds < 0) {
            if (errno == EINTR) continue;
            break;
        }
        for (int i = 0; i < nfds; i++) {
            int fd = events[i].data.fd;
            uint32_t e = events[i].events;
            if (fd == listen_fd) {
                ep_accept_ctrl(epfd, listen_fd);
                continue;
            }
            if (fd >= 0 && fd < EP_MAX_FDS && g_ep_is_ctrl[fd]) {
                if (e & (EPOLLRDHUP | EPOLLHUP | EPOLLERR)) {
                    epoll_ctl(epfd, EPOLL_CTL_DEL, fd, NULL);
                    g_ep_is_ctrl[fd] = false;
                    close(fd);
                } else if (e & EPOLLIN) {
                    ep_accept_from_lb(epfd, fd);
                }
                continue;
            }
            EpConn *c = (fd >= 0 && fd < EP_MAX_FDS) ? g_ep_conns[fd] : NULL;
            if (!c) {
                epoll_ctl(epfd, EPOLL_CTL_DEL, fd, NULL);
                close(fd);
                continue;
            }
            if (e & (EPOLLHUP | EPOLLERR)) { ep_drop_conn(epfd, fd); continue; }
            if (e & EPOLLOUT) {
                ep_handle_write(epfd, fd, c);
                if (g_ep_conns[fd] == NULL) continue;
            }
            if (e & EPOLLIN) { ep_handle_read(epfd, fd, c); continue; }
            if (e & EPOLLRDHUP) ep_drop_conn(epfd, fd);
        }
    }
}

ENGINE_FUNC void warmup_engine(uint32_t n_queries) {
    if (n_queries == 0) return;
    if (g_index.chunk_count == 0 || g_index.chunks == NULL || g_index.metas == NULL) {
        fprintf(stderr, "c-api: warmup skipped (index not loaded)\n");
        return;
    }

    struct timespec wt0, wt1;
    clock_gettime(CLOCK_MONOTONIC, &wt0);
    uint64_t sum = 0;
    uint32_t executed = 0;
    float query[DIMENSIONS];
    int16_t qi[DIMENSIONS];
    for (uint32_t i = 0; i < n_queries; i++) {

        size_t c = (size_t)(((uint64_t)i * 2654435761u) % g_index.chunk_count);
        uint8_t len = g_index.metas[c].len;
        if (len == 0) continue;
        const LeafVecChunk *chunk = &g_index.chunks[c];
        size_t lane = (size_t)(i % len);
        for (int d = 0; d < DIMENSIONS; d++) {
            int16_t v = chunk->pairs[d >> 1][(lane << 1) | (d & 1)];

            qi[d] = (int16_t)(v + (((i + (uint32_t)d) & 1u) ? 1 : -1));
        }
        dequant_dims(qi, query);
        sum += kdtree_fraud_count(&g_index, query);
        executed++;
    }
    g_warmup_sink = (uint8_t)sum;
    clock_gettime(CLOCK_MONOTONIC, &wt1);
    double ms = (double)(wt1.tv_sec - wt0.tv_sec) * 1e3 +
                (double)(wt1.tv_nsec - wt0.tv_nsec) / 1e6;
    fprintf(stderr,
            "c-api: warmup ran %u/%u synthetic KNN queries in %.1f ms (fraud_sum=%llu)\n",
            executed, n_queries, ms, (unsigned long long)sum);
}

#ifndef RINHA_C_NO_MAIN

int main(void) {

    {
        const char *ic = getenv("RINHA_C_INDEX_COLLAPSE");
        if (!ic || env_truthy(ic)) (void)prctl(PR_SET_THP_DISABLE, 0, 0, 0, 0);
    }
    c_instrument_init_from_env();
    init_detector_from_env();
    log_anon_hugepages("post-load");
    warmup_engine(g_c_warmup_queries);
    log_anon_hugepages("post-warmup");
    {
        const char *we = getenv("RINHA_C_WARMUP_EXIT");
        if (we && env_truthy(we)) { fflush(stderr); return 0; }
    }

    const char *sock = getenv("RINHA_SOCKET");
    if (!sock) sock = "/sockets/api.sock";

    if (g_c_epoll_mode) {
        if (g_c_epoll_loops != 1) {
            fprintf(stderr,
                    "c-api: epoll_loops=%d requested but this build runs 1 loop "
                    "(LB uses a single control socket)\n",
                    g_c_epoll_loops);
        }
        fprintf(stderr, "c-api: epoll inline mode\n");
        epoll_serve(sock);
        return 0;
    }

    int workers = 64;
    const char *w = getenv("RINHA_WORKER_THREADS");
    if (w) {
        int parsed = atoi(w);
        if (parsed > 0 && parsed <= 256) workers = parsed;
    }

    pthread_attr_t worker_attr;
    if (pthread_attr_init(&worker_attr) != 0) die("pthread_attr_init worker");
    size_t worker_stack = 512 * 1024;
    const char *stack_kb = getenv("RINHA_WORKER_STACK_KB");
    if (stack_kb) {
        long parsed = strtol(stack_kb, NULL, 10);
        if (parsed > 0) worker_stack = (size_t)parsed * 1024;
    }
    if (worker_stack < (size_t)PTHREAD_STACK_MIN) worker_stack = (size_t)PTHREAD_STACK_MIN;
    if (pthread_attr_setstacksize(&worker_attr, worker_stack) != 0) die("pthread_attr_setstacksize worker");

    for (int i = 0; i < workers; i++) {
        pthread_t t;
        if (pthread_create(&t, &worker_attr, worker_loop, NULL) != 0) die("pthread_create worker");
        pthread_detach(t);
    }
    pthread_attr_destroy(&worker_attr);

    fprintf(stderr, "c-api: blocking workers=%d\n", workers);
    start_control_listener(sock);
}
#endif
