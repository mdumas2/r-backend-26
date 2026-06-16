#define _GNU_SOURCE 
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <sys/socket.h>
#include <sys/prctl.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>
#define MAX_UPSTREAMS 16
#define LISTEN_BACKLOG 4096
#define LB_INSTRUMENT_BUCKETS 2048
#define LB_BATCH_BUCKETS 64
#define LB_CONNECT_ATTEMPTS 50
#define LB_CONNECT_RETRY_US 2000
#define LB_BREAKER_COOLDOWN_NS 500000000ull
typedef struct {
    char path[sizeof(((struct sockaddr_un *)0)->sun_path)];
    int fd;
    uint64_t down_until_ns;
} Upstream;
static Upstream g_upstreams[MAX_UPSTREAMS];
static size_t g_upstream_count = 0;
static uint32_t g_next = 0;
static bool g_nonblock_accept = false;
static bool g_accept_batch = false;
static int g_defer_accept_secs = 0;
static bool g_lb_instrument = false;
static unsigned g_lb_instrument_interval_secs = 5;
enum {
    LB_STAGE_CONNECT = 0,
    LB_STAGE_SENDMSG,
    LB_STAGE_HANDOFF,
    LB_STAGE_POST_ACCEPT,
    LB_STAGE_COUNT,
};
static const char *LB_STAGE_NAMES[LB_STAGE_COUNT] = {
    "connect",
    "sendmsg",
    "handoff",
    "post_accept",
};
static _Atomic uint64_t g_lb_hist[LB_STAGE_COUNT][LB_INSTRUMENT_BUCKETS];
static _Atomic uint64_t g_lb_batch_hist[LB_BATCH_BUCKETS];
static void die(const char *msg) {
    perror(msg);
    exit(1);
}
static bool env_truthy(const char *value) {
    if (!value || value[0] == '\0') return false;
    return strcmp(value, "0") != 0 &&
           strcasecmp(value, "false") != 0 &&
           strcasecmp(value, "no") != 0 &&
           strcasecmp(value, "off") != 0;
}
static inline uint64_t lb_now_ns(void) {
    if (!g_lb_instrument) return 0;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
static inline uint64_t lb_mono_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
static inline void lb_record_ns(int stage, uint64_t ns) {
    if (!g_lb_instrument || stage < 0 || stage >= LB_STAGE_COUNT) return;
    uint64_t us = ns / 1000ull;
    if (us >= LB_INSTRUMENT_BUCKETS) us = LB_INSTRUMENT_BUCKETS - 1;
    atomic_fetch_add_explicit(&g_lb_hist[stage][us], 1, memory_order_relaxed);
}
static void lb_dump_stage(int stage) {
    uint64_t total = 0;
    uint64_t max_us = 0;
    uint64_t weighted_us = 0;
    for (size_t i = 0; i < LB_INSTRUMENT_BUCKETS; i++) {
        uint64_t n = atomic_load_explicit(&g_lb_hist[stage][i], memory_order_relaxed);
        total += n;
        weighted_us += n * i;
        if (n != 0) max_us = i;
    }
    if (total == 0) return;
    uint64_t p50_at = (total * 50 + 99) / 100;
    uint64_t p95_at = (total * 95 + 99) / 100;
    uint64_t p99_at = (total * 99 + 99) / 100;
    uint64_t seen = 0, p50 = 0, p95 = 0, p99 = 0;
    for (size_t i = 0; i < LB_INSTRUMENT_BUCKETS; i++) {
        uint64_t n = atomic_load_explicit(&g_lb_hist[stage][i], memory_order_relaxed);
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
            "c_lb_instrument: stage=%s n=%llu avg_us=%.1f p50_us=%llu p95_us=%llu p99_us=%llu max_us=%llu\n",
            LB_STAGE_NAMES[stage],
            (unsigned long long)total,
            avg,
            (unsigned long long)p50,
            (unsigned long long)p95,
            (unsigned long long)p99,
            (unsigned long long)max_us);
}
static inline void lb_record_batch(unsigned n) {
    if (n >= LB_BATCH_BUCKETS) n = LB_BATCH_BUCKETS - 1;
    atomic_fetch_add_explicit(&g_lb_batch_hist[n], 1, memory_order_relaxed);
}
static void lb_dump_batch(void) {
    uint64_t total = 0, weighted = 0, max_n = 0, ge2 = 0, n5plus = 0;
    uint64_t c[5] = {0, 0, 0, 0, 0};
    for (size_t i = 0; i < LB_BATCH_BUCKETS; i++) {
        uint64_t v = atomic_load_explicit(&g_lb_batch_hist[i], memory_order_relaxed);
        total += v;
        weighted += v * i;
        if (v) max_n = i;
        if (i >= 2) ge2 += v;
        if (i < 5) c[i] = v; else n5plus += v;
    }
    if (total == 0) return;
    double avg = (double)weighted / (double)total;
    double pct_ge2 = 100.0 * (double)ge2 / (double)total;
    fprintf(stderr,
            "c_lb_accept_batch: loops=%llu avg_n=%.4f max_n=%llu pct_n_ge2=%.3f%% n0=%llu n1=%llu n2=%llu n3=%llu n4=%llu n5plus=%llu\n",
            (unsigned long long)total, avg, (unsigned long long)max_n, pct_ge2,
            (unsigned long long)c[0], (unsigned long long)c[1],
            (unsigned long long)c[2], (unsigned long long)c[3],
            (unsigned long long)c[4], (unsigned long long)n5plus);
}
static void *lb_instrument_loop(void *arg) {
    (void)arg;
    while (true) {
        sleep(g_lb_instrument_interval_secs);
        for (int stage = 0; stage < LB_STAGE_COUNT; stage++) lb_dump_stage(stage);
        lb_dump_batch();
    }
    return NULL;
}
static void lb_instrument_init_from_env(void) {
    g_lb_instrument = env_truthy(getenv("RINHA_C_LB_INSTRUMENT"));
    if (!g_lb_instrument) return;
    const char *interval = getenv("RINHA_C_LB_INSTRUMENT_INTERVAL_SECS");
    if (interval && interval[0] != '\0') {
        long parsed = strtol(interval, NULL, 10);
        if (parsed > 0 && parsed < 3600) g_lb_instrument_interval_secs = (unsigned)parsed;
    }
    fprintf(stderr, "c_lb_instrument: enabled interval_secs=%u bucket_us=1 cap_us=%u\n",
            g_lb_instrument_interval_secs, LB_INSTRUMENT_BUCKETS - 1);
    pthread_t t;
    if (pthread_create(&t, NULL, lb_instrument_loop, NULL) == 0) {
        pthread_detach(t);
    }
}
static void apply_timerslack_from_env(void) {
    const char *slack = getenv("RINHA_C_LB_TIMERSLACK_NS");
    if (!slack || slack[0] == '\0') return;
    long ns = strtol(slack, NULL, 10);
    if (ns > 0) prctl(PR_SET_TIMERSLACK, (unsigned long)ns, 0, 0, 0);
}
static void apply_sched_from_env(void) {
    const char *raw = getenv("RINHA_C_LB_SCHED_POLICY");
    if (!raw || raw[0] == '\0' || !env_truthy(raw)) return;
    int policy;
    const char *name;
    if (strcasecmp(raw, "fifo") == 0 || strcmp(raw, "1") == 0 || strcasecmp(raw, "true") == 0) {
        policy = SCHED_FIFO;
        name = "SCHED_FIFO";
    } else if (strcasecmp(raw, "rr") == 0) {
        policy = SCHED_RR;
        name = "SCHED_RR";
    } else {
        fprintf(stderr, "c-lb: unknown RINHA_C_LB_SCHED_POLICY=%s, staying default\n", raw);
        return;
    }
    int prio = 5;
    const char *prio_env = getenv("RINHA_C_LB_SCHED_PRIO");
    if (prio_env && prio_env[0] != '\0') {
        long parsed = strtol(prio_env, NULL, 10);
        if (parsed > 0 && parsed < 100) prio = (int)parsed;
    }
    struct sched_param sp = {.sched_priority = prio};
    if (sched_setscheduler(0, policy, &sp) == 0) {
        fprintf(stderr, "c-lb: scheduler %s prio=%d\n", name, prio);
    } else {
        fprintf(stderr, "c-lb: sched_setscheduler failed: %s\n", strerror(errno));
    }
}
static int connect_upstream(const char *path) {
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path);
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}
static int ensure_upstream(Upstream *u) {
    if (u->fd >= 0) return u->fd;
    if (u->down_until_ns != 0) {
        if (lb_mono_ns() < u->down_until_ns) return -1;
        u->down_until_ns = 0;
    }
    for (int attempt = 0; attempt < LB_CONNECT_ATTEMPTS; attempt++) {
        uint64_t t0 = lb_now_ns();
        int fd = connect_upstream(u->path);
        uint64_t t1 = lb_now_ns();
        if (t1 != 0 && t0 != 0) lb_record_ns(LB_STAGE_CONNECT, t1 - t0);
        if (fd >= 0) {
            u->fd = fd;
            return fd;
        }
        if (attempt + 1 < LB_CONNECT_ATTEMPTS) usleep(LB_CONNECT_RETRY_US);
    }
    u->down_until_ns = lb_mono_ns() + LB_BREAKER_COOLDOWN_NS;
    return -1;
}
static bool send_fd_once(int control_fd, int client_fd, uint64_t lb_accept_ns) {
    struct iovec iov = {.iov_base = &lb_accept_ns, .iov_len = sizeof(lb_accept_ns)};
    char cmsgbuf[CMSG_SPACE(sizeof(int))];
    memset(cmsgbuf, 0, sizeof(cmsgbuf));
    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsgbuf;
    msg.msg_controllen = sizeof(cmsgbuf);
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cmsg), &client_fd, sizeof(client_fd));
    uint64_t t0 = lb_now_ns();
    ssize_t ret = sendmsg(control_fd, &msg, MSG_NOSIGNAL);
    uint64_t t1 = lb_now_ns();
    if (t1 != 0 && t0 != 0) lb_record_ns(LB_STAGE_SENDMSG, t1 - t0);
    return ret == (ssize_t)sizeof(lb_accept_ns);
}

/*
 * sendmmsg batch handoff: send multiple fds to an upstream in a single
 * syscall.  Each fd still needs its own msghdr (SCM_RIGHTS carries one
 * fd per cmsg), but sendmmsg batches the kernel entries.
 *
 * Enabled via RINHA_C_LB_SENDMMSG=1.
 */
#define LB_SENDMMSG_BATCH 32

static bool g_lb_sendmmsg = false;

typedef struct {
    int client_fd;
    uint64_t lb_accept_ns;
} PendingHandoff;

static int send_fd_batch(int control_fd, PendingHandoff *items, size_t count) {
    struct mmsghdr mmsg[LB_SENDMMSG_BATCH];
    struct iovec   iovecs[LB_SENDMMSG_BATCH];
    char           cmsgbufs[LB_SENDMMSG_BATCH][CMSG_SPACE(sizeof(int))];

    size_t n = (count < LB_SENDMMSG_BATCH) ? count : LB_SENDMMSG_BATCH;
    for (size_t i = 0; i < n; i++) {
        memset(&mmsg[i], 0, sizeof(mmsg[i]));
        iovecs[i].iov_base = &items[i].lb_accept_ns;
        iovecs[i].iov_len  = sizeof(items[i].lb_accept_ns);
        mmsg[i].msg_hdr.msg_iov    = &iovecs[i];
        mmsg[i].msg_hdr.msg_iovlen = 1;
        mmsg[i].msg_hdr.msg_control    = cmsgbufs[i];
        mmsg[i].msg_hdr.msg_controllen = CMSG_SPACE(sizeof(int));
        memset(cmsgbufs[i], 0, CMSG_SPACE(sizeof(int)));
        struct cmsghdr *cmsg = CMSG_FIRSTHDR(&mmsg[i].msg_hdr);
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type  = SCM_RIGHTS;
        cmsg->cmsg_len   = CMSG_LEN(sizeof(int));
        memcpy(CMSG_DATA(cmsg), &items[i].client_fd, sizeof(int));
    }

    uint64_t t0 = lb_now_ns();
    int sent = sendmmsg(control_fd, mmsg, (unsigned int)n, MSG_NOSIGNAL);
    uint64_t t1 = lb_now_ns();
    if (t1 != 0 && t0 != 0) lb_record_ns(LB_STAGE_SENDMSG, t1 - t0);

    return (sent > 0) ? sent : 0;
}

/*
 * Batch handoff: accumulate pending fds per-upstream, flush with
 * sendmmsg when the batch is full or when we're done accepting.
 */
static size_t handoff_fd_batched(size_t start, PendingHandoff *pending, size_t *pending_len, size_t pending_cap) {
    size_t handed = 0;

    for (size_t i = 0; i < g_upstream_count && *pending_len > 0; i++) {
        Upstream *u = &g_upstreams[(start + i) % g_upstream_count];
        int cfd = ensure_upstream(u);
        if (cfd < 0) continue;

        if (g_lb_sendmmsg && *pending_len > 1) {
            /* Batch send */
            int sent = send_fd_batch(cfd, pending, *pending_len);
            for (int s = 0; s < sent; s++) close(pending[s].client_fd);
            if ((size_t)sent < *pending_len) {
                /* Partial send — upstream might be congested */
                memmove(pending, pending + sent, (*pending_len - (size_t)sent) * sizeof(pending[0]));
                *pending_len -= (size_t)sent;
                close(cfd);
                u->fd = -1;
            } else {
                *pending_len = 0;
            }
            handed += (size_t)sent;
        } else {
            /* Single send (fallback or batch size 1) */
            if (send_fd_once(cfd, pending[0].client_fd, pending[0].lb_accept_ns)) {
                close(pending[0].client_fd);
                memmove(pending, pending + 1, (*pending_len - 1) * sizeof(pending[0]));
                (*pending_len)--;
                handed++;
            } else {
                close(cfd);
                u->fd = -1;
            }
        }
        if (*pending_len == 0) break;
    }
    return handed;
}
static bool handoff_fd(size_t start, int client_fd, uint64_t lb_accept_ns) {
    for (size_t i = 0; i < g_upstream_count; i++) {
        Upstream *u = &g_upstreams[(start + i) % g_upstream_count];
        int cfd = ensure_upstream(u);
        if (cfd < 0) continue;
        if (send_fd_once(cfd, client_fd, lb_accept_ns)) return true;
        close(cfd);
        u->fd = -1;
        cfd = ensure_upstream(u);
        if (cfd >= 0 && send_fd_once(cfd, client_fd, lb_accept_ns)) return true;
        if (cfd >= 0) { close(cfd); u->fd = -1; }
    }
    return false;
}
static void parse_upstreams(const char *csv) {
    char *copy = strdup(csv);
    if (!copy) die("strdup");
    char *save = NULL;
    for (char *tok = strtok_r(copy, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        while (*tok == ' ' || *tok == '\t') tok++;
        if (*tok == '\0') continue;
        if (g_upstream_count == MAX_UPSTREAMS) {
            fprintf(stderr, "too many upstreams, max=%d\n", MAX_UPSTREAMS);
            exit(1);
        }
        Upstream *u = &g_upstreams[g_upstream_count++];
        snprintf(u->path, sizeof(u->path), "%s", tok);
        u->fd = -1;
        u->down_until_ns = 0;
    }
    free(copy);
    if (g_upstream_count == 0) {
        fprintf(stderr, "RINHA_LB_UPSTREAMS is empty\n");
        exit(1);
    }
}
static int create_listener(const char *addr) {
    char host[64] = "0.0.0.0";
    int port = 9999;
    const char *colon = strrchr(addr, ':');
    if (colon) {
        size_t host_len = (size_t)(colon - addr);
        if (host_len > 0 && host_len < sizeof(host)) {
            memcpy(host, addr, host_len);
            host[host_len] = '\0';
        }
        port = atoi(colon + 1);
    }
    if (port <= 0 || port > 65535) {
        fprintf(stderr, "invalid listen addr: %s\n", addr);
        exit(1);
    }
    int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) die("socket tcp");
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (g_defer_accept_secs > 0) {
        setsockopt(fd, IPPROTO_TCP, TCP_DEFER_ACCEPT, &g_defer_accept_secs, sizeof(g_defer_accept_secs));
    }
    struct sockaddr_in sin;
    memset(&sin, 0, sizeof(sin));
    sin.sin_family = AF_INET;
    sin.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &sin.sin_addr) != 1) {
        fprintf(stderr, "invalid listen host: %s\n", host);
        exit(1);
    }
    if (bind(fd, (struct sockaddr *)&sin, sizeof(sin)) != 0) die("bind tcp");
    if (listen(fd, LISTEN_BACKLOG) != 0) die("listen tcp");
    return fd;
}
int main(void) {
    signal(SIGPIPE, SIG_IGN);
    lb_instrument_init_from_env();
    apply_timerslack_from_env();
    apply_sched_from_env();
    const char *addr = getenv("RINHA_LB_ADDR");
    if (!addr) addr = "0.0.0.0:9999";
    const char *upstreams = getenv("RINHA_LB_UPSTREAMS");
    if (!upstreams) upstreams = "/sockets/api1.sock,/sockets/api2.sock";
    g_nonblock_accept = env_truthy(getenv("RINHA_C_LB_NONBLOCK_ACCEPT"));
    g_accept_batch = env_truthy(getenv("RINHA_C_LB_ACCEPT_BATCH"));
    g_lb_sendmmsg = env_truthy(getenv("RINHA_C_LB_SENDMMSG"));
    const char *defer_accept = getenv("RINHA_C_LB_DEFER_ACCEPT_SECS");
    if (defer_accept && defer_accept[0] != '\0') {
        long parsed = strtol(defer_accept, NULL, 10);
        if (parsed >= 0 && parsed <= 60) g_defer_accept_secs = (int)parsed;
    }
    parse_upstreams(upstreams);
    int lfd = create_listener(addr);
    if (g_accept_batch) {
        int fl = fcntl(lfd, F_GETFL, 0);
        if (fl >= 0) (void)fcntl(lfd, F_SETFL, fl | O_NONBLOCK);
    }
    fprintf(stderr,
            "c-lb: listening on %s upstreams=%zu fd_handoff=1 nonblock_accept=%d accept_batch=%d defer_accept_secs=%d sendmmsg=%d\n",
            addr,
            g_upstream_count,
            g_nonblock_accept ? 1 : 0,
            g_accept_batch ? 1 : 0,
            g_defer_accept_secs,
            g_lb_sendmmsg ? 1 : 0);
    for (;;) {
        if (g_accept_batch) {
            struct pollfd pfd = {.fd = lfd, .events = POLLIN, .revents = 0};
            int pr = poll(&pfd, 1, -1);
            if (pr < 0) {
                if (errno == EINTR) continue;
                die("poll");
            }
            unsigned batch = 0;
            PendingHandoff pending[LB_SENDMMSG_BATCH];
            size_t pending_len = 0;
            for (;;) {
                int cfd = accept4(lfd, NULL, NULL, SOCK_CLOEXEC);
                if (cfd < 0) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) break;
                    if (errno == EINTR) continue;
                    die("accept tcp");
                }
                batch++;
                uint64_t bacc_ns = lb_now_ns();
                int bone = 1;
                setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &bone, sizeof(bone));

                if (g_lb_sendmmsg && pending_len < LB_SENDMMSG_BATCH) {
                    /* Accumulate for batch send */
                    pending[pending_len].client_fd = cfd;
                    pending[pending_len].lb_accept_ns = bacc_ns;
                    pending_len++;
                    if (pending_len == LB_SENDMMSG_BATCH) {
                        size_t bstartidx = (g_next++) % g_upstream_count;
                        uint64_t bstart = lb_now_ns();
                        handoff_fd_batched(bstartidx, pending, &pending_len, LB_SENDMMSG_BATCH);
                        uint64_t bdone = lb_now_ns();
                        if (bdone != 0 && bstart != 0) lb_record_ns(LB_STAGE_HANDOFF, bdone - bstart);
                    }
                } else {
                    size_t bstartidx = (g_next++) % g_upstream_count;
                    uint64_t bstart = lb_now_ns();
                    (void)handoff_fd(bstartidx, cfd, bacc_ns);
                    uint64_t bdone = lb_now_ns();
                    if (bdone != 0 && bstart != 0) lb_record_ns(LB_STAGE_HANDOFF, bdone - bstart);
                    close(cfd);
                }
                uint64_t bpost = lb_now_ns();
                if (bpost != 0 && bacc_ns != 0) lb_record_ns(LB_STAGE_POST_ACCEPT, bpost - bacc_ns);
            }
            /* Flush remaining batched fds */
            if (pending_len > 0) {
                size_t bstartidx = (g_next++) % g_upstream_count;
                uint64_t bstart = lb_now_ns();
                handoff_fd_batched(bstartidx, pending, &pending_len, LB_SENDMMSG_BATCH);
                uint64_t bdone = lb_now_ns();
                if (bdone != 0 && bstart != 0) lb_record_ns(LB_STAGE_HANDOFF, bdone - bstart);
                for (size_t pi = 0; pi < pending_len; pi++) close(pending[pi].client_fd);
            }
            lb_record_batch(batch);
            continue;
        }
        int flags = SOCK_CLOEXEC | (g_nonblock_accept ? SOCK_NONBLOCK : 0);
        int client_fd = accept4(lfd, NULL, NULL, flags);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            die("accept tcp");
        }
        uint64_t accepted_ns = lb_now_ns();
        int one = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        size_t startidx = (g_next++) % g_upstream_count;
        uint64_t handoff_start = lb_now_ns();
        (void)handoff_fd(startidx, client_fd, accepted_ns);
        uint64_t handoff_done = lb_now_ns();
        if (handoff_done != 0 && handoff_start != 0) {
            lb_record_ns(LB_STAGE_HANDOFF, handoff_done - handoff_start);
        }
        close(client_fd);
        uint64_t post_done = lb_now_ns();
        if (post_done != 0 && accepted_ns != 0) {
            lb_record_ns(LB_STAGE_POST_ACCEPT, post_done - accepted_ns);
        }
    }
}
