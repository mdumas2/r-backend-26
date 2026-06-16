/*
 * api_uring.c — io_uring-based API server for Rinha de Backend 2026.
 *
 * Replaces the epoll event loop with io_uring for:
 *   - Batched submission/completion (fewer syscalls)
 *   - No per-recv/per-send syscall overhead
 *   - Kernel-side polling with SQPOLL if available
 *
 * Links against engine.o (compiled from main.c -DRINHA_C_NO_MAIN)
 * for the KD-tree search engine and JSON parsing.
 */

#define _GNU_SOURCE
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <liburing.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <poll.h>

/* ═══════════════════════════════════════════════════════════════════
 * External declarations from engine.o (main.c -DRINHA_C_NO_MAIN)
 * ═══════════════════════════════════════════════════════════════════ */

extern void init_response_lengths(void);
extern void init_detector_from_env(void);
extern uint8_t decide_frauds(const uint8_t *body, size_t body_len);
extern void warmup_engine(uint32_t n_queries);

/* ═══════════════════════════════════════════════════════════════════
 * io_uring constants
 * ═══════════════════════════════════════════════════════════════════ */

#define URING_ENTRIES      4096
#define URING_CQ_SIZE     8192
#define URING_MAX_FDS     65536
#define URING_BUF_CAP     131072

typedef enum {
    OP_ACCEPT = 1,
    OP_RECV,
    OP_SEND,
    OP_CTRL_POLL,  /* poll on control fd for incoming SCM_RIGHTS */
} OpType;

typedef struct {
    int       fd;
    uint8_t  *buf;
    size_t    buf_len;
    size_t    header_end;
    size_t    body_start;
    size_t    body_len;
    const char *send_ptr;
    size_t    send_len;
    size_t    send_off;
    uint64_t  lb_accept_ns;
    bool      in_use;
} UConn;

/* ═══════════════════════════════════════════════════════════════════
 * Globals
 * ═══════════════════════════════════════════════════════════════════ */

static struct io_uring  g_ring;
static UConn           *g_conns[URING_MAX_FDS];
static int              g_listen_fd = -1;
static int              g_ctrl_fd = -1;

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
static size_t RESP_DECISION_LEN[6];

/* ═══════════════════════════════════════════════════════════════════
 * Connection management
 * ═══════════════════════════════════════════════════════════════════ */

static UConn *conn_get(int fd) {
    if (fd < 0 || fd >= URING_MAX_FDS) return NULL;
    UConn *c = g_conns[fd];
    if (!c) {
        c = (UConn *)calloc(1, sizeof(UConn));
        if (!c) return NULL;
        c->buf = (uint8_t *)malloc(URING_BUF_CAP);
        if (!c->buf) { free(c); return NULL; }
        g_conns[fd] = c;
    }
    return c;
}

static void conn_reset(UConn *c) {
    c->buf_len = 0;
    c->header_end = 0;
    c->body_start = 0;
    c->body_len = 0;
    c->send_ptr = NULL;
    c->send_len = 0;
    c->send_off = 0;
    c->in_use = true;
}

static void conn_drop(int fd) {
    if (fd < 0 || fd >= URING_MAX_FDS) return;
    UConn *c = g_conns[fd];
    if (c) c->in_use = false;
    close(fd);
}

/* ═══════════════════════════════════════════════════════════════════
 * HTTP parsing (same as main.c)
 * ═══════════════════════════════════════════════════════════════════ */

static ssize_t find_header_end(const uint8_t *buf, size_t len) {
    const uint8_t needle[] = "\r\n\r\n";
    const uint8_t *p = (const uint8_t *)memmem(buf, len, needle, sizeof(needle) - 1);
    return p ? (ssize_t)((p - buf) + sizeof(needle) - 1) : -1;
}

static int header_content_length(const uint8_t *buf, size_t header_len) {
    const uint8_t *p = buf;
    const uint8_t *end = buf + header_len;
    while (p < end) {
        const uint8_t *nl = (const uint8_t *)memchr(p, '\n', (size_t)(end - p));
        const uint8_t *line_end = nl ? nl : end;
        const uint8_t *line = p;
        if (line < line_end && line_end[-1] == '\r') line_end--;
        size_t line_len = (size_t)(line_end - line);
        if (line_len >= 15) {
            int match = 1;
            const char *cl = "content-length:";
            for (int i = 0; i < 15; i++)
                if (tolower((int)line[i]) != (int)cl[i]) { match = 0; break; }
            if (match) {
                line += 15;
                while (line < line_end && (*line == ' ' || *line == '\t')) line++;
                int v = 0;
                while (line < line_end && isdigit(*line))
                    v = v * 10 + (*line++ - '0');
                return v;
            }
        }
        if (!nl) break;
        p = nl + 1;
    }
    return -1;
}

/* ═══════════════════════════════════════════════════════════════════
 * SCM_RIGHTS receiver
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct { int fd; uint64_t lb_accept_ns; } FdRecv;

static FdRecv recv_fd_from_lb(int ctrl_fd) {
    FdRecv result = {.fd = -1, .lb_accept_ns = 0};
    uint64_t lb_ns = 0;
    struct iovec iov = {.iov_base = &lb_ns, .iov_len = sizeof(lb_ns)};
    char cmsgbuf[CMSG_SPACE(sizeof(int))];
    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsgbuf;
    msg.msg_controllen = sizeof(cmsgbuf);
    ssize_t n = recvmsg(ctrl_fd, &msg, MSG_DONTWAIT);
    if (n <= 0) return result;
    result.lb_accept_ns = (n == (ssize_t)sizeof(lb_ns)) ? lb_ns : 0;
    for (struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg); cmsg;
         cmsg = CMSG_NXTHDR(&msg, cmsg)) {
        if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
            memcpy(&result.fd, CMSG_DATA(cmsg), sizeof(int));
            return result;
        }
    }
    return result;
}

/* ═══════════════════════════════════════════════════════════════════
 * SQE submission
 * ═══════════════════════════════════════════════════════════════════ */

static inline uint64_t mk_ud(int fd, OpType op) {
    return (uint64_t)(uint32_t)fd | ((uint64_t)op << 32);
}

static void submit_recv(UConn *c) {
    struct io_uring_sqe *sqe = io_uring_get_sqe(&g_ring);
    if (!sqe) return;
    io_uring_prep_recv(sqe, c->fd, c->buf + c->buf_len,
                       URING_BUF_CAP - c->buf_len, 0);
    sqe->user_data = mk_ud(c->fd, OP_RECV);
}

static void submit_send(UConn *c) {
    struct io_uring_sqe *sqe = io_uring_get_sqe(&g_ring);
    if (!sqe) return;
    io_uring_prep_send(sqe, c->fd, c->send_ptr + c->send_off,
                       c->send_len - c->send_off, MSG_NOSIGNAL);
    sqe->user_data = mk_ud(c->fd, OP_SEND);
}

static void submit_accept(void) {
    struct io_uring_sqe *sqe = io_uring_get_sqe(&g_ring);
    if (!sqe) return;
    io_uring_prep_accept(sqe, g_listen_fd, NULL, NULL, SOCK_CLOEXEC);
    sqe->user_data = mk_ud(g_listen_fd, OP_ACCEPT);
}

/*
 * Submit a poll SQE on the control fd. When the LB sends an SCM_RIGHTS
 * message, the fd becomes readable and io_uring wakes us up.
 */
static void submit_ctrl_poll(void) {
    if (g_ctrl_fd < 0) return;
    struct io_uring_sqe *sqe = io_uring_get_sqe(&g_ring);
    if (!sqe) return;
    io_uring_prep_poll_add(sqe, g_ctrl_fd, POLLIN);
    sqe->user_data = mk_ud(g_ctrl_fd, OP_CTRL_POLL);
}

/* ═══════════════════════════════════════════════════════════════════
 * Process a complete HTTP request
 * ═══════════════════════════════════════════════════════════════════ */

static void process_conn(UConn *c) {
    ssize_t h = find_header_end(c->buf, c->buf_len);
    if (h < 0) {
        if (c->buf_len >= URING_BUF_CAP) { conn_drop(c->fd); return; }
        submit_recv(c);
        return;
    }
    size_t header_len = (size_t)h;

    if (header_len >= 14 && memcmp(c->buf, "GET /ready ", 11) == 0) {
        c->send_ptr = RESP_READY;
        c->send_len = sizeof(RESP_READY) - 1;
        c->send_off = 0;
        c->body_start = header_len;
        c->body_len = 0;
        submit_send(c);
        return;
    }

    if (!(header_len >= 24 && memcmp(c->buf, "POST /fraud-score ", 18) == 0)) {
        c->send_ptr = RESP_BAD;
        c->send_len = sizeof(RESP_BAD) - 1;
        c->send_off = 0;
        submit_send(c);
        return;
    }

    int cl = header_content_length(c->buf, header_len);
    if (cl < 0 || cl > 100000) {
        c->send_ptr = RESP_BAD;
        c->send_len = sizeof(RESP_BAD) - 1;
        c->send_off = 0;
        submit_send(c);
        return;
    }

    if (c->buf_len < header_len + (size_t)cl) {
        if (c->buf_len >= URING_BUF_CAP) { conn_drop(c->fd); return; }
        c->body_start = header_len;
        c->body_len = (size_t)cl;
        submit_recv(c);
        return;
    }

    /* ── Fraud detection (synchronous CPU work) ── */
    uint8_t frauds = decide_frauds(c->buf + header_len, (size_t)cl);
    if (frauds > 5) frauds = 5;

    c->send_ptr = RESP_DECISION[frauds];
    c->send_len = RESP_DECISION_LEN[frauds];
    c->send_off = 0;
    c->body_start = header_len;
    c->body_len = (size_t)cl;
    submit_send(c);
}

/* ═══════════════════════════════════════════════════════════════════
 * Drain fds from the LB control connection
 * ═══════════════════════════════════════════════════════════════════ */

static void drain_ctrl_fds(void) {
    if (g_ctrl_fd < 0) return;
    for (;;) {
        FdRecv fr = recv_fd_from_lb(g_ctrl_fd);
        if (fr.fd < 0) break;
        int fd = fr.fd;
        if (fd >= URING_MAX_FDS) { close(fd); continue; }
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        UConn *c = conn_get(fd);
        if (!c) { close(fd); continue; }
        c->fd = fd;
        c->lb_accept_ns = fr.lb_accept_ns;
        conn_reset(c);
        submit_recv(c);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Event loop
 * ═══════════════════════════════════════════════════════════════════ */

static void uring_loop(void) {
    struct io_uring_cqe *cqe;

    while (true) {
        int ret = io_uring_submit_and_wait(&g_ring, 1);
        if (ret < 0 && ret != -EINTR) {
            fprintf(stderr, "c-api-uring: submit_and_wait: %s\n", strerror(-ret));
            break;
        }

        unsigned head, count = 0;
        io_uring_for_each_cqe(&g_ring, head, cqe) {
            count++;
            uint64_t ud = cqe->user_data;
            int fd = (int)(ud & 0xFFFFFFFF);
            OpType op = (OpType)(ud >> 32);
            int res = cqe->res;

            switch (op) {
            case OP_ACCEPT:
                if (res >= 0) {
                    g_ctrl_fd = res;
                    fprintf(stderr, "c-api-uring: control fd=%d\n", res);
                    /* Re-arm accept for next LB reconnection */
                    submit_accept();
                    /* Start polling the control fd for SCM_RIGHTS */
                    submit_ctrl_poll();
                    /* Drain any fds the LB already queued */
                    drain_ctrl_fds();
                } else {
                    submit_accept();
                }
                break;

            case OP_CTRL_POLL:
                /* Control fd is readable — drain all pending fds */
                drain_ctrl_fds();
                /* Re-arm the poll */
                submit_ctrl_poll();
                break;

            case OP_RECV: {
                UConn *c = (fd >= 0 && fd < URING_MAX_FDS) ? g_conns[fd] : NULL;
                if (!c || !c->in_use) { if (fd >= 0) close(fd); break; }
                if (res <= 0) { conn_drop(fd); break; }
                c->buf_len += (size_t)res;
                process_conn(c);
                break;
            }

            case OP_SEND: {
                UConn *c = (fd >= 0 && fd < URING_MAX_FDS) ? g_conns[fd] : NULL;
                if (!c || !c->in_use) { if (fd >= 0) close(fd); break; }
                if (res < 0) { conn_drop(fd); break; }
                c->send_off += (size_t)res;
                if (c->send_off >= c->send_len) {
                    size_t consumed = c->body_start + c->body_len;
                    if (consumed > 0 && consumed < c->buf_len) {
                        memmove(c->buf, c->buf + consumed, c->buf_len - consumed);
                        c->buf_len -= consumed;
                        conn_reset(c);
                        process_conn(c);
                    } else {
                        conn_drop(fd);
                    }
                } else {
                    submit_send(c);
                }
                break;
            }

            default:
                break;
            }
        }
        if (count > 0) io_uring_cq_advance(&g_ring, count);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Entry point
 * ═══════════════════════════════════════════════════════════════════ */

int main(void) {
    signal(SIGPIPE, SIG_IGN);

    for (size_t i = 0; i < 6; i++)
        RESP_DECISION_LEN[i] = strlen(RESP_DECISION[i]);

    init_response_lengths();
    init_detector_from_env();
    warmup_engine(50000);

    struct io_uring_params params = {0};
    params.cq_entries = URING_CQ_SIZE;
    params.flags = IORING_SETUP_CQSIZE | IORING_SETUP_COOP_TASKRUN;

    int ret = io_uring_queue_init_params(URING_ENTRIES, &g_ring, &params);
    if (ret < 0) {
        fprintf(stderr, "c-api-uring: io_uring init failed (%s)\n", strerror(-ret));
        return 1;
    }
    fprintf(stderr, "c-api-uring: ring ready entries=%u cq=%u\n",
            URING_ENTRIES, params.cq_entries);

    const char *sock = getenv("RINHA_SOCKET");
    if (!sock) sock = "/sockets/api.sock";
    g_listen_fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (g_listen_fd < 0) { perror("socket unix"); return 1; }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", sock);
    unlink(sock);
    if (bind(g_listen_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        perror("bind unix"); return 1;
    }
    if (listen(g_listen_fd, 1024) != 0) {
        perror("listen unix"); return 1;
    }
    fprintf(stderr, "c-api-uring: listening on %s\n", sock);

    submit_accept();
    io_uring_submit(&g_ring);

    uring_loop();

    io_uring_queue_exit(&g_ring);
    return 0;
}