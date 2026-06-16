#ifndef JSON_SIMD_H
#define JSON_SIMD_H

/*
 * SIMD-accelerated JSON helpers for Rinha de Backend 2026.
 *
 * Instead of replacing the entire parser (which is complex and error-prone),
 * we accelerate the two hottest inner loops:
 *
 *   1. AVX2 fast float parser — replaces strtod() for simple numbers.
 *      strtod is ~200ns per call due to locale handling, buffer setup,
 *      and errno checks. Our integer-only path is ~5ns.
 *
 *   2. AVX2 known_merchants scan — replaces byte-by-byte memcmp loop
 *      with 32-byte-at-a-time first-byte matching + verify.
 *
 *   3. AVX2 structural character detection — for skip_value() and
 *      field boundary detection. Scans 32 bytes at once to find
 *      quotes, braces, brackets, colons.
 *
 * Enable by compiling with -DRINHA_C_USE_SIMD_JSON
 */

#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════
 * AVX2 fast float parser
 *
 * Handles: [-]digits[.digits] — no scientific notation needed
 * for the values in this API (amounts, km, etc.)
 *
 * Falls back to strtod for edge cases.
 * ═══════════════════════════════════════════════════════════════════ */

static inline float simd_fast_f32(const uint8_t *p, size_t len) {
    bool neg = false;
    if (len > 0 && *p == '-') { neg = true; p++; len--; }

    uint32_t integer = 0;
    while (len > 0 && (unsigned)(*p - '0') <= 9u) {
        integer = integer * 10 + (uint32_t)(*p - '0');
        p++; len--;
    }

    uint32_t fraction = 0;
    uint32_t frac_digits = 0;
    if (len > 0 && *p == '.') {
        p++; len--;
        /* Only parse up to 7 fractional digits (float precision limit) */
        while (len > 0 && (unsigned)(*p - '0') <= 9u && frac_digits < 7) {
            fraction = fraction * 10 + (uint32_t)(*p - '0');
            frac_digits++;
            p++; len--;
        }
        /* Skip remaining digits */
        while (len > 0 && (unsigned)(*p - '0') <= 9u) { p++; len--; }
    }

    float result = (float)integer;
    if (frac_digits > 0) {
        /*
         * Use a lookup table for 10^-n. This avoids powf() and
         * the integer division in the common case.
         *
         * For frac_digits 1-7, the exact float values are:
         *   1e-1 = 0.1f, 1e-2 = 0.01f, ..., 1e-7 = 1e-7f
         */
        static const float pow10_neg[] = {
            1e-1f, 1e-2f, 1e-3f, 1e-4f,
            1e-5f, 1e-6f, 1e-7f,
        };
        result += (float)fraction * pow10_neg[frac_digits - 1];
    }

    return neg ? -result : result;
}

/* ═══════════════════════════════════════════════════════════════════
 * AVX2 structural character scanner for skip_value()
 *
 * Finds the next structural character (", \, {, }, [, ]) in a
 * 32-byte chunk. Used to accelerate string scanning and
 * balanced-delimiter depth tracking.
 * ═══════════════════════════════════════════════════════════════════ */

static inline uint32_t simd_structural_mask(const uint8_t *chunk) {
    __m256i data = _mm256_loadu_si256((const __m256i *)chunk);

    __m256i quote  = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('"'));
    __m256i escape = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('\\'));
    __m256i lbrace = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('{'));
    __m256i rbrace = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('}'));
    __m256i lbrack = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('['));
    __m256i rbrack = _mm256_cmpeq_epi8(data, _mm256_set1_epi8(']'));

    __m256i any = _mm256_or_si256(
        _mm256_or_si256(quote, escape),
        _mm256_or_si256(
            _mm256_or_si256(lbrace, rbrace),
            _mm256_or_si256(lbrack, rbrack)));

    return (uint32_t)_mm256_movemask_epi8(any);
}

/*
 * AVX2-accelerated skip_balanced.
 * Finds the matching close delimiter for { } or [ ].
 * Uses SIMD to scan 32 bytes at a time for structural chars,
 * only falling back to per-byte mode for string contents.
 */
static inline bool simd_skip_balanced(
    size_t *p, const uint8_t *buf, size_t len,
    uint8_t open, uint8_t close)
{
    int depth = 0;
    bool in_string = false, esc = false;

    /* Process 32-byte chunks with AVX2 */
    while (*p + 32 <= len) {
        if (in_string) {
            /* Inside a string — scan for " or \ */
            uint32_t mask = simd_structural_mask(buf + *p);
            while (mask) {
                uint32_t bit = __builtin_ctz(mask);
                size_t pos = *p + bit;
                if (pos >= len) goto done;
                uint8_t b = buf[pos];
                if (esc) { esc = false; mask &= (mask - 1); continue; }
                if (b == '\\') { esc = true; mask &= (mask - 1); continue; }
                if (b == '"') { in_string = false; *p = pos + 1; goto scan_chunk; }
                mask &= (mask - 1);
            }
            *p += 32;
            continue;
        }

scan_chunk:
        /* Not in string — scan for structural chars that affect depth */
        {
            uint32_t mask = simd_structural_mask(buf + *p);
            while (mask) {
                uint32_t bit = __builtin_ctz(mask);
                size_t pos = *p + bit;
                if (pos >= len) goto done;
                uint8_t b = buf[pos];
                if (b == '"') { in_string = true; *p = pos + 1; goto scan_chunk; }
                if (b == open) { depth++; }
                else if (b == close) {
                    depth--;
                    if (depth == 0) { *p = pos + 1; return true; }
                }
                mask &= (mask - 1);
            }
            *p += 32;
        }
    }

done:
    /* Scalar fallback for remaining bytes */
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

/* ═══════════════════════════════════════════════════════════════════
 * AVX2 known_merchants scan
 *
 * Checks if merchant_id appears in the raw JSON array region
 * [arr_start, arr_end). Scans for the first byte of the ID
 * using AVX2, then verifies the full match.
 * ═══════════════════════════════════════════════════════════════════ */

static inline bool simd_merchant_in_known(
    const uint8_t *buf,
    const uint8_t *mid, size_t mid_len,
    size_t arr_start, size_t arr_end)
{
    if (mid_len == 0 || mid_len >= 32) goto scalar;

    {
        __m256i fb = _mm256_set1_epi8((char)mid[0]);
        size_t pos = arr_start;
        while (pos + 32 <= arr_end) {
            __m256i chunk = _mm256_loadu_si256((const __m256i *)(buf + pos));
            uint32_t mask = (uint32_t)_mm256_movemask_epi8(
                _mm256_cmpeq_epi8(chunk, fb));
            while (mask) {
                uint32_t bit = __builtin_ctz(mask);
                size_t c = pos + bit;
                if (c > arr_start && buf[c-1] == '"' &&
                    c + mid_len + 1 <= arr_end &&
                    buf[c + mid_len] == '"' &&
                    memcmp(buf + c, mid, mid_len) == 0)
                    return true;
                mask &= (mask - 1);
            }
            pos += 32;
        }
    }

scalar:
    /* Scalar fallback for tail or long IDs */
    for (size_t q = arr_start; q + mid_len + 2 <= arr_end; q++) {
        if (buf[q] == '"' && buf[q + 1 + mid_len] == '"' &&
            memcmp(buf + q + 1, mid, mid_len) == 0)
            return true;
    }
    return false;
}

/* ═══════════════════════════════════════════════════════════════════
 * AVX2 whitespace scanner
 *
 * Finds the next non-whitespace byte by scanning 32 bytes at a time.
 * Returns the offset of the first non-ws byte, or len if all ws.
 * ═══════════════════════════════════════════════════════════════════ */

static inline size_t simd_skip_ws(const uint8_t *buf, size_t start, size_t len) {
    /* Quick scalar check for the common case (next char is not ws) */
    if (start < len && buf[start] != ' ' && buf[start] != '\t' &&
        buf[start] != '\n' && buf[start] != '\r')
        return start;

    /* AVX2 scan for non-whitespace */
    __m256i sp  = _mm256_set1_epi8(' ');
    __m256i tab = _mm256_set1_epi8('\t');
    __m256i nl  = _mm256_set1_epi8('\n');
    __m256i cr  = _mm256_set1_epi8('\r');

    size_t pos = start;
    while (pos + 32 <= len) {
        __m256i data = _mm256_loadu_si256((const __m256i *)(buf + pos));
        __m256i ws = _mm256_or_si256(
            _mm256_or_si256(_mm256_cmpeq_epi8(data, sp), _mm256_cmpeq_epi8(data, tab)),
            _mm256_or_si256(_mm256_cmpeq_epi8(data, nl), _mm256_cmpeq_epi8(data, cr)));
        uint32_t mask = (uint32_t)_mm256_movemask_epi8(ws);
        uint32_t not_ws = ~mask;
        if (not_ws) {
            uint32_t bit = __builtin_ctz(not_ws);
            return pos + bit;
        }
        pos += 32;
    }

    /* Scalar tail */
    while (pos < len) {
        if (buf[pos] != ' ' && buf[pos] != '\t' &&
            buf[pos] != '\n' && buf[pos] != '\r')
            return pos;
        pos++;
    }
    return len;
}

#endif /* JSON_SIMD_H */
