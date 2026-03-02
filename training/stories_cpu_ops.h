// stories_cpu_ops.h — CPU operations: RMSNorm, cross-entropy, Adam, RoPE
#pragma once
#include "stories_config.h"
#include <arm_neon.h>

static float *g_rms_tmp = NULL;
static float *g_rms_ss = NULL;

static void rmsnorm(float *out, const float *x, const float *w, float *rrms_out, int d, int S) {
    if (!g_rms_tmp) g_rms_tmp = (float*)malloc(S*4);
    if (!g_rms_ss) g_rms_ss = (float*)malloc(S*4);
    memset(g_rms_ss, 0, S*4);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vadd(g_rms_tmp, 1, g_rms_ss, 1, g_rms_ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f/d, eps=1e-5f;
    vDSP_vsmsa(g_rms_ss, 1, &invd, &eps, g_rms_ss, 1, (vDSP_Length)S);
    int n = S; vvrsqrtf(g_rms_ss, g_rms_ss, &n);
    if (rrms_out) memcpy(rrms_out, g_rms_ss, S*4);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, g_rms_ss, 1, out+i*S, 1, (vDSP_Length)S);
        vDSP_vsmul(out+i*S, 1, &w[i], out+i*S, 1, (vDSP_Length)S);
    }
}

static void rmsnorm_dw(float *dw, const float *dy, const float *x, const float *rrms, int d, int S) {
    if (!g_rms_tmp) g_rms_tmp = (float*)malloc(S*4);
    for (int i=0; i<d; i++) {
        vDSP_vmul(dy+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vmul(g_rms_tmp, 1, rrms, 1, g_rms_tmp, 1, (vDSP_Length)S);
        float s; vDSP_sve(g_rms_tmp, 1, &s, (vDSP_Length)S);
        dw[i] += s;
    }
}

static void adam_update(float *w, const float *g, AdamState *s, int t, float lr, float b1, float b2, float eps) {
    float bc1 = 1.0f - powf(b1, t), bc2 = 1.0f - powf(b2, t);
    for (size_t i=0; i<s->n; i++) {
        s->m[i] = b1*s->m[i] + (1-b1)*g[i];
        s->v[i] = b2*s->v[i] + (1-b2)*g[i]*g[i];
        float mh = s->m[i]/bc1, vh = s->v[i]/bc2;
        w[i] -= lr * mh / (sqrtf(vh) + eps);
    }
}

// Cross-entropy loss + gradient for logits (column-major: [VOCAB, SEQ])
// logits[v*SEQ+t] = logit for vocab v, position t
// targets[t] = target token id for position t
// Returns mean CE loss, writes dlogits = softmax(logits) - one_hot(targets)
// Transpose to [S,V] for contiguous per-position softmax (128KB/row fits L1)
static float *g_xent_buf = NULL;
static float cross_entropy_loss(float *dlogits, const float *logits, const uint16_t *targets, int V, int S) {
    if (!g_xent_buf) g_xent_buf = (float*)malloc((size_t)S * V * 4);
    float *buf = g_xent_buf;

    // Transpose [V,S] → [S,V]: buf[t*V+v] = logits[v*S+t]
    vDSP_mtrans(logits, 1, buf, 1, (vDSP_Length)S, (vDSP_Length)V);

    float total_loss = 0;
    float invS = 1.0f / S;
    for (int t = 0; t < S; t++) {
        float *row = buf + t * V;
        float maxv;
        vDSP_maxv(row, 1, &maxv, (vDSP_Length)V);
        float neg_max = -maxv;
        vDSP_vsadd(row, 1, &neg_max, row, 1, (vDSP_Length)V);
        int n = V;
        vvexpf(row, row, &n);
        float sum;
        vDSP_sve(row, 1, &sum, (vDSP_Length)V);
        float inv_sum = 1.0f / sum;
        vDSP_vsmul(row, 1, &inv_sum, row, 1, (vDSP_Length)V);
        total_loss -= logf(row[targets[t]] + 1e-10f);
        row[targets[t]] -= 1.0f;
        vDSP_vsmul(row, 1, &invS, row, 1, (vDSP_Length)V);
    }
    // Transpose back [S,V] → [V,S]
    vDSP_mtrans(buf, 1, dlogits, 1, (vDSP_Length)V, (vDSP_Length)S);
    return total_loss / S;
}

// RoPE forward (channel-first [DIM, SEQ] layout)
// In-place: q[h*HD+i, t], index = (h*HD+i)*S + t
// Loop: heads → dim pairs (outer) → positions (inner) for cache-friendly row access
static void cpu_rope_cf(float *q, float *k, int S, int n_heads, int head_dim) {
    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(10000.0f, (float)i / head_dim);
            int row0 = (h * head_dim + i) * S;
            int row1 = (h * head_dim + i + 1) * S;
            for (int t = 0; t < S; t++) {
                float cos_v = cosf(t * freq), sin_v = sinf(t * freq);
                float q0 = q[row0 + t], q1 = q[row1 + t];
                q[row0 + t] = q0 * cos_v - q1 * sin_v;
                q[row1 + t] = q0 * sin_v + q1 * cos_v;
                float k0 = k[row0 + t], k1 = k[row1 + t];
                k[row0 + t] = k0 * cos_v - k1 * sin_v;
                k[row1 + t] = k0 * sin_v + k1 * cos_v;
            }
        }
    }
}

// RoPE backward (channel-first [DIM, SEQ] layout)
// Inverse rotation: transpose of rotation matrix (cos, +sin; -sin, cos)
static void cpu_rope_backward_cf(float *dq, float *dk, int S, int n_heads, int head_dim) {
    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(10000.0f, (float)i / head_dim);
            int row0 = (h * head_dim + i) * S;
            int row1 = (h * head_dim + i + 1) * S;
            for (int t = 0; t < S; t++) {
                float cos_v = cosf(t * freq), sin_v = sinf(t * freq);
                float dq0 = dq[row0 + t], dq1 = dq[row1 + t];
                dq[row0 + t] =  dq0 * cos_v + dq1 * sin_v;
                dq[row1 + t] = -dq0 * sin_v + dq1 * cos_v;
                float dk0 = dk[row0 + t], dk1 = dk[row1 + t];
                dk[row0 + t] =  dk0 * cos_v + dk1 * sin_v;
                dk[row1 + t] = -dk0 * sin_v + dk1 * cos_v;
            }
        }
    }
}

// Precomputed cos/sin table for RoPE: [HD/2][SEQ][2] as fp16
// Table layout: for dim pair j and position t, cos at (j*SEQ+t)*2, sin at (j*SEQ+t)*2+1
// Same frequencies as cpu_rope_cf: freq_j = 1/10000^(2j/HD)
static _Float16 *g_rope_table = NULL;

static void rope_init_table(void) {
    if (g_rope_table) return;
    int half_hd = HD / 2;
    g_rope_table = (_Float16*)malloc(half_hd * SEQ * 2 * sizeof(_Float16));
    for (int j = 0; j < half_hd; j++) {
        float freq = 1.0f / powf(10000.0f, (float)(2 * j) / HD);
        for (int t = 0; t < SEQ; t++) {
            float angle = t * freq;
            g_rope_table[(j * SEQ + t) * 2]     = (_Float16)cosf(angle);
            g_rope_table[(j * SEQ + t) * 2 + 1] = (_Float16)sinf(angle);
        }
    }
}

// NEON fp16 RoPE forward: fp16 src → fp16 dst, no format conversion
// src/dst are raw IOSurface base pointers, offsets in fp16 elements
// Rotation: (cos, -sin; sin, cos) applied to each (row0, row1) pair
static void neon_rope_fwd_f16(const _Float16 *src, _Float16 *dst,
                               int src_q_off, int src_k_off,
                               int dst_q_off, int dst_k_off) {
    const _Float16 *qs = src + src_q_off, *ks = src + src_k_off;
    _Float16 *qd = dst + dst_q_off, *kd = dst + dst_k_off;
    for (int h = 0; h < HEADS; h++) {
        for (int j = 0; j < HD/2; j++) {
            int row0 = (h * HD + 2*j) * SEQ;
            int row1 = (h * HD + 2*j + 1) * SEQ;
            const _Float16 *tbl = g_rope_table + j * SEQ * 2;
            int t = 0;
            for (; t + 7 < SEQ; t += 8) {
                // Load interleaved [cos,sin,cos,sin,...] → deinterleave
                float16x8x2_t cs = vld2q_f16((const __fp16*)(tbl + t*2));
                float16x8_t cv = cs.val[0], sv = cs.val[1];
                // Q rotation
                float16x8_t q0 = vld1q_f16((const __fp16*)(qs + row0 + t));
                float16x8_t q1 = vld1q_f16((const __fp16*)(qs + row1 + t));
                vst1q_f16((__fp16*)(qd + row0 + t), vfmsq_f16(vmulq_f16(q0, cv), q1, sv));
                vst1q_f16((__fp16*)(qd + row1 + t), vfmaq_f16(vmulq_f16(q0, sv), q1, cv));
                // K rotation
                float16x8_t k0 = vld1q_f16((const __fp16*)(ks + row0 + t));
                float16x8_t k1 = vld1q_f16((const __fp16*)(ks + row1 + t));
                vst1q_f16((__fp16*)(kd + row0 + t), vfmsq_f16(vmulq_f16(k0, cv), k1, sv));
                vst1q_f16((__fp16*)(kd + row1 + t), vfmaq_f16(vmulq_f16(k0, sv), k1, cv));
            }
            for (; t < SEQ; t++) {
                _Float16 cv = tbl[t*2], sv = tbl[t*2+1];
                _Float16 q0v = qs[row0+t], q1v = qs[row1+t];
                qd[row0+t] = q0v*cv - q1v*sv; qd[row1+t] = q0v*sv + q1v*cv;
                _Float16 k0v = ks[row0+t], k1v = ks[row1+t];
                kd[row0+t] = k0v*cv - k1v*sv; kd[row1+t] = k0v*sv + k1v*cv;
            }
        }
    }
}

// NEON fp16 RoPE backward: fp16 src → fp16 dst + fp32 output for dW sgemm
// Inverse rotation: (cos, +sin; -sin, cos)
// Writes rotated fp16 to dst (for qkvBwd ANE) and converts to fp32 (for dW)
static void neon_rope_bwd_f16(const _Float16 *src, _Float16 *dst,
                               float *fp32_dq, float *fp32_dk,
                               int src_q_off, int src_k_off,
                               int dst_q_off, int dst_k_off) {
    const _Float16 *dqs = src + src_q_off, *dks = src + src_k_off;
    _Float16 *dqd = dst + dst_q_off, *dkd = dst + dst_k_off;
    for (int h = 0; h < HEADS; h++) {
        for (int j = 0; j < HD/2; j++) {
            int row0 = (h * HD + 2*j) * SEQ;
            int row1 = (h * HD + 2*j + 1) * SEQ;
            const _Float16 *tbl = g_rope_table + j * SEQ * 2;
            int t = 0;
            for (; t + 7 < SEQ; t += 8) {
                float16x8x2_t cs = vld2q_f16((const __fp16*)(tbl + t*2));
                float16x8_t cv = cs.val[0], sv = cs.val[1];
                // Inverse dQ rotation: (dq0*cos + dq1*sin, -dq0*sin + dq1*cos)
                float16x8_t dq0 = vld1q_f16((const __fp16*)(dqs + row0 + t));
                float16x8_t dq1 = vld1q_f16((const __fp16*)(dqs + row1 + t));
                float16x8_t rq0 = vfmaq_f16(vmulq_f16(dq0, cv), dq1, sv);
                float16x8_t rq1 = vfmsq_f16(vmulq_f16(dq1, cv), dq0, sv);
                vst1q_f16((__fp16*)(dqd + row0 + t), rq0);
                vst1q_f16((__fp16*)(dqd + row1 + t), rq1);
                // Convert to fp32 for dW sgemm
                vst1q_f32(fp32_dq + row0 + t,     vcvt_f32_f16(vget_low_f16(rq0)));
                vst1q_f32(fp32_dq + row0 + t + 4,  vcvt_f32_f16(vget_high_f16(rq0)));
                vst1q_f32(fp32_dq + row1 + t,     vcvt_f32_f16(vget_low_f16(rq1)));
                vst1q_f32(fp32_dq + row1 + t + 4,  vcvt_f32_f16(vget_high_f16(rq1)));
                // Inverse dK rotation
                float16x8_t dk0 = vld1q_f16((const __fp16*)(dks + row0 + t));
                float16x8_t dk1 = vld1q_f16((const __fp16*)(dks + row1 + t));
                float16x8_t rk0 = vfmaq_f16(vmulq_f16(dk0, cv), dk1, sv);
                float16x8_t rk1 = vfmsq_f16(vmulq_f16(dk1, cv), dk0, sv);
                vst1q_f16((__fp16*)(dkd + row0 + t), rk0);
                vst1q_f16((__fp16*)(dkd + row1 + t), rk1);
                vst1q_f32(fp32_dk + row0 + t,     vcvt_f32_f16(vget_low_f16(rk0)));
                vst1q_f32(fp32_dk + row0 + t + 4,  vcvt_f32_f16(vget_high_f16(rk0)));
                vst1q_f32(fp32_dk + row1 + t,     vcvt_f32_f16(vget_low_f16(rk1)));
                vst1q_f32(fp32_dk + row1 + t + 4,  vcvt_f32_f16(vget_high_f16(rk1)));
            }
            for (; t < SEQ; t++) {
                _Float16 cv = tbl[t*2], sv = tbl[t*2+1];
                _Float16 dq0v = dqs[row0+t], dq1v = dqs[row1+t];
                _Float16 rq0v = dq0v*cv + dq1v*sv, rq1v = -dq0v*sv + dq1v*cv;
                dqd[row0+t] = rq0v; dqd[row1+t] = rq1v;
                fp32_dq[row0+t] = (float)rq0v; fp32_dq[row1+t] = (float)rq1v;
                _Float16 dk0v = dks[row0+t], dk1v = dks[row1+t];
                _Float16 rk0v = dk0v*cv + dk1v*sv, rk1v = -dk0v*sv + dk1v*cv;
                dkd[row0+t] = rk0v; dkd[row1+t] = rk1v;
                fp32_dk[row0+t] = (float)rk0v; fp32_dk[row1+t] = (float)rk1v;
            }
        }
    }
}

// Embedding lookup: token_ids → x [DIM, SEQ] (channel-first)
// embed is [VOCAB, DIM] row-major (vocab_size rows, dim cols)
static void embed_lookup(float *x, const float *embed, const uint16_t *tokens, int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        for (int d = 0; d < dim; d++) {
            x[d*seq + t] = embed[tok*dim + d];
        }
    }
}

// Embedding backward: accumulate dE[tok] += dx[:,t] for each position
static void embed_backward(float *d_embed, const float *dx, const uint16_t *tokens, int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        for (int d = 0; d < dim; d++) {
            d_embed[tok*dim + d] += dx[d*seq + t];
        }
    }
}
