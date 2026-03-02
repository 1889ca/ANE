// stories_cpu_ops.h — CPU operations: RMSNorm, cross-entropy, Adam, softmax
#pragma once
#include "stories_config.h"

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
