// kv_cache.h — KV cache for autoregressive inference
// Channel-first fp16 layout [kv_dim, max_seq] matching IOSurface format
#pragma once
#include "infer_config.h"
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>

typedef struct {
    _Float16 *k;    // [kv_dim, max_seq] channel-first
    _Float16 *v;    // [kv_dim, max_seq] channel-first
} KVLayer;

typedef struct {
    KVLayer *layers;
    int n_layers;
    int kv_dim;
    int max_seq;
    int len;        // current number of cached positions
} KVCache;

static KVCache *kv_cache_alloc(const InferConfig *cfg) {
    KVCache *kv = (KVCache *)calloc(1, sizeof(KVCache));
    kv->n_layers = cfg->n_layers;
    kv->kv_dim = cfg->kv_dim;
    kv->max_seq = cfg->max_seq;
    kv->len = 0;
    kv->layers = (KVLayer *)calloc(cfg->n_layers, sizeof(KVLayer));
    size_t layer_bytes = (size_t)cfg->kv_dim * cfg->max_seq * sizeof(_Float16);
    for (int l = 0; l < cfg->n_layers; l++) {
        kv->layers[l].k = (_Float16 *)calloc(1, layer_bytes);
        kv->layers[l].v = (_Float16 *)calloc(1, layer_bytes);
    }
    return kv;
}

// Append n_new token K/V vectors to layer cache
// k_new, v_new: [kv_dim, n_new] channel-first fp16
static void kv_cache_append(KVCache *kv, int layer, const _Float16 *k_new, const _Float16 *v_new, int n_new) {
    int pos = kv->len;
    int S = kv->max_seq;
    int D = kv->kv_dim;
    _Float16 *kc = kv->layers[layer].k;
    _Float16 *vc = kv->layers[layer].v;
    // Channel-first: element [c, t] is at c * max_seq + t
    // k_new[c, i] is at c * n_new + i
    for (int c = 0; c < D; c++) {
        memcpy(kc + c * S + pos, k_new + c * n_new, n_new * sizeof(_Float16));
        memcpy(vc + c * S + pos, v_new + c * n_new, n_new * sizeof(_Float16));
    }
}

// Append a single token K/V (scatter pattern, NEON vectorized)
static void kv_cache_append1(KVCache *kv, int layer, const _Float16 *k_new, const _Float16 *v_new) {
    int pos = kv->len;
    int S = kv->max_seq;
    int D = kv->kv_dim;
    _Float16 *kc = kv->layers[layer].k;
    _Float16 *vc = kv->layers[layer].v;
    // k_new[c] goes to kc[c * S + pos]
    for (int c = 0; c < D; c++) {
        kc[c * S + pos] = k_new[c];
        vc[c * S + pos] = v_new[c];
    }
}

static void kv_cache_advance(KVCache *kv, int n) {
    kv->len += n;
}

static void kv_cache_reset(KVCache *kv) {
    kv->len = 0;
}

static void kv_cache_free(KVCache *kv) {
    if (!kv) return;
    for (int l = 0; l < kv->n_layers; l++) {
        free(kv->layers[l].k);
        free(kv->layers[l].v);
    }
    free(kv->layers);
    free(kv);
}
