// infer.m — Bonsai-8B inference on Apple Neural Engine
// Loads GGUF model + optional LoRA adapter, compiles ANE kernels, runs autoregressive generation
//
// Usage: ./infer --model <gguf> [--lora <gguf>] [--prompt "text"] [--max-tokens N] [--temp F]
//
// Architecture: Qwen3 (4096 dim, 11008 hidden, 32 heads, 4 KV heads GQA, 28 layers)
// Quantization: Q1_0_g128 base weights, F16 LoRA adapters
//
// Strategy:
//   - QKV projection: ANE kernel (RMSNorm + Wq/Wk/Wv conv)
//   - Attention decode: CPU (single-query, BLAS-friendly)
//   - FFN: ANE kernel (RMSNorm + W1/W3/SiLU/W2 + residual)
//   - Classifier: CPU matmul (vocab too large for ANE conv)
//   - RoPE: CPU NEON fp16

#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <getopt.h>
#include <arm_neon.h>

// infer_mil.h pulls in stories_io.h → stories_config.h (includes sys/stat.h)
// Must come before bonsai_lora.h which also uses stat/fstat
#include "infer_config.h"
#include "kv_cache.h"
#include "infer_mil.h"
#include "bonsai_lora.h"
#include "infer_tokenizer.h"
#include "infer_xor_patch.h"

// ========== QK Norm (Qwen3) ==========
// Per-head RMSNorm on Q[dim] and K[kv_dim] with learned weights[head_dim]
// Applied after projection, before RoPE
static void qk_norm_f16(const InferConfig *c, _Float16 *q, _Float16 *k,
                         const _Float16 *q_norm_w, const _Float16 *k_norm_w) {
    int hd = c->head_dim;
    // Q: n_heads groups of head_dim
    for (int h = 0; h < c->n_heads; h++) {
        _Float16 *qh = q + h * hd;
        float ss = 0;
        for (int i = 0; i < hd; i++) ss += (float)qh[i] * (float)qh[i];
        float rrms = 1.0f / sqrtf(ss / hd + 1e-6f);
        for (int i = 0; i < hd; i++)
            qh[i] = (_Float16)((float)qh[i] * rrms * (float)q_norm_w[i]);
    }
    // K: n_kv_heads groups of head_dim
    for (int h = 0; h < c->n_kv_heads; h++) {
        _Float16 *kh = k + h * hd;
        float ss = 0;
        for (int i = 0; i < hd; i++) ss += (float)kh[i] * (float)kh[i];
        float rrms = 1.0f / sqrtf(ss / hd + 1e-6f);
        for (int i = 0; i < hd; i++)
            kh[i] = (_Float16)((float)kh[i] * rrms * (float)k_norm_w[i]);
    }
}

// ========== RoPE for inference ==========
// Single-position RoPE on fp16 Q[dim] and K[kv_dim] at position pos
static void rope_single_pos(const InferConfig *c, _Float16 *q, _Float16 *k, int pos) {
    float theta = c->rope_theta;
    // Q: all n_heads
    for (int h = 0; h < c->n_heads; h++) {
        for (int i = 0; i < c->head_dim; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / c->head_dim);
            float angle = pos * freq;
            float cos_v = cosf(angle), sin_v = sinf(angle);
            int idx0 = h * c->head_dim + i;
            int idx1 = idx0 + 1;
            float q0 = (float)q[idx0], q1 = (float)q[idx1];
            q[idx0] = (_Float16)(q0 * cos_v - q1 * sin_v);
            q[idx1] = (_Float16)(q0 * sin_v + q1 * cos_v);
        }
    }
    // K: only n_kv_heads
    for (int h = 0; h < c->n_kv_heads; h++) {
        for (int i = 0; i < c->head_dim; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / c->head_dim);
            float angle = pos * freq;
            float cos_v = cosf(angle), sin_v = sinf(angle);
            int idx0 = h * c->head_dim + i;
            int idx1 = idx0 + 1;
            float k0 = (float)k[idx0], k1 = (float)k[idx1];
            k[idx0] = (_Float16)(k0 * cos_v - k1 * sin_v);
            k[idx1] = (_Float16)(k0 * sin_v + k1 * cos_v);
        }
    }
}

// Multi-position RoPE on channel-first fp16 layout [C, S]
static void rope_prefill(const InferConfig *c, _Float16 *q, _Float16 *k, int S) {
    float theta = c->rope_theta;
    for (int h = 0; h < c->n_heads; h++) {
        for (int i = 0; i < c->head_dim; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / c->head_dim);
            int row0 = (h * c->head_dim + i) * S;
            int row1 = (h * c->head_dim + i + 1) * S;
            for (int t = 0; t < S; t++) {
                float cos_v = cosf(t * freq), sin_v = sinf(t * freq);
                float q0 = (float)q[row0 + t], q1 = (float)q[row1 + t];
                q[row0 + t] = (_Float16)(q0 * cos_v - q1 * sin_v);
                q[row1 + t] = (_Float16)(q0 * sin_v + q1 * cos_v);
            }
        }
    }
    for (int h = 0; h < c->n_kv_heads; h++) {
        for (int i = 0; i < c->head_dim; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / c->head_dim);
            int row0 = (h * c->head_dim + i) * S;
            int row1 = (h * c->head_dim + i + 1) * S;
            for (int t = 0; t < S; t++) {
                float cos_v = cosf(t * freq), sin_v = sinf(t * freq);
                float k0 = (float)k[row0 + t], k1 = (float)k[row1 + t];
                k[row0 + t] = (_Float16)(k0 * cos_v - k1 * sin_v);
                k[row1 + t] = (_Float16)(k0 * sin_v + k1 * cos_v);
            }
        }
    }
}

// ========== CPU RMSNorm ==========
// fp16 in/out version (for final norm / small models)
static void rmsnorm_f16(const InferConfig *c, _Float16 *out, const _Float16 *x, const _Float16 *w) {
    float ss = 0;
    for (int i = 0; i < c->dim; i++) ss += (float)x[i] * (float)x[i];
    ss = 1.0f / sqrtf(ss / c->dim + 1e-5f);
    for (int i = 0; i < c->dim; i++)
        out[i] = (_Float16)((float)x[i] * ss * (float)w[i]);
}

// fp32 input → fp16 output (for ANE kernel input prep)
static void rmsnorm_f32_to_f16(int dim, _Float16 *out, const float *x, const _Float16 *w) {
    float ss = 0;
    for (int i = 0; i < dim; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / dim + 1e-6f);
    for (int i = 0; i < dim; i++)
        out[i] = (_Float16)(x[i] * ss * (float)w[i]);
}

// ========== CPU classifier ==========
// embed: [vocab, dim] fp16, x: [dim] fp16 → logits: [vocab] fp32
// Parallel NEON: dispatch_apply across vocab rows
static void classifier_f16(const InferConfig *c, float *logits, const _Float16 *embed, const _Float16 *x) {
    int V = c->vocab_size, D = c->dim;
    int chunk = 256;  // process 256 vocab rows per dispatch unit
    int n_chunks = (V + chunk - 1) / chunk;
    dispatch_apply(n_chunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t ci) {
      int v_start = (int)ci * chunk;
      int v_end = v_start + chunk; if (v_end > V) v_end = V;
      for (int v = v_start; v < v_end; v++) {
        const _Float16 *row = embed + v * D;
        int i = 0;
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);
        for (; i + 15 < D; i += 16) {
            float16x8_t a0 = vld1q_f16((const __fp16*)(row + i));
            float16x8_t b0 = vld1q_f16((const __fp16*)(x + i));
            float16x8_t a1 = vld1q_f16((const __fp16*)(row + i + 8));
            float16x8_t b1 = vld1q_f16((const __fp16*)(x + i + 8));
            acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(a0)), vcvt_f32_f16(vget_low_f16(b0)));
            acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(a0)), vcvt_f32_f16(vget_high_f16(b0)));
            acc2 = vfmaq_f32(acc2, vcvt_f32_f16(vget_low_f16(a1)), vcvt_f32_f16(vget_low_f16(b1)));
            acc3 = vfmaq_f32(acc3, vcvt_f32_f16(vget_high_f16(a1)), vcvt_f32_f16(vget_high_f16(b1)));
        }
        for (; i + 7 < D; i += 8) {
            float16x8_t a = vld1q_f16((const __fp16*)(row + i));
            float16x8_t b = vld1q_f16((const __fp16*)(x + i));
            acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(a)), vcvt_f32_f16(vget_low_f16(b)));
            acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(a)), vcvt_f32_f16(vget_high_f16(b)));
        }
        acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        float dot = vaddvq_f32(acc0);
        for (; i < D; i++) dot += (float)row[i] * (float)x[i];
        logits[v] = dot;
      }
    });
}

// ========== Sampling ==========
static int sample_argmax(const float *logits, int n) {
    int best = 0;
    float best_v = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    return best;
}

static int sample_topp(const float *logits, int n, float temperature, float topp) {
    // Temperature scaling + softmax
    float *probs = (float *)malloc(n * sizeof(float));
    float maxv = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > maxv) maxv = logits[i];
    float sum = 0;
    for (int i = 0; i < n; i++) {
        probs[i] = expf((logits[i] - maxv) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++) probs[i] /= sum;

    // Sort indices by probability descending
    int *idx = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) idx[i] = i;
    // Simple selection for top candidates (avoid full sort)
    float cumsum = 0;
    int cutoff = n;
    // Find top-p cutoff by partial sort
    for (int i = 0; i < n && cumsum < topp; i++) {
        // Find max remaining
        int best = i;
        for (int j = i + 1; j < n; j++)
            if (probs[idx[j]] > probs[idx[best]]) best = j;
        if (best != i) { int tmp = idx[i]; idx[i] = idx[best]; idx[best] = tmp; }
        cumsum += probs[idx[i]];
        cutoff = i + 1;
    }

    // Sample from top-p tokens
    float r = (float)drand48() * cumsum;
    float cs = 0;
    int result = idx[0];
    for (int i = 0; i < cutoff; i++) {
        cs += probs[idx[i]];
        if (cs >= r) { result = idx[i]; break; }
    }
    free(probs);
    free(idx);
    return result;
}

// ========== llama2.c tokenizer ==========
typedef struct {
    char **vocab;
    float *scores;
    int vocab_size;
} Tokenizer;

static Tokenizer *tokenizer_load(const char *path, int vocab_size) {
    Tokenizer *t = (Tokenizer *)calloc(1, sizeof(Tokenizer));
    t->vocab_size = vocab_size;
    t->vocab = (char **)calloc(vocab_size, sizeof(char*));
    t->scores = (float *)calloc(vocab_size, sizeof(float));

    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open tokenizer: %s\n", path); free(t); return NULL; }

    int max_len; fread(&max_len, 4, 1, f);  // unused
    for (int i = 0; i < vocab_size; i++) {
        fread(&t->scores[i], 4, 1, f);
        int slen; fread(&slen, 4, 1, f);
        t->vocab[i] = (char *)malloc(slen + 1);
        fread(t->vocab[i], 1, slen, f);
        t->vocab[i][slen] = '\0';
    }
    fclose(f);
    return t;
}

// Greedy BPE encode: start with one token per char, then merge best pairs
static int tokenizer_encode(const Tokenizer *t, const char *text, int *tokens, int max_tokens) {
    int n = 0;
    // Initial: one token per byte (find matching single-char vocab entries)
    for (int i = 0; text[i] && n < max_tokens; i++) {
        char s[2] = {text[i], '\0'};
        for (int v = 0; v < t->vocab_size; v++) {
            if (strcmp(t->vocab[v], s) == 0) { tokens[n++] = v; break; }
        }
    }

    // BPE merge loop
    char merge_buf[512];
    while (1) {
        float best_score = -1e30f;
        int best_idx = -1, best_tok = -1;
        for (int i = 0; i < n - 1; i++) {
            snprintf(merge_buf, sizeof(merge_buf), "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            for (int v = 0; v < t->vocab_size; v++) {
                if (strcmp(t->vocab[v], merge_buf) == 0 && t->scores[v] > best_score) {
                    best_score = t->scores[v];
                    best_idx = i;
                    best_tok = v;
                }
            }
        }
        if (best_idx < 0) break;
        tokens[best_idx] = best_tok;
        // Shift remaining tokens left
        for (int i = best_idx + 1; i < n - 1; i++) tokens[i] = tokens[i + 1];
        n--;
    }
    return n;
}

static const char *tokenizer_decode(const Tokenizer *t, int token) {
    if (token >= 0 && token < t->vocab_size) return t->vocab[token];
    return "";
}

static void tokenizer_free(Tokenizer *t) {
    if (!t) return;
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab); free(t->scores); free(t);
}

// ANE minimum spatial dimension for reliable operation
// S=1 may produce zeros on some hardware; S=4 is safe
#define DECODE_S 32

// ========== Per-layer ANE kernels ==========
typedef struct {
    Kern *qkv;         // QKV projection (ANE, no RMSNorm)
    Kern *wo;          // Wo projection (ANE)
    Kern *ffn;         // FFN (ANE, no RMSNorm, no residual)
    _Float16 *rms_att; // RMSNorm weights [dim] for attention (CPU)
    _Float16 *rms_ffn; // RMSNorm weights [dim] for FFN (CPU)
    _Float16 *q_norm;  // QK norm weights [head_dim]
    _Float16 *k_norm;  // QK norm weights [head_dim]
} InferLayerKernels;

// ========== GGUF tensor name helpers for Qwen3 ==========
static char *qwen3_name(char *buf, int layer, const char *suffix) {
    if (layer < 0)
        snprintf(buf, 256, "%s", suffix);
    else
        snprintf(buf, 256, "blk.%d.%s", layer, suffix);
    return buf;
}

// ========== LoRA merge: W_merged = W_base + (alpha/rank) * B @ A ==========
// W_base: [out, in] fp16, A: [rank, in] fp16, B: [out, rank] fp16
// Result: [out, in] fp16 (in-place into W_base)
static void lora_merge_fp16(_Float16 *W, int out_ch, int in_ch,
                            const _Float16 *A, const _Float16 *B, int rank, float alpha) {
    float scale = alpha / (float)rank;
    // W[i,j] += scale * sum_r(B[i,r] * A[r,j])
    for (int i = 0; i < out_ch; i++) {
        for (int j = 0; j < in_ch; j++) {
            float correction = 0;
            for (int r = 0; r < rank; r++)
                correction += (float)B[i * rank + r] * (float)A[r * in_ch + j];
            W[i * in_ch + j] += (_Float16)(correction * scale);
        }
    }
}

// ========== Main ==========
int main(int argc, char **argv) {
    @autoreleasepool {

    // Parse args
    const char *model_path = NULL;
    const char *lora_path = NULL;
    const char *xor_path = NULL;
    const char *prompt = "Once upon a time";
    int max_tokens = 128;
    float temperature = 0.7f;
    float topp = 0.9f;
    int use_stories = 0;  // --stories flag for testing with Stories110M

    static struct option long_opts[] = {
        {"model", required_argument, 0, 'm'},
        {"lora", required_argument, 0, 'l'},
        {"prompt", required_argument, 0, 'p'},
        {"max-tokens", required_argument, 0, 'n'},
        {"temp", required_argument, 0, 't'},
        {"topp", required_argument, 0, 'k'},
        {"stories", no_argument, 0, 's'},
        {"xor", required_argument, 0, 'x'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:l:p:n:t:k:sx:", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'm': model_path = optarg; break;
            case 'l': lora_path = optarg; break;
            case 'p': prompt = optarg; break;
            case 'n': max_tokens = atoi(optarg); break;
            case 't': temperature = atof(optarg); break;
            case 'k': topp = atof(optarg); break;
            case 's': use_stories = 1; break;
            case 'x': xor_path = optarg; break;
        }
    }

    if (!model_path && !use_stories) {
        fprintf(stderr, "Usage: %s --model <gguf> [--lora <gguf>] [--xor <patch.json>] "
                "[--prompt \"text\"] [--max-tokens N] [--temp F] [--topp F] [--stories]\n", argv[0]);
        return 1;
    }

    // Init ANE runtime
    ane_init();
    mach_timebase_info(&g_tb);

    InferConfig cfg;
    _Float16 *embed = NULL;      // [vocab, dim] embedding weights
    _Float16 *output_w = NULL;   // [vocab, dim] classifier weights (may == embed)
    _Float16 **wo_weights = NULL; // [n_layers] Wo weight arrays for CPU attention
    InferLayerKernels *layers = NULL;
    _Float16 *rms_final_w = NULL;

    if (use_stories) {
        // ====== Stories110M path (for testing) ======
        cfg = infer_config_stories110m();
        printf("Config: Stories110M (%d layers, dim=%d, heads=%d, kv_heads=%d)\n",
               cfg.n_layers, cfg.dim, cfg.n_heads, cfg.n_kv_heads);

        // Load from llama2.c format
        FILE *f = fopen(model_path ? model_path : "../../assets/models/stories110M.bin", "rb");
        if (!f) { fprintf(stderr, "Cannot open model file\n"); return 1; }
        int hdr[7]; fread(hdr, 4, 7, f);
        int V = abs(hdr[5]);

        // Read all weights into fp32 then convert
        float *tmp = (float *)malloc(V * cfg.dim * sizeof(float));
        fread(tmp, 4, V * cfg.dim, f);
        embed = (_Float16 *)malloc(V * cfg.dim * sizeof(_Float16));
        for (int i = 0; i < V * cfg.dim; i++) embed[i] = (_Float16)tmp[i];

        // rms_att weights per layer
        _Float16 **rms_att = (_Float16 **)calloc(cfg.n_layers, sizeof(_Float16*));
        for (int L = 0; L < cfg.n_layers; L++) {
            rms_att[L] = (_Float16 *)malloc(cfg.dim * sizeof(_Float16));
            fread(tmp, 4, cfg.dim, f);
            for (int i = 0; i < cfg.dim; i++) rms_att[L][i] = (_Float16)tmp[i];
        }

        // Wq, Wk, Wv, Wo per layer
        _Float16 **wq = (_Float16 **)calloc(cfg.n_layers, sizeof(_Float16*));
        _Float16 **wk = (_Float16 **)calloc(cfg.n_layers, sizeof(_Float16*));
        _Float16 **wv = (_Float16 **)calloc(cfg.n_layers, sizeof(_Float16*));
        wo_weights = (_Float16 **)calloc(cfg.n_layers, sizeof(_Float16*));

        for (int L = 0; L < cfg.n_layers; L++) {
            wq[L] = (_Float16 *)malloc(cfg.dim * cfg.dim * 2);
            fread(tmp, 4, cfg.dim * cfg.dim, f);
            for (int i = 0; i < cfg.dim * cfg.dim; i++) wq[L][i] = (_Float16)tmp[i];
        }
        for (int L = 0; L < cfg.n_layers; L++) {
            wk[L] = (_Float16 *)malloc(cfg.kv_dim * cfg.dim * 2);
            fread(tmp, 4, cfg.kv_dim * cfg.dim, f);
            for (int i = 0; i < cfg.kv_dim * cfg.dim; i++) wk[L][i] = (_Float16)tmp[i];
        }
        for (int L = 0; L < cfg.n_layers; L++) {
            wv[L] = (_Float16 *)malloc(cfg.kv_dim * cfg.dim * 2);
            fread(tmp, 4, cfg.kv_dim * cfg.dim, f);
            for (int i = 0; i < cfg.kv_dim * cfg.dim; i++) wv[L][i] = (_Float16)tmp[i];
        }
        for (int L = 0; L < cfg.n_layers; L++) {
            wo_weights[L] = (_Float16 *)malloc(cfg.dim * cfg.dim * 2);
            fread(tmp, 4, cfg.dim * cfg.dim, f);
            for (int i = 0; i < cfg.dim * cfg.dim; i++) wo_weights[L][i] = (_Float16)tmp[i];
        }

        // rms_ffn per layer
        _Float16 **rms_ffn = (_Float16 **)calloc(cfg.n_layers, sizeof(_Float16*));
        for (int L = 0; L < cfg.n_layers; L++) {
            rms_ffn[L] = (_Float16 *)malloc(cfg.dim * sizeof(_Float16));
            fread(tmp, 4, cfg.dim, f);
            for (int i = 0; i < cfg.dim; i++) rms_ffn[L][i] = (_Float16)tmp[i];
        }

        // W1, W2, W3 per layer
        _Float16 **w1 = (_Float16 **)calloc(cfg.n_layers, sizeof(_Float16*));
        _Float16 **w2 = (_Float16 **)calloc(cfg.n_layers, sizeof(_Float16*));
        _Float16 **w3 = (_Float16 **)calloc(cfg.n_layers, sizeof(_Float16*));
        for (int L = 0; L < cfg.n_layers; L++) {
            w1[L] = (_Float16 *)malloc(cfg.hidden_dim * cfg.dim * 2);
            fread(tmp, 4, cfg.hidden_dim * cfg.dim, f);
            for (int i = 0; i < cfg.hidden_dim * cfg.dim; i++) w1[L][i] = (_Float16)tmp[i];
        }
        for (int L = 0; L < cfg.n_layers; L++) {
            w2[L] = (_Float16 *)malloc(cfg.dim * cfg.hidden_dim * 2);
            fread(tmp, 4, cfg.dim * cfg.hidden_dim, f);
            for (int i = 0; i < cfg.dim * cfg.hidden_dim; i++) w2[L][i] = (_Float16)tmp[i];
        }
        for (int L = 0; L < cfg.n_layers; L++) {
            w3[L] = (_Float16 *)malloc(cfg.hidden_dim * cfg.dim * 2);
            fread(tmp, 4, cfg.hidden_dim * cfg.dim, f);
            for (int i = 0; i < cfg.hidden_dim * cfg.dim; i++) w3[L][i] = (_Float16)tmp[i];
        }

        // rms_final
        rms_final_w = (_Float16 *)malloc(cfg.dim * sizeof(_Float16));
        fread(tmp, 4, cfg.dim, f);
        for (int i = 0; i < cfg.dim; i++) rms_final_w[i] = (_Float16)tmp[i];
        fclose(f);
        free(tmp);

        // Compile ANE kernels per layer (decode mode: S=1)
        printf("Compiling %d layers of ANE kernels (decode S=%d)...\n", cfg.n_layers, DECODE_S);
        layers = (InferLayerKernels *)calloc(cfg.n_layers, sizeof(InferLayerKernels));

        for (int L = 0; L < cfg.n_layers; L++) {
            printf("  Layer %d/%d\r", L+1, cfg.n_layers); fflush(stdout);

            // QKV kernel (no RMSNorm — done on CPU)
            int io_s = DECODE_S;
            NSString *qkv_mil = gen_infer_qkv(&cfg, io_s);
            NSDictionary *qkv_w = @{
                @"@model_path/weights/wq.bin": @{@"offset":@0, @"data":bonsai_build_blob_fp16(wq[L], cfg.dim, cfg.dim)},
                @"@model_path/weights/wk.bin": @{@"offset":@0, @"data":bonsai_build_blob_fp16(wk[L], cfg.kv_dim, cfg.dim)},
                @"@model_path/weights/wv.bin": @{@"offset":@0, @"data":bonsai_build_blob_fp16(wv[L], cfg.kv_dim, cfg.dim)},
            };
            layers[L].qkv = compile_kern_mil_w(qkv_mil, qkv_w,
                cfg.dim * io_s * 2, (cfg.dim + 2 * cfg.kv_dim) * io_s * 2);

            // Wo projection kernel
            NSString *wo_mil = gen_infer_wo(&cfg, io_s);
            NSDictionary *wo_w = @{
                @"@model_path/weights/wo.bin": @{@"offset":@0, @"data":bonsai_build_blob_fp16(wo_weights[L], cfg.dim, cfg.dim)},
            };
            layers[L].wo = compile_kern_mil_w(wo_mil, wo_w, cfg.dim * io_s * 2, cfg.dim * io_s * 2);

            // FFN kernel (no RMSNorm, no residual)
            NSString *ffn_mil = gen_infer_ffn(&cfg, io_s);
            NSDictionary *ffn_w = @{
                @"@model_path/weights/w1.bin": @{@"offset":@0, @"data":bonsai_build_blob_fp16(w1[L], cfg.hidden_dim, cfg.dim)},
                @"@model_path/weights/w3.bin": @{@"offset":@0, @"data":bonsai_build_blob_fp16(w3[L], cfg.hidden_dim, cfg.dim)},
                @"@model_path/weights/w2.bin": @{@"offset":@0, @"data":bonsai_build_blob_fp16(w2[L], cfg.dim, cfg.hidden_dim)},
            };
            layers[L].ffn = compile_kern_mil_w(ffn_mil, ffn_w, cfg.dim * io_s * 2, cfg.dim * io_s * 2);

            // Store RMSNorm weights for CPU
            layers[L].rms_att = rms_att[L]; rms_att[L] = NULL;
            layers[L].rms_ffn = rms_ffn[L]; rms_ffn[L] = NULL;

            // Free layer weights baked into kernels
            free(wo_weights[L]); wo_weights[L] = NULL;
            free(wq[L]); free(wk[L]); free(wv[L]);
            free(rms_ffn[L]); free(w1[L]); free(w2[L]); free(w3[L]);
        }
        free(rms_att); free(wq); free(wk); free(wv);
        free(rms_ffn); free(w1); free(w2); free(w3);
        output_w = embed;  // Stories uses shared embed/unembed
        printf("  Done (%d kernels compiled)\n", g_compile_count);

    } else {
        // ====== Bonsai-8B GGUF path ======
        cfg = infer_config_bonsai_8b();
        printf("Config: Bonsai-8B (%d layers, dim=%d, heads=%d, kv_heads=%d, vocab=%d)\n",
               cfg.n_layers, cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size);

        GGUFFile *gf = gguf_open(model_path);
        if (!gf) { fprintf(stderr, "Failed to open GGUF: %s\n", model_path); return 1; }

        // Load LoRA adapter if provided
        GGUFFile *lora_gf = NULL;
        if (lora_path) {
            lora_gf = gguf_open(lora_path);
            if (!lora_gf) fprintf(stderr, "Warning: failed to open LoRA GGUF: %s\n", lora_path);
            else {
                cfg.lora_rank = 16;  // Bonsai default
                printf("LoRA adapter loaded (rank=%d, alpha=%.1f)\n", cfg.lora_rank, cfg.lora_alpha);
            }
        }

        // Load XOR patch if provided
        XORPatch *xor_patch = NULL;
        if (xor_path) {
            xor_patch = xor_patch_load(xor_path);
        }

        // Load embedding weights (may be Q1_0_g128, F16, or F32)
        char namebuf[256];
        int erows, ecols;
        embed = gguf_load_as_f16(gf, "token_embd.weight", &erows, &ecols);
        if (!embed) {
            fprintf(stderr, "Failed to load embedding weights\n");
            gguf_close(gf); return 1;
        }
        printf("Embeddings: %dx%d\n", erows, ecols);

        // Load output projection (separate from embed in Qwen3)
        int orows, ocols;
        output_w = gguf_load_as_f16(gf, "output.weight", &orows, &ocols);
        if (!output_w) {
            printf("No separate output.weight, using embed for classifier\n");
            output_w = embed;  // shared weights fallback
        } else {
            printf("Output projection: %dx%d\n", orows, ocols);
        }

        // Load rms_final (F32 in Qwen3 GGUF)
        int rr, rc;
        rms_final_w = gguf_load_as_f16(gf, "output_norm.weight", &rr, &rc);
        if (!rms_final_w) { fprintf(stderr, "Failed to load output_norm\n"); return 1; }

        // Compile ANE kernels per layer
        printf("Compiling %d layers...\n", cfg.n_layers);
        layers = (InferLayerKernels *)calloc(cfg.n_layers, sizeof(InferLayerKernels));
        wo_weights = (_Float16 **)calloc(cfg.n_layers, sizeof(_Float16*));

        for (int L = 0; L < cfg.n_layers; L++) {
            printf("  Layer %d/%d\r", L+1, cfg.n_layers); fflush(stdout);

            // Load layer weights from GGUF
            int r, c_dim;
            _Float16 *wq_f16 = gguf_load_q1_fp16(gf, qwen3_name(namebuf, L, "attn_q.weight"), &r, &c_dim);
            _Float16 *wk_f16 = gguf_load_q1_fp16(gf, qwen3_name(namebuf, L, "attn_k.weight"), &r, &c_dim);
            _Float16 *wv_f16 = gguf_load_q1_fp16(gf, qwen3_name(namebuf, L, "attn_v.weight"), &r, &c_dim);
            _Float16 *wo_f16 = gguf_load_q1_fp16(gf, qwen3_name(namebuf, L, "attn_output.weight"), &r, &c_dim);

            _Float16 *w1_f16 = gguf_load_q1_fp16(gf, qwen3_name(namebuf, L, "ffn_gate.weight"), &r, &c_dim);
            _Float16 *w3_f16 = gguf_load_q1_fp16(gf, qwen3_name(namebuf, L, "ffn_up.weight"), &r, &c_dim);
            _Float16 *w2_f16 = gguf_load_q1_fp16(gf, qwen3_name(namebuf, L, "ffn_down.weight"), &r, &c_dim);

            int rr2, rc2;
            _Float16 *rms1 = gguf_load_as_f16(gf, qwen3_name(namebuf, L, "attn_norm.weight"), &rr2, &rc2);
            _Float16 *rms2 = gguf_load_as_f16(gf, qwen3_name(namebuf, L, "ffn_norm.weight"), &rr2, &rc2);

            // QK norm weights (Qwen3)
            _Float16 *qnorm = NULL, *knorm = NULL;
            if (cfg.qk_norm) {
                qnorm = gguf_load_as_f16(gf, qwen3_name(namebuf, L, "attn_q_norm.weight"), &rr2, &rc2);
                knorm = gguf_load_as_f16(gf, qwen3_name(namebuf, L, "attn_k_norm.weight"), &rr2, &rc2);
            }

            if (!wq_f16 || !wk_f16 || !wv_f16 || !wo_f16 || !w1_f16 || !w3_f16 || !w2_f16 || !rms1 || !rms2) {
                fprintf(stderr, "Failed to load weights for layer %d\n", L);
                return 1;
            }

            // Merge LoRA if adapter provided
            if (lora_gf && cfg.lora_rank > 0) {
                char lora_name[256];
                int lr, lc;
                const char *projs[] = {"attn_q", "attn_k", "attn_v", "attn_output"};
                _Float16 *proj_ptrs[] = {wq_f16, wk_f16, wv_f16, wo_f16};
                int out_dims[] = {cfg.dim, cfg.kv_dim, cfg.kv_dim, cfg.dim};

                for (int p = 0; p < 4; p++) {
                    snprintf(lora_name, 256, "blk.%d.%s.weight.lora_a", L, projs[p]);
                    _Float16 *lora_a = gguf_load_f16(lora_gf, lora_name, &lr, &lc);
                    snprintf(lora_name, 256, "blk.%d.%s.weight.lora_b", L, projs[p]);
                    _Float16 *lora_b = gguf_load_f16(lora_gf, lora_name, &lr, &lc);
                    if (lora_a && lora_b) {
                        lora_merge_fp16(proj_ptrs[p], out_dims[p], cfg.dim,
                                       lora_a, lora_b, cfg.lora_rank, cfg.lora_alpha);
                        free(lora_a); free(lora_b);
                    }
                }
            }

            // Apply XOR patch flips (negate rows in weight matrices)
            if (xor_patch) {
                xor_patch_apply_f16(wq_f16, cfg.dim, cfg.dim, xor_patch, L, XOR_PROJ_Q);
                xor_patch_apply_f16(wk_f16, cfg.kv_dim, cfg.dim, xor_patch, L, XOR_PROJ_K);
                xor_patch_apply_f16(wv_f16, cfg.kv_dim, cfg.dim, xor_patch, L, XOR_PROJ_V);
                xor_patch_apply_f16(wo_f16, cfg.dim, cfg.dim, xor_patch, L, XOR_PROJ_O);
                xor_patch_apply_f16(w1_f16, cfg.hidden_dim, cfg.dim, xor_patch, L, XOR_PROJ_GATE);
                xor_patch_apply_f16(w3_f16, cfg.hidden_dim, cfg.dim, xor_patch, L, XOR_PROJ_UP);
                xor_patch_apply_f16(w2_f16, cfg.dim, cfg.hidden_dim, xor_patch, L, XOR_PROJ_DOWN);
            }

            // QKV kernel (no RMSNorm — done on CPU in fp32)
            int io_s = DECODE_S;
            NSString *qkv_mil = gen_infer_qkv(&cfg, io_s);
            NSDictionary *qkv_w = @{
                @"@model_path/weights/wq.bin": @{@"offset":@0, @"data":bonsai_build_blob_fp16(wq_f16, cfg.dim, cfg.dim)},
                @"@model_path/weights/wk.bin": @{@"offset":@0, @"data":bonsai_build_blob_fp16(wk_f16, cfg.kv_dim, cfg.dim)},
                @"@model_path/weights/wv.bin": @{@"offset":@0, @"data":bonsai_build_blob_fp16(wv_f16, cfg.kv_dim, cfg.dim)},
            };
            layers[L].qkv = compile_kern_mil_w(qkv_mil, qkv_w,
                cfg.dim*io_s*2, (cfg.dim+2*cfg.kv_dim)*io_s*2);

            // Wo projection kernel (ANE)
            NSString *wo_mil = gen_infer_wo(&cfg, io_s);
            NSDictionary *wo_w = @{
                @"@model_path/weights/wo.bin": @{@"offset":@0, @"data":bonsai_build_blob_fp16(wo_f16, cfg.dim, cfg.dim)},
            };
            layers[L].wo = compile_kern_mil_w(wo_mil, wo_w, cfg.dim*io_s*2, cfg.dim*io_s*2);

            // FFN kernel (no RMSNorm, no residual)
            NSString *ffn_mil = gen_infer_ffn(&cfg, io_s);
            NSDictionary *ffn_w = @{
                @"@model_path/weights/w1.bin": @{@"offset":@0, @"data":bonsai_build_blob_fp16(w1_f16, cfg.hidden_dim, cfg.dim)},
                @"@model_path/weights/w3.bin": @{@"offset":@0, @"data":bonsai_build_blob_fp16(w3_f16, cfg.hidden_dim, cfg.dim)},
                @"@model_path/weights/w2.bin": @{@"offset":@0, @"data":bonsai_build_blob_fp16(w2_f16, cfg.dim, cfg.hidden_dim)},
            };
            layers[L].ffn = compile_kern_mil_w(ffn_mil, ffn_w, cfg.dim*io_s*2, cfg.dim*io_s*2);

            // Store RMS norm weights for CPU, QK norm weights
            layers[L].rms_att = rms1; rms1 = NULL;
            layers[L].rms_ffn = rms2; rms2 = NULL;
            layers[L].q_norm = qnorm;
            layers[L].k_norm = knorm;

            // Free weights baked into kernels
            free(wq_f16); free(wk_f16); free(wv_f16); free(wo_f16);
            free(w1_f16); free(w2_f16); free(w3_f16);
        }
        printf("  Done (%d kernels compiled)\n", g_compile_count);

        if (lora_gf) gguf_close(lora_gf);
        xor_patch_free(xor_patch);
        gguf_close(gf);
    }

    // ====== Tokenization ======
    Tokenizer *tokenizer = NULL;        // llama2.c tokenizer (Stories)
    BPETokenizer *bpe_tokenizer = NULL; // GGUF BPE tokenizer (Bonsai/Qwen3)

    if (use_stories) {
        const char *tok_paths[] = {
            "/Users/mcm/Development/_ditch/assets/models/tokenizer.bin",
            "../../assets/models/tokenizer.bin", "tokenizer.bin", NULL
        };
        for (int i = 0; tok_paths[i]; i++) {
            if (access(tok_paths[i], R_OK) == 0) {
                tokenizer = tokenizer_load(tok_paths[i], cfg.vocab_size);
                if (tokenizer) printf("Tokenizer: %d tokens from %s\n", tokenizer->vocab_size, tok_paths[i]);
                break;
            }
        }
    } else if (model_path) {
        bpe_tokenizer = bpe_tokenizer_from_gguf(model_path);
    }

    // ====== Generation loop ======
    printf("\nPrompt: \"%s\"\n", prompt);
    printf("Generating (max %d tokens, temp=%.2f, topp=%.2f)...\n\n", max_tokens, temperature, topp);

    int *tokens = (int *)malloc((cfg.max_seq + max_tokens) * sizeof(int));
    int n_tokens = 0;

    if (use_stories && tokenizer) {
        tokens[n_tokens++] = 1;  // BOS
        n_tokens += tokenizer_encode(tokenizer, prompt, tokens + 1, cfg.max_seq - 1);
    } else if (bpe_tokenizer) {
        n_tokens = bpe_encode(bpe_tokenizer, prompt, tokens, cfg.max_seq);
    } else {
        for (int i = 0; prompt[i] && n_tokens < cfg.max_seq; i++)
            tokens[n_tokens++] = (unsigned char)prompt[i];
    }
    printf("[%d prompt tokens:", n_tokens);
    for (int i = 0; i < n_tokens && i < 8; i++) printf(" %d", tokens[i]);
    if (n_tokens > 8) printf(" ...");
    printf("]\n");

    // Allocate inference state
    KVCache *kv = kv_cache_alloc(&cfg);
    AttnDecodeScratch *attn_scratch = attn_decode_scratch_alloc(&cfg);
    float *x = (float *)calloc(cfg.dim, sizeof(float));                  // fp32 residual stream
    _Float16 *x_f16 = (_Float16 *)calloc(cfg.dim, sizeof(_Float16));    // fp16 for ANE I/O
    _Float16 *delta_f16 = (_Float16 *)calloc(cfg.dim, sizeof(_Float16));// fp16 delta from ANE
    _Float16 *q_buf = (_Float16 *)calloc(cfg.dim, sizeof(_Float16));
    _Float16 *k_buf = (_Float16 *)calloc(cfg.kv_dim, sizeof(_Float16));
    _Float16 *v_buf = (_Float16 *)calloc(cfg.kv_dim, sizeof(_Float16));
    _Float16 *x_norm = (_Float16 *)calloc(cfg.dim, sizeof(_Float16));
    float *logits = (float *)malloc(cfg.vocab_size * sizeof(float));

    uint64_t t_start = mach_absolute_time();
    int generated = 0;

    double t_ane = 0, t_attn = 0, t_cls = 0, t_other = 0;

    // Process tokens one at a time (simple decode loop, no prefill optimization yet)
    for (int pos = 0; pos < n_tokens + max_tokens - 1; pos++) {
        int tok = (pos < n_tokens) ? tokens[pos] : tokens[n_tokens - 1 + generated];

        // Embedding lookup: x = embed[tok] (fp16 → fp32)
        for (int ci = 0; ci < cfg.dim; ci++)
            x[ci] = (float)embed[tok * cfg.dim + ci];


        // Process through all layers
        for (int L = 0; L < cfg.n_layers; L++) {
            uint64_t t0 = mach_absolute_time();
            // CPU RMSNorm (fp32→fp16) then QKV projection on ANE
            rmsnorm_f32_to_f16(cfg.dim, x_f16, x, layers[L].rms_att);

            IOSurfaceLock(layers[L].qkv->ioIn, 0, NULL);
            _Float16 *qkv_inp = (_Float16*)IOSurfaceGetBaseAddress(layers[L].qkv->ioIn);
            memset(qkv_inp, 0, cfg.dim * DECODE_S * sizeof(_Float16));
            for (int ci = 0; ci < cfg.dim; ci++)
                qkv_inp[ci * DECODE_S] = x_f16[ci];
            IOSurfaceUnlock(layers[L].qkv->ioIn, 0, NULL);

            ane_eval(layers[L].qkv);
            uint64_t t1 = mach_absolute_time();

            // Read QKV output — channel-first [dim+2*kv_dim, DECODE_S], extract position 0
            IOSurfaceLock(layers[L].qkv->ioOut, kIOSurfaceLockReadOnly, NULL);
            const _Float16 *qkv_outp = (const _Float16*)IOSurfaceGetBaseAddress(layers[L].qkv->ioOut);
            int qkv_S = DECODE_S;
            // Q: channels [0, dim), K: [dim, dim+kv_dim), V: [dim+kv_dim, dim+2*kv_dim)
            for (int ci = 0; ci < cfg.dim; ci++)
                q_buf[ci] = qkv_outp[ci * qkv_S];
            for (int ci = 0; ci < cfg.kv_dim; ci++)
                k_buf[ci] = qkv_outp[(cfg.dim + ci) * qkv_S];
            for (int ci = 0; ci < cfg.kv_dim; ci++)
                v_buf[ci] = qkv_outp[(cfg.dim + cfg.kv_dim + ci) * qkv_S];

            IOSurfaceUnlock(layers[L].qkv->ioOut, kIOSurfaceLockReadOnly, NULL);

            // QK norm (Qwen3): per-head RMSNorm on Q and K before RoPE
            if (cfg.qk_norm && layers[L].q_norm)
                qk_norm_f16(&cfg, q_buf, k_buf, layers[L].q_norm, layers[L].k_norm);

            // RoPE on Q and K for this position
            rope_single_pos(&cfg, q_buf, k_buf, pos);

            // Append K, V to cache
            kv_cache_append1(kv, L, k_buf, v_buf);

            uint64_t t2 = mach_absolute_time();
            // CPU attention decode → raw attention output [dim]
            int T = pos + 1;
            cpu_attn_decode(&cfg, q_buf, kv->layers[L].k, kv->layers[L].v,
                           delta_f16, T, attn_scratch);

            // Wo projection on ANE: attn_out → Wo @ attn_out
            IOSurfaceLock(layers[L].wo->ioIn, 0, NULL);
            _Float16 *wo_inp = (_Float16*)IOSurfaceGetBaseAddress(layers[L].wo->ioIn);
            memset(wo_inp, 0, cfg.dim * DECODE_S * sizeof(_Float16));
            for (int ci = 0; ci < cfg.dim; ci++)
                wo_inp[ci * DECODE_S] = delta_f16[ci];
            IOSurfaceUnlock(layers[L].wo->ioIn, 0, NULL);

            ane_eval(layers[L].wo);

            IOSurfaceLock(layers[L].wo->ioOut, kIOSurfaceLockReadOnly, NULL);
            const _Float16 *wo_outp = (const _Float16*)IOSurfaceGetBaseAddress(layers[L].wo->ioOut);
            for (int ci = 0; ci < cfg.dim; ci++)
                x[ci] += (float)wo_outp[ci * DECODE_S];  // residual add in fp32
            IOSurfaceUnlock(layers[L].wo->ioOut, kIOSurfaceLockReadOnly, NULL);


            uint64_t t3 = mach_absolute_time();
            // CPU RMSNorm (fp32→fp16) then FFN on ANE
            rmsnorm_f32_to_f16(cfg.dim, x_f16, x, layers[L].rms_ffn);

            IOSurfaceLock(layers[L].ffn->ioIn, 0, NULL);
            _Float16 *ffn_inp = (_Float16*)IOSurfaceGetBaseAddress(layers[L].ffn->ioIn);
            memset(ffn_inp, 0, cfg.dim * DECODE_S * sizeof(_Float16));
            for (int ci = 0; ci < cfg.dim; ci++)
                ffn_inp[ci * DECODE_S] = x_f16[ci];
            IOSurfaceUnlock(layers[L].ffn->ioIn, 0, NULL);

            ane_eval(layers[L].ffn);

            IOSurfaceLock(layers[L].ffn->ioOut, kIOSurfaceLockReadOnly, NULL);
            const _Float16 *ffn_outp = (const _Float16*)IOSurfaceGetBaseAddress(layers[L].ffn->ioOut);
            for (int ci = 0; ci < cfg.dim; ci++)
                x[ci] += (float)ffn_outp[ci * DECODE_S];  // residual add in fp32
            IOSurfaceUnlock(layers[L].ffn->ioOut, kIOSurfaceLockReadOnly, NULL);

            uint64_t t4 = mach_absolute_time();
            t_ane += tb_ms(t1 - t0) + tb_ms(t4 - t3);  // QKV + FFN ANE
            t_attn += tb_ms(t3 - t2);  // attention + Wo ANE
            t_other += tb_ms(t2 - t1);  // QKV I/O
        }

        // Advance KV cache position
        kv_cache_advance(kv, 1);


        // Only sample after processing prompt
        if (pos >= n_tokens - 1) {
            // Final RMSNorm in fp32 → fp16
            rmsnorm_f32_to_f16(cfg.dim, x_norm, x, rms_final_w);

            // Classifier: embed @ x_norm → logits
            uint64_t tc0 = mach_absolute_time();
            classifier_f16(&cfg, logits, output_w, x_norm);
            t_cls += tb_ms(mach_absolute_time() - tc0);

            // Sample
            int next_tok;
            if (temperature < 0.01f)
                next_tok = sample_argmax(logits, cfg.vocab_size);
            else
                next_tok = sample_topp(logits, cfg.vocab_size, temperature, topp);

            generated++;
            tokens[n_tokens - 1 + generated] = next_tok;

            // Print token
            // Check EOS
            if (next_tok == 0) break;  // PAD
            if (next_tok == 2) break;  // llama2 EOS
            if (bpe_tokenizer && next_tok == bpe_tokenizer->eos_id) break;

            // Decode and print
            const char *piece = NULL;
            if (bpe_tokenizer)
                piece = bpe_decode_token(bpe_tokenizer, next_tok);
            else if (tokenizer)
                piece = tokenizer_decode(tokenizer, next_tok);

            if (piece && piece[0]) {
                if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && strlen(piece) == 6) {
                    printf("%c", (char)strtol(piece + 3, NULL, 16));
                } else {
                    printf("%s", piece);
                }
            } else if (next_tok > 0 && next_tok < 256) {
                printf("%c", (char)next_tok);
            }
            fflush(stdout);

            if (generated >= max_tokens) break;
            if (pos + 1 >= cfg.max_seq) { printf("\n[max seq reached]"); break; }
        }
    }

    uint64_t t_end = mach_absolute_time();
    double elapsed_ms = tb_ms(t_end - t_start);
    printf("\n\n--- %d tokens in %.1f ms (%.1f ms/tok, %.1f tok/s) ---\n"
           "    ane=%.1fms attn=%.1fms cls=%.1fms io=%.1fms\n",
           generated, elapsed_ms, elapsed_ms / (generated ? generated : 1),
           generated * 1000.0 / elapsed_ms,
           t_ane, t_attn, t_cls, t_other);

    // Cleanup
    kv_cache_free(kv);
    attn_decode_scratch_free(attn_scratch);
    free(x); free(x_f16); free(delta_f16); free(q_buf); free(k_buf); free(v_buf); free(x_norm);
    free(logits); free(tokens); free(embed);
    if (output_w != embed) free(output_w);
    free(rms_final_w);
    for (int L = 0; L < cfg.n_layers; L++) {
        free_kern(layers[L].qkv);
        free_kern(layers[L].wo);
        free_kern(layers[L].ffn);
        free(layers[L].rms_att);
        free(layers[L].rms_ffn);
        free(layers[L].q_norm);
        free(layers[L].k_norm);
    }
    free(layers);
    if (wo_weights) free(wo_weights);
    tokenizer_free(tokenizer);
    bpe_tokenizer_free(bpe_tokenizer);

    } // @autoreleasepool
    return 0;
}
