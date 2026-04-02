// infer_config.h — Runtime model configuration for inference
#pragma once
#include <stdint.h>

typedef struct {
    int dim;            // model dimension (4096 for Bonsai-8B)
    int hidden_dim;     // FFN hidden dimension (11008)
    int n_heads;        // number of query heads (32)
    int n_kv_heads;     // number of KV heads for GQA (4)
    int head_dim;       // dim / n_heads (128)
    int kv_dim;         // n_kv_heads * head_dim (512)
    int kv_group;       // n_heads / n_kv_heads (8) — repeat factor for GQA
    int n_layers;       // number of transformer layers (28)
    int vocab_size;     // vocabulary size (151669)
    int max_seq;        // max sequence length for KV cache
    int lora_rank;      // 0 = no LoRA
    float lora_alpha;   // LoRA scaling (32.0)
    float rope_theta;   // RoPE base frequency (1000000.0 for Qwen3)
} InferConfig;

static InferConfig infer_config_bonsai_8b(void) {
    return (InferConfig){
        .dim        = 4096,
        .hidden_dim = 11008,
        .n_heads    = 32,
        .n_kv_heads = 4,
        .head_dim   = 128,
        .kv_dim     = 4 * 128,   // 512
        .kv_group   = 32 / 4,    // 8
        .n_layers   = 28,
        .vocab_size = 151669,
        .max_seq    = 2048,
        .lora_rank  = 0,
        .lora_alpha = 32.0f,
        .rope_theta = 1000000.0f,
    };
}

// Stories110M config for testing inference pipeline
static InferConfig infer_config_stories110m(void) {
    return (InferConfig){
        .dim        = 768,
        .hidden_dim = 2048,
        .n_heads    = 12,
        .n_kv_heads = 12,
        .head_dim   = 64,
        .kv_dim     = 12 * 64,   // 768
        .kv_group   = 1,
        .n_layers   = 12,
        .vocab_size = 32000,
        .max_seq    = 256,
        .lora_rank  = 0,
        .lora_alpha = 0.0f,
        .rope_theta = 10000.0f,
    };
}
