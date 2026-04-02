// bonsai_lora.h — LoRA kernel generators for Bonsai-8B 1-bit models on ANE
// Fused conv: y = conv(W_binary, x) + conv(A, conv(B, x))
// W_binary: frozen 1-bit weights dequantized to fp16 [out_ch, in_ch]
// B: LoRA down-projection [rank, in_ch] (projects x to low-rank space)
// A: LoRA up-projection [out_ch, rank] (projects back to full dim)
#pragma once
#import <Foundation/Foundation.h>
#include <math.h>

// Bonsai-8B config
#define BONSAI_DIM 4096
#define BONSAI_HIDDEN 11008
#define BONSAI_HEADS 32
#define BONSAI_KV_HEADS 4
#define BONSAI_HD (BONSAI_DIM / BONSAI_HEADS)  // 128
#define BONSAI_KV_DIM (BONSAI_KV_HEADS * BONSAI_HD)  // 512
#define BONSAI_LAYERS 28
#define BONSAI_VOCAB 151669
#define BONSAI_Q1_GROUP 128  // Q1_0_g128: 128 weights per group

// ===== Q1_0_g128 dequantization =====
// GGUF format: each block = 2 bytes fp16 scale + 16 bytes packed sign bits (128 bits)
// Reconstruct: weight[i] = sign_bit[i] ? -scale : +scale
typedef struct {
    _Float16 scale;
    uint8_t signs[16];  // 128 bits = 16 bytes
} Q1Block;

// Dequantize Q1_0_g128 blocks to fp32 array
// n_blocks = (out_ch * in_ch) / 128
static void q1_dequantize(float *out, const Q1Block *blocks, int n_weights) {
    int n_blocks = n_weights / BONSAI_Q1_GROUP;
    for (int b = 0; b < n_blocks; b++) {
        float scale = (float)blocks[b].scale;
        for (int i = 0; i < BONSAI_Q1_GROUP; i++) {
            int byte_idx = i / 8;
            int bit_idx = i % 8;
            int sign = (blocks[b].signs[byte_idx] >> bit_idx) & 1;
            out[b * BONSAI_Q1_GROUP + i] = sign ? -scale : scale;
        }
    }
}

// Dequantize directly to fp16 blob for ANE weight baking
static _Float16 *q1_dequantize_fp16(const Q1Block *blocks, int n_weights) {
    int n_blocks = n_weights / BONSAI_Q1_GROUP;
    _Float16 *out = (_Float16 *)malloc(n_weights * sizeof(_Float16));
    for (int b = 0; b < n_blocks; b++) {
        _Float16 scale = blocks[b].scale;
        for (int i = 0; i < BONSAI_Q1_GROUP; i++) {
            int byte_idx = i / 8;
            int bit_idx = i % 8;
            int sign = (blocks[b].signs[byte_idx] >> bit_idx) & 1;
            out[b * BONSAI_Q1_GROUP + i] = sign ? -scale : scale;
        }
    }
    return out;
}

// ===== Weight blob builders =====

// Build fp16 blob from fp32 weights (reusable)
static NSData *bonsai_build_blob(const float *w, int out_ch, int in_ch) {
    NSUInteger wsize = (NSUInteger)out_ch * in_ch * 2;
    NSUInteger total = 64 + 64 + wsize;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t *chunk = buf + 64;
    chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;
    chunk[4] = 0x01;
    *(uint32_t *)(chunk + 8) = (uint32_t)wsize;
    *(uint32_t *)(chunk + 16) = 128;
    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (NSUInteger i = 0; i < (NSUInteger)out_ch * in_ch; i++)
        fp16[i] = (_Float16)w[i];
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// Build fp16 blob directly from fp16 data (for pre-dequantized Q1 weights)
static NSData *bonsai_build_blob_fp16(const _Float16 *w, int out_ch, int in_ch) {
    NSUInteger wsize = (NSUInteger)out_ch * in_ch * 2;
    NSUInteger total = 64 + 64 + wsize;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t *chunk = buf + 64;
    chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;
    chunk[4] = 0x01;
    *(uint32_t *)(chunk + 8) = (uint32_t)wsize;
    *(uint32_t *)(chunk + 16) = 128;
    memcpy(buf + 128, w, wsize);
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// Chunk size for multi-weight blob layout: 64-byte header + data
static NSUInteger blob_chunk_size(int out_ch, int in_ch) {
    return 64 + (NSUInteger)out_ch * in_ch * 2;
}

// Build combined LoRA weight blob: W[out,in] + B[rank,in] + A[out,rank]
// Layout: global_header(64) | W_chunk(64+data) | B_chunk(64+data) | A_chunk(64+data)
static NSData *bonsai_build_lora_blob(const float *W, int out_ch, int in_ch,
                                       const float *B, const float *A, int rank) {
    NSUInteger w_data = (NSUInteger)out_ch * in_ch * 2;
    NSUInteger b_data = (NSUInteger)rank * in_ch * 2;
    NSUInteger a_data = (NSUInteger)out_ch * rank * 2;
    NSUInteger w_chunk = 64 + w_data;
    NSUInteger b_chunk = 64 + b_data;
    NSUInteger a_chunk = 64 + a_data;
    NSUInteger total = 64 + w_chunk + b_chunk + a_chunk;

    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;

    // W chunk
    uint8_t *wc = buf + 64;
    wc[0] = 0xEF; wc[1] = 0xBE; wc[2] = 0xAD; wc[3] = 0xDE; wc[4] = 0x01;
    *(uint32_t *)(wc + 8) = (uint32_t)w_data;
    *(uint32_t *)(wc + 16) = (uint32_t)(64 + 64);  // absolute offset
    _Float16 *wp = (_Float16 *)(wc + 64);
    for (NSUInteger i = 0; i < (NSUInteger)out_ch * in_ch; i++) wp[i] = (_Float16)W[i];

    // B chunk (LoRA down: [rank, in_ch])
    uint8_t *bc = buf + 64 + w_chunk;
    bc[0] = 0xEF; bc[1] = 0xBE; bc[2] = 0xAD; bc[3] = 0xDE; bc[4] = 0x01;
    *(uint32_t *)(bc + 8) = (uint32_t)b_data;
    *(uint32_t *)(bc + 16) = (uint32_t)(64 + w_chunk + 64);
    _Float16 *bp = (_Float16 *)(bc + 64);
    for (NSUInteger i = 0; i < (NSUInteger)rank * in_ch; i++) bp[i] = (_Float16)B[i];

    // A chunk (LoRA up: [out_ch, rank])
    uint8_t *ac = buf + 64 + w_chunk + b_chunk;
    ac[0] = 0xEF; ac[1] = 0xBE; ac[2] = 0xAD; ac[3] = 0xDE; ac[4] = 0x01;
    *(uint32_t *)(ac + 8) = (uint32_t)a_data;
    *(uint32_t *)(ac + 16) = (uint32_t)(64 + w_chunk + b_chunk + 64);
    _Float16 *ap = (_Float16 *)(ac + 64);
    for (NSUInteger i = 0; i < (NSUInteger)out_ch * rank; i++) ap[i] = (_Float16)A[i];

    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// ===== MIL generators =====

// Generate MIL for LoRA-fused conv:
// y = conv(W, x) + conv(A, conv(B, x))
// Input:  [1, in_ch, 1, spatial] fp32
// Output: [1, out_ch, 1, spatial] fp32
// Weights in separate files: w.bin[out_ch,in_ch], lora_b.bin[rank,in_ch], lora_a.bin[out_ch,rank]
static NSString *bonsai_gen_lora_conv(int in_ch, int out_ch, int rank, int spatial) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        string pad_type = const()[name = string(\"pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> strides = const()[name = string(\"strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pad = const()[name = string(\"pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dilations = const()[name = string(\"dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 groups = const()[name = string(\"groups\"), val = int32(1)];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> B = const()[name = string(\"B\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/lora_b.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> A = const()[name = string(\"A\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/lora_a.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> wx = conv(dilations = dilations, groups = groups, "
        "pad = pad, pad_type = pad_type, strides = strides, weight = W, x = x)[name = string(\"conv_w\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> bx = conv(dilations = dilations, groups = groups, "
        "pad = pad, pad_type = pad_type, strides = strides, weight = B, x = x)[name = string(\"conv_b\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> abx = conv(dilations = dilations, groups = groups, "
        "pad = pad, pad_type = pad_type, strides = strides, weight = A, x = bx)[name = string(\"conv_a\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y = add(x = wx, y = abx)[name = string(\"lora_add\")];\n"
        "    } -> (y);\n"
        "}\n",
        in_ch, spatial,                                    // input
        out_ch, in_ch, out_ch, in_ch,                     // W
        rank, in_ch, rank, in_ch,                         // B
        out_ch, rank, out_ch, rank,                       // A
        out_ch, spatial,                                   // conv_w
        rank, spatial,                                     // conv_b
        out_ch, spatial,                                   // conv_a
        out_ch, spatial];                                  // lora_add
}

// Build weight dict for LoRA kernel (separate files per weight)
static NSDictionary *bonsai_lora_weights(const float *W, int out_ch, int in_ch,
                                          const float *B, const float *A, int rank) {
    return @{
        @"@model_path/weights/w.bin": @{@"offset": @0, @"data": bonsai_build_blob(W, out_ch, in_ch)},
        @"@model_path/weights/lora_b.bin": @{@"offset": @0, @"data": bonsai_build_blob(B, rank, in_ch)},
        @"@model_path/weights/lora_a.bin": @{@"offset": @0, @"data": bonsai_build_blob(A, out_ch, rank)},
    };
}

// Generate MIL for plain conv (no LoRA) — for frozen layers or baseline comparison
static NSString *bonsai_gen_conv(int in_ch, int out_ch, int spatial) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        string pad_type = const()[name = string(\"pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> strides = const()[name = string(\"strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pad = const()[name = string(\"pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dilations = const()[name = string(\"dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 groups = const()[name = string(\"groups\"), val = int32(1)];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations = dilations, groups = groups, "
        "pad = pad, pad_type = pad_type, strides = strides, weight = W, x = x)[name = string(\"conv\")];\n"
        "    } -> (y);\n"
        "}\n",
        in_ch, spatial,
        out_ch, in_ch, out_ch, in_ch,
        out_ch, spatial];
}

// ===== LoRA initialization =====

// Kaiming init for LoRA A (normal, fan_out), zero init for B
// This matches PyTorch LoRA convention: A has signal, B starts at zero → adapter starts as identity
static void lora_init(float *A, float *B, int out_ch, int in_ch, int rank) {
    // B = zero (adapter contribution starts as zero)
    memset(B, 0, (size_t)rank * in_ch * sizeof(float));
    // A = kaiming normal (fan_out = out_ch)
    float std = 1.0f / sqrtf((float)rank);
    srand48(42);
    for (size_t i = 0; i < (size_t)out_ch * rank; i++) {
        // Box-Muller for normal distribution
        double u1 = drand48(), u2 = drand48();
        A[i] = std * (float)(sqrt(-2.0 * log(u1 + 1e-12)) * cos(2.0 * M_PI * u2));
    }
}
