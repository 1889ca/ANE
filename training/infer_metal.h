// infer_metal.h — Full Metal GPU inference for all projections
// Replaces ANE for large models where baked-weight eval overhead dominates
// Uses shared memory (zero-copy on Apple Silicon unified memory)
#pragma once
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "infer_config.h"

static NSString *const metal_shader_source = @R"(
#include <metal_stdlib>
using namespace metal;

// Matrix-vector multiply: one simdgroup (32 threads) per output row
// 32 threads stride through K elements with half4 loads, reduce via simd_sum
// AMX simdgroup_matrix was tested but threadgroup broadcast overhead
// for the x vector tile outweighed the AMX throughput gain.
kernel void matvec_f16(
    device const half *W [[buffer(0)]],
    device const half *x [[buffer(1)]],
    device half *out [[buffer(2)]],
    constant uint &K [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    float sum = 0.0f;
    uint base = row * K;
    for (uint j = lane * 4; j < K; j += 128) {
        half4 w = *reinterpret_cast<device const half4*>(W + base + j);
        half4 v = *reinterpret_cast<device const half4*>(x + j);
        sum += dot(float4(w), float4(v));
    }
    sum = simd_sum(sum);
    if (lane == 0) out[row] = half(sum);
}

// Residual add: out[i] = a[i] + b[i]
kernel void residual_add_f16(
    device const half *a [[buffer(0)]],
    device const half *b [[buffer(1)]],
    device half *out [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    out[i] = a[i] + b[i];
}

// RMSNorm: out[i] = x[i] * rrms * w[i]  where rrms = 1/sqrt(mean(x^2) + eps)
// Two-phase: phase 1 computes rrms into scratch[0], phase 2 normalizes
// Must be dispatched as ONE threadgroup of 1024 threads (handles dim up to 4096)
kernel void rmsnorm_f16(
    device const half *x [[buffer(0)]],
    device const half *w [[buffer(1)]],
    device half *out [[buffer(2)]],
    constant uint &dim [[buffer(3)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]])
{
    // Phase 1: compute sum of squares
    float local_ss = 0.0f;
    for (uint i = tid; i < dim; i += 1024)
        local_ss += float(x[i]) * float(x[i]);
    shared[tid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction (1024 → 1)
    for (uint s = 512; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rrms = rsqrt(shared[0] / float(dim) + 1e-6f);

    // Phase 2: normalize
    for (uint i = tid; i < dim; i += 1024)
        out[i] = half(float(x[i]) * rrms * float(w[i]));
}

// Fused SiLU gate: out[i] = silu(h1[i]) * h3[i]
kernel void silu_gate_f16(
    device const half *h1 [[buffer(0)]],
    device const half *h3 [[buffer(1)]],
    device half *out [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    float v = float(h1[i]);
    float sig = 1.0f / (1.0f + exp(-v));
    out[i] = half(v * sig * float(h3[i]));
}
)";

// Per-layer weight buffers
typedef struct {
    id<MTLBuffer> Wqkv;    // fused [dim + 2*kv_dim, dim] fp16
    id<MTLBuffer> rms_att; // [dim] fp16 attention RMSNorm weights
    id<MTLBuffer> Wo;      // [dim, dim] fp16
    id<MTLBuffer> rms_ffn; // [dim] fp16 FFN RMSNorm weights
    id<MTLBuffer> W1;      // [hidden, dim] fp16
    id<MTLBuffer> W3;      // [hidden, dim] fp16
    id<MTLBuffer> W2;      // [dim, hidden] fp16
} MetalLayerWeights;

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> matvec_pipe;
    id<MTLComputePipelineState> silu_gate_pipe;
    id<MTLComputePipelineState> residual_add_pipe;
    id<MTLComputePipelineState> rmsnorm_pipe;
    // Shared I/O buffers (reused across layers)
    id<MTLBuffer> buf_x;       // [dim] input
    id<MTLBuffer> buf_qkv;     // [dim + 2*kv_dim] QKV output
    id<MTLBuffer> buf_attn;    // [dim] attention output (for Wo)
    id<MTLBuffer> buf_wo;      // [dim] Wo output
    id<MTLBuffer> buf_ffn_in;  // [dim] FFN input (after rmsnorm)
    id<MTLBuffer> buf_h13;     // [2*hidden] fused W1+W3 output
    id<MTLBuffer> buf_gate;    // [hidden] after silu*gate
    id<MTLBuffer> buf_ffn_out; // [dim] FFN output
    id<MTLBuffer> buf_x_res;   // [dim] x residual (for fused Wo+FFN)
    id<MTLBuffer> buf_x2;      // [dim] x + Wo(attn) intermediate
    MetalLayerWeights *layers;
    id<MTLBuffer> buf_cls_w;   // [vocab, dim] classifier weights
    id<MTLBuffer> buf_cls_out; // [vocab] classifier logits (fp16 → convert to fp32 on CPU)
    int n_layers, dim, kv_dim, hidden_dim, vocab_size;
    NSUInteger tgSize;
} MetalInfer;

static MetalInfer *metal_init(const InferConfig *cfg) {
    MetalInfer *m = (MetalInfer *)calloc(1, sizeof(MetalInfer));
    m->dim = cfg->dim;
    m->kv_dim = cfg->kv_dim;
    m->hidden_dim = cfg->hidden_dim;
    m->n_layers = cfg->n_layers;
    m->vocab_size = cfg->vocab_size;

    m->device = MTLCreateSystemDefaultDevice();
    if (!m->device) { fprintf(stderr, "Metal: no device\n"); free(m); return NULL; }
    m->queue = [m->device newCommandQueue];

    NSError *err = nil;
    id<MTLLibrary> lib = [m->device newLibraryWithSource:metal_shader_source options:nil error:&err];
    if (!lib) { fprintf(stderr, "Metal compile: %s\n", [[err description] UTF8String]); free(m); return NULL; }

    m->matvec_pipe = [m->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"matvec_f16"] error:&err];
    m->silu_gate_pipe = [m->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"silu_gate_f16"] error:&err];
    m->residual_add_pipe = [m->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"residual_add_f16"] error:&err];
    m->rmsnorm_pipe = [m->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rmsnorm_f16"] error:&err];
    m->tgSize = m->matvec_pipe.maxTotalThreadsPerThreadgroup;
    if (m->tgSize > 256) m->tgSize = 256;

    int qkv_dim = cfg->dim + 2 * cfg->kv_dim;
    m->buf_x       = [m->device newBufferWithLength:cfg->dim * 2      options:MTLResourceStorageModeShared];
    m->buf_qkv     = [m->device newBufferWithLength:qkv_dim * 2       options:MTLResourceStorageModeShared];
    m->buf_attn    = [m->device newBufferWithLength:cfg->dim * 2      options:MTLResourceStorageModeShared];
    m->buf_wo      = [m->device newBufferWithLength:cfg->dim * 2      options:MTLResourceStorageModeShared];
    m->buf_ffn_in  = [m->device newBufferWithLength:cfg->dim * 2      options:MTLResourceStorageModeShared];
    m->buf_h13     = [m->device newBufferWithLength:cfg->hidden_dim*2*2 options:MTLResourceStorageModeShared];
    m->buf_gate    = [m->device newBufferWithLength:cfg->hidden_dim * 2 options:MTLResourceStorageModeShared];
    m->buf_ffn_out = [m->device newBufferWithLength:cfg->dim * 2      options:MTLResourceStorageModeShared];
    m->buf_x_res   = [m->device newBufferWithLength:cfg->dim * 2      options:MTLResourceStorageModeShared];
    m->buf_x2      = [m->device newBufferWithLength:cfg->dim * 2      options:MTLResourceStorageModeShared];

    m->layers = (MetalLayerWeights *)calloc(cfg->n_layers, sizeof(MetalLayerWeights));

    printf("Metal: %s — full inference pipeline\n", [[m->device name] UTF8String]);
    return m;
}

// Fuse Wq[dim,dim] + Wk[kv_dim,dim] + Wv[kv_dim,dim] into one [dim+2*kv_dim, dim] matrix
static id<MTLBuffer> metal_fuse_qkv(MetalInfer *m,
                                     const _Float16 *wq, const _Float16 *wk, const _Float16 *wv) {
    int D = m->dim, KD = m->kv_dim;
    int rows = D + 2 * KD;
    size_t sz = (size_t)rows * D * 2;
    _Float16 *fused = (_Float16 *)malloc(sz);
    memcpy(fused, wq, (size_t)D * D * 2);
    memcpy(fused + D * D, wk, (size_t)KD * D * 2);
    memcpy(fused + (D + KD) * D, wv, (size_t)KD * D * 2);
    id<MTLBuffer> buf = [m->device newBufferWithBytes:fused length:sz options:MTLResourceStorageModeShared];
    free(fused);
    return buf;
}

static void metal_load_layer(MetalInfer *m, int L,
                              const _Float16 *wq, const _Float16 *wk, const _Float16 *wv,
                              const _Float16 *wo, const _Float16 *rms_ffn_w,
                              const _Float16 *w1, const _Float16 *w3, const _Float16 *w2) {
    int D = m->dim, KD = m->kv_dim, H = m->hidden_dim;
    m->layers[L].Wqkv = metal_fuse_qkv(m, wq, wk, wv);
    m->layers[L].Wo = [m->device newBufferWithBytes:wo length:(size_t)D*D*2 options:MTLResourceStorageModeShared];
    m->layers[L].rms_ffn = [m->device newBufferWithBytes:rms_ffn_w length:(size_t)D*2 options:MTLResourceStorageModeShared];
    m->layers[L].W1 = [m->device newBufferWithBytes:w1 length:(size_t)H*D*2 options:MTLResourceStorageModeShared];
    m->layers[L].W3 = [m->device newBufferWithBytes:w3 length:(size_t)H*D*2 options:MTLResourceStorageModeShared];
    m->layers[L].W2 = [m->device newBufferWithBytes:w2 length:(size_t)D*H*2 options:MTLResourceStorageModeShared];
}

// ===== Eval functions =====

// RMSNorm + QKV in one command buffer (2 dispatches, 1 sync)
// x: [dim] fp16 (raw residual stream), outputs Q, K, V
static void metal_eval_rms_qkv(MetalInfer *m, int L, const _Float16 *x_fp16,
                                _Float16 *q_out, _Float16 *k_out, _Float16 *v_out) {
    int D = m->dim, KD = m->kv_dim;
    memcpy([m->buf_x contents], x_fp16, D * 2);

    @autoreleasepool {
    id<MTLCommandBuffer> cmd = [m->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    // RMSNorm(x, rms_att) → buf_ffn_in (reuse as temp normalized output)
    uint32_t dim32 = (uint32_t)D;
    [enc setComputePipelineState:m->rmsnorm_pipe];
    [enc setBuffer:m->buf_x offset:0 atIndex:0];
    [enc setBuffer:m->layers[L].rms_att offset:0 atIndex:1];
    [enc setBuffer:m->buf_ffn_in offset:0 atIndex:2];  // reuse as temp
    [enc setBytes:&dim32 length:4 atIndex:3];
    [enc setThreadgroupMemoryLength:1024 * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];

    // Fused QKV matvec
    uint32_t K = (uint32_t)D;
    [enc setComputePipelineState:m->matvec_pipe];
    [enc setBuffer:m->layers[L].Wqkv offset:0 atIndex:0];
    [enc setBuffer:m->buf_ffn_in offset:0 atIndex:1];
    [enc setBuffer:m->buf_qkv offset:0 atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(D + 2*KD, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    }

    const _Float16 *qkv = (const _Float16 *)[m->buf_qkv contents];
    memcpy(q_out, qkv, D * 2);
    memcpy(k_out, qkv + D, KD * 2);
    memcpy(v_out, qkv + D + KD, KD * 2);
}

// QKV projection: x_norm[dim] → Q[dim], K[kv_dim], V[kv_dim]
// Uses fused Wqkv matrix — one matvec dispatch
static void metal_eval_qkv(MetalInfer *m, int L, const _Float16 *x_norm,
                            _Float16 *q_out, _Float16 *k_out, _Float16 *v_out) {
    int D = m->dim, KD = m->kv_dim;
    memcpy([m->buf_x contents], x_norm, D * 2);

    @autoreleasepool {
    id<MTLCommandBuffer> cmd = [m->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    uint32_t K = (uint32_t)D;
    [enc setComputePipelineState:m->matvec_pipe];
    [enc setBuffer:m->layers[L].Wqkv offset:0 atIndex:0];
    [enc setBuffer:m->buf_x offset:0 atIndex:1];
    [enc setBuffer:m->buf_qkv offset:0 atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(D + 2*KD, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    }

    // Split fused output
    const _Float16 *qkv = (const _Float16 *)[m->buf_qkv contents];
    memcpy(q_out, qkv, D * 2);
    memcpy(k_out, qkv + D, KD * 2);
    memcpy(v_out, qkv + D + KD, KD * 2);
}

// Wo projection: attn_out[dim] → wo_delta[dim]
static void metal_eval_wo(MetalInfer *m, int L, const _Float16 *attn_out, _Float16 *wo_out) {
    int D = m->dim;
    memcpy([m->buf_attn contents], attn_out, D * 2);

    @autoreleasepool {
    id<MTLCommandBuffer> cmd = [m->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    uint32_t K = (uint32_t)D;
    [enc setComputePipelineState:m->matvec_pipe];
    [enc setBuffer:m->layers[L].Wo offset:0 atIndex:0];
    [enc setBuffer:m->buf_attn offset:0 atIndex:1];
    [enc setBuffer:m->buf_wo offset:0 atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    }

    memcpy(wo_out, [m->buf_wo contents], D * 2);
}

// FFN: x_norm[dim] → W1/W3 → silu*gate → W2 → delta[dim]
static void metal_eval_ffn(MetalInfer *m, int L, const _Float16 *x_norm, _Float16 *out) {
    int D = m->dim, H = m->hidden_dim;
    memcpy([m->buf_ffn_in contents], x_norm, D * 2);

    @autoreleasepool {
    id<MTLCommandBuffer> cmd = [m->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    uint32_t K;

    // W1 @ x → h1
    K = (uint32_t)D;
    [enc setComputePipelineState:m->matvec_pipe];
    [enc setBuffer:m->layers[L].W1 offset:0 atIndex:0];
    [enc setBuffer:m->buf_ffn_in offset:0 atIndex:1];
    [enc setBuffer:m->buf_h13 offset:0 atIndex:2];  // h1 in first half
    [enc setBytes:&K length:4 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(H, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    // W3 @ x → h3 (offset into second half of buf_h13)
    [enc setBuffer:m->layers[L].W3 offset:0 atIndex:0];
    [enc setBuffer:m->buf_h13 offset:H*2 atIndex:2];  // h3 in second half
    [enc dispatchThreadgroups:MTLSizeMake(H, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    // silu(h1) * h3 → gate
    [enc setComputePipelineState:m->silu_gate_pipe];
    [enc setBuffer:m->buf_h13 offset:0 atIndex:0];    // h1
    [enc setBuffer:m->buf_h13 offset:H*2 atIndex:1];  // h3
    [enc setBuffer:m->buf_gate offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(H, 1, 1) threadsPerThreadgroup:MTLSizeMake(m->tgSize, 1, 1)];

    // W2 @ gate → out
    K = (uint32_t)H;
    [enc setComputePipelineState:m->matvec_pipe];
    [enc setBuffer:m->layers[L].W2 offset:0 atIndex:0];
    [enc setBuffer:m->buf_gate offset:0 atIndex:1];
    [enc setBuffer:m->buf_ffn_out offset:0 atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    }

    memcpy(out, [m->buf_ffn_out contents], D * 2);
}

// Full layer: QKV + Wo + FFN in one command buffer submission
// Requires attention to be done on CPU between QKV and Wo
// So we split into: qkv_eval → [cpu attention] → wo_ffn_eval
static void metal_eval_wo_ffn(MetalInfer *m, int L,
                               const _Float16 *attn_out,    // [dim] from CPU attention
                               const _Float16 *ffn_x_norm,  // [dim] RMSNorm'd for FFN
                               _Float16 *wo_out,             // [dim] Wo projection delta
                               _Float16 *ffn_out) {          // [dim] FFN delta
    int D = m->dim, H = m->hidden_dim;
    memcpy([m->buf_attn contents], attn_out, D * 2);
    memcpy([m->buf_ffn_in contents], ffn_x_norm, D * 2);

    @autoreleasepool {
    id<MTLCommandBuffer> cmd = [m->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    uint32_t K;

    // Wo @ attn_out → wo_delta
    K = (uint32_t)D;
    [enc setComputePipelineState:m->matvec_pipe];
    [enc setBuffer:m->layers[L].Wo offset:0 atIndex:0];
    [enc setBuffer:m->buf_attn offset:0 atIndex:1];
    [enc setBuffer:m->buf_wo offset:0 atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    // W1 @ ffn_x_norm → h1
    [enc setBuffer:m->layers[L].W1 offset:0 atIndex:0];
    [enc setBuffer:m->buf_ffn_in offset:0 atIndex:1];
    [enc setBuffer:m->buf_h13 offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(H, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    // W3 @ ffn_x_norm → h3
    [enc setBuffer:m->layers[L].W3 offset:0 atIndex:0];
    [enc setBuffer:m->buf_h13 offset:H*2 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(H, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    // silu(h1) * h3 → gate
    [enc setComputePipelineState:m->silu_gate_pipe];
    [enc setBuffer:m->buf_h13 offset:0 atIndex:0];
    [enc setBuffer:m->buf_h13 offset:H*2 atIndex:1];
    [enc setBuffer:m->buf_gate offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(H, 1, 1) threadsPerThreadgroup:MTLSizeMake(m->tgSize, 1, 1)];

    // W2 @ gate → ffn_out
    K = (uint32_t)H;
    [enc setComputePipelineState:m->matvec_pipe];
    [enc setBuffer:m->layers[L].W2 offset:0 atIndex:0];
    [enc setBuffer:m->buf_gate offset:0 atIndex:1];
    [enc setBuffer:m->buf_ffn_out offset:0 atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    }

    memcpy(wo_out, [m->buf_wo contents], D * 2);
    memcpy(ffn_out, [m->buf_ffn_out contents], D * 2);
}

// Async command buffer for pipelining with ANE
static id<MTLCommandBuffer> g_pending_cmd = nil;

// Launch Wo+RMS+FFN async (no wait). Call metal_sync_wo_rms_ffn to wait and read results.
static void metal_launch_wo_rms_ffn(MetalInfer *m, int L,
                                     const _Float16 *x_res, const _Float16 *attn_out) {
    int D = m->dim, H = m->hidden_dim;
    memcpy([m->buf_x_res contents], x_res, D * 2);
    memcpy([m->buf_attn contents], attn_out, D * 2);

    @autoreleasepool {
    id<MTLCommandBuffer> cmd = [m->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    uint32_t K;

    // 1. Wo
    K = (uint32_t)D;
    [enc setComputePipelineState:m->matvec_pipe];
    [enc setBuffer:m->layers[L].Wo offset:0 atIndex:0];
    [enc setBuffer:m->buf_attn offset:0 atIndex:1];
    [enc setBuffer:m->buf_wo offset:0 atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    // 2. x2 = x_res + wo
    [enc setComputePipelineState:m->residual_add_pipe];
    [enc setBuffer:m->buf_x_res offset:0 atIndex:0];
    [enc setBuffer:m->buf_wo offset:0 atIndex:1];
    [enc setBuffer:m->buf_x2 offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    // 3. RMSNorm
    uint32_t dim32 = (uint32_t)D;
    [enc setComputePipelineState:m->rmsnorm_pipe];
    [enc setBuffer:m->buf_x2 offset:0 atIndex:0];
    [enc setBuffer:m->layers[L].rms_ffn offset:0 atIndex:1];
    [enc setBuffer:m->buf_ffn_in offset:0 atIndex:2];
    [enc setBytes:&dim32 length:4 atIndex:3];
    [enc setThreadgroupMemoryLength:1024 * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];

    // 4-7. FFN
    K = (uint32_t)D;
    [enc setComputePipelineState:m->matvec_pipe];
    [enc setBuffer:m->layers[L].W1 offset:0 atIndex:0];
    [enc setBuffer:m->buf_ffn_in offset:0 atIndex:1];
    [enc setBuffer:m->buf_h13 offset:0 atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(H, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    [enc setBuffer:m->layers[L].W3 offset:0 atIndex:0];
    [enc setBuffer:m->buf_h13 offset:H*2 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(H, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    [enc setComputePipelineState:m->silu_gate_pipe];
    [enc setBuffer:m->buf_h13 offset:0 atIndex:0];
    [enc setBuffer:m->buf_h13 offset:H*2 atIndex:1];
    [enc setBuffer:m->buf_gate offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(H, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    K = (uint32_t)H;
    [enc setComputePipelineState:m->matvec_pipe];
    [enc setBuffer:m->layers[L].W2 offset:0 atIndex:0];
    [enc setBuffer:m->buf_gate offset:0 atIndex:1];
    [enc setBuffer:m->buf_ffn_out offset:0 atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    [enc endEncoding];
    [cmd commit];
    g_pending_cmd = cmd;  // don't wait yet!
    }
}

// Wait for async Metal and read results
static void metal_sync_wo_rms_ffn(MetalInfer *m, _Float16 *wo_out, _Float16 *ffn_out) {
    if (g_pending_cmd) {
        [g_pending_cmd waitUntilCompleted];
        g_pending_cmd = nil;
    }
    int D = m->dim;
    memcpy(wo_out, [m->buf_wo contents], D * 2);
    memcpy(ffn_out, [m->buf_ffn_out contents], D * 2);
}

// Synchronous version (for compatibility)
// Fused Wo + residual + RMSNorm + FFN in ONE command buffer (1 GPU sync)
static void metal_eval_wo_rms_ffn(MetalInfer *m, int L,
                                   const _Float16 *x_res, const _Float16 *attn_out,
                                   _Float16 *wo_out, _Float16 *ffn_out) {
    int D = m->dim, H = m->hidden_dim;
    memcpy([m->buf_x_res contents], x_res, D * 2);
    memcpy([m->buf_attn contents], attn_out, D * 2);

    @autoreleasepool {
    id<MTLCommandBuffer> cmd = [m->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    uint32_t K;

    // 1. Wo @ attn_out → buf_wo [dim]
    K = (uint32_t)D;
    [enc setComputePipelineState:m->matvec_pipe];
    [enc setBuffer:m->layers[L].Wo offset:0 atIndex:0];
    [enc setBuffer:m->buf_attn offset:0 atIndex:1];
    [enc setBuffer:m->buf_wo offset:0 atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    // 2. x2 = x_res + wo_delta [dim]
    [enc setComputePipelineState:m->residual_add_pipe];
    [enc setBuffer:m->buf_x_res offset:0 atIndex:0];
    [enc setBuffer:m->buf_wo offset:0 atIndex:1];
    [enc setBuffer:m->buf_x2 offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    // 3. RMSNorm(x2, rms_ffn) → buf_ffn_in [dim]
    uint32_t dim32 = (uint32_t)D;
    [enc setComputePipelineState:m->rmsnorm_pipe];
    [enc setBuffer:m->buf_x2 offset:0 atIndex:0];
    [enc setBuffer:m->layers[L].rms_ffn offset:0 atIndex:1];
    [enc setBuffer:m->buf_ffn_in offset:0 atIndex:2];
    [enc setBytes:&dim32 length:4 atIndex:3];
    [enc setThreadgroupMemoryLength:1024 * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];

    // 4. W1 @ x_norm → h1 [hidden]
    K = (uint32_t)D;
    [enc setComputePipelineState:m->matvec_pipe];
    [enc setBuffer:m->layers[L].W1 offset:0 atIndex:0];
    [enc setBuffer:m->buf_ffn_in offset:0 atIndex:1];
    [enc setBuffer:m->buf_h13 offset:0 atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(H, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    // 5. W3 @ x_norm → h3 [hidden]
    [enc setBuffer:m->layers[L].W3 offset:0 atIndex:0];
    [enc setBuffer:m->buf_h13 offset:H*2 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(H, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    // 6. silu(h1) * h3 → gate [hidden]
    [enc setComputePipelineState:m->silu_gate_pipe];
    [enc setBuffer:m->buf_h13 offset:0 atIndex:0];
    [enc setBuffer:m->buf_h13 offset:H*2 atIndex:1];
    [enc setBuffer:m->buf_gate offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(H, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    // 7. W2 @ gate → ffn_out [dim]
    K = (uint32_t)H;
    [enc setComputePipelineState:m->matvec_pipe];
    [enc setBuffer:m->layers[L].W2 offset:0 atIndex:0];
    [enc setBuffer:m->buf_gate offset:0 atIndex:1];
    [enc setBuffer:m->buf_ffn_out offset:0 atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    }

    memcpy(wo_out, [m->buf_wo contents], D * 2);
    memcpy(ffn_out, [m->buf_ffn_out contents], D * 2);
}

// Load classifier weights to Metal
static void metal_load_classifier(MetalInfer *m, const _Float16 *w, int vocab, int dim) {
    size_t sz = (size_t)vocab * dim * 2;
    m->buf_cls_w = [m->device newBufferWithBytes:w length:sz options:MTLResourceStorageModeShared];
    m->buf_cls_out = [m->device newBufferWithLength:vocab * 2 options:MTLResourceStorageModeShared];
}

// Metal classifier: output_w[vocab, dim] @ x_norm[dim] → logits[vocab] fp32
static void metal_classifier(MetalInfer *m, const _Float16 *x_norm, float *logits) {
    int V = m->vocab_size, D = m->dim;
    memcpy([m->buf_x contents], x_norm, D * 2);

    @autoreleasepool {
    id<MTLCommandBuffer> cmd = [m->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    uint32_t K = (uint32_t)D;
    [enc setComputePipelineState:m->matvec_pipe];
    [enc setBuffer:m->buf_cls_w offset:0 atIndex:0];
    [enc setBuffer:m->buf_x offset:0 atIndex:1];
    [enc setBuffer:m->buf_cls_out offset:0 atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(V, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    }

    // Convert fp16 → fp32
    const _Float16 *cls_f16 = (const _Float16 *)[m->buf_cls_out contents];
    for (int i = 0; i < V; i++) logits[i] = (float)cls_f16[i];
}

static void metal_free(MetalInfer *m) {
    if (!m) return;
    free(m->layers);
    free(m);
}
