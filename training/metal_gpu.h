// metal_gpu.h — GPU RoPE via zero-copy IOSurface→MTLBuffer
// Eliminates fp16→fp32→rotate→fp32→fp16 round-trips by operating directly on fp16 IOSurface data
#pragma once
#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>
#include "stories_config.h"

#define METAL_SEQ_STR "___SEQ___"
#define METAL_HD_STR  "___HD___"

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> rope_fwd;
    id<MTLComputePipelineState> rope_bwd;
    id<MTLBuffer> cos_sin_table;  // precomputed [HD/2 * SEQ * 2] fp16
} MetalCtx;

static MetalCtx g_metal;

// RoPE forward shader: (cos, -sin; sin, cos) rotation on Q and K simultaneously
// Channel-first layout: element [h*HD+i, t] at offset (h*HD+i)*SEQ + t
// Thread grid: HEADS * (HD/2) * SEQ = 12 * 32 * 256 = 98304 threads
static NSString *metal_rope_fwd_src = @
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void rope_fwd(\n"
    "    device const half* src [[buffer(0)]],\n"
    "    device half* dst       [[buffer(1)]],\n"
    "    device const half* tbl [[buffer(2)]],\n"
    "    constant uint& q_src_off  [[buffer(3)]],\n"
    "    constant uint& k_src_off  [[buffer(4)]],\n"
    "    constant uint& q_dst_off  [[buffer(5)]],\n"
    "    constant uint& k_dst_off  [[buffer(6)]],\n"
    "    uint tid [[thread_position_in_grid]])\n"
    "{\n"
    "    const uint SEQ = " @METAL_SEQ_STR ";\n"
    "    const uint HD  = " @METAL_HD_STR ";\n"
    "    const uint HALF_HD = HD / 2;\n"
    "    // tid = h * HALF_HD * SEQ + j * SEQ + t\n"
    "    uint t = tid % SEQ;\n"
    "    uint j = (tid / SEQ) % HALF_HD;\n"
    "    uint h = tid / (HALF_HD * SEQ);\n"
    "    // Source/dest row indices (channel-first)\n"
    "    uint row0 = (h * HD + 2 * j) * SEQ + t;\n"
    "    uint row1 = (h * HD + 2 * j + 1) * SEQ + t;\n"
    "    // Cos/sin lookup (same for all heads)\n"
    "    uint tbl_idx = (j * SEQ + t) * 2;\n"
    "    half cos_v = tbl[tbl_idx];\n"
    "    half sin_v = tbl[tbl_idx + 1];\n"
    "    // Q rotation\n"
    "    half q0 = src[q_src_off + row0];\n"
    "    half q1 = src[q_src_off + row1];\n"
    "    dst[q_dst_off + row0] = q0 * cos_v - q1 * sin_v;\n"
    "    dst[q_dst_off + row1] = q0 * sin_v + q1 * cos_v;\n"
    "    // K rotation\n"
    "    half k0 = src[k_src_off + row0];\n"
    "    half k1 = src[k_src_off + row1];\n"
    "    dst[k_dst_off + row0] = k0 * cos_v - k1 * sin_v;\n"
    "    dst[k_dst_off + row1] = k0 * sin_v + k1 * cos_v;\n"
    "}\n";

// RoPE backward shader: inverse rotation (cos, +sin; -sin, cos)
static NSString *metal_rope_bwd_src = @
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void rope_bwd(\n"
    "    device const half* src [[buffer(0)]],\n"
    "    device half* dst       [[buffer(1)]],\n"
    "    device const half* tbl [[buffer(2)]],\n"
    "    constant uint& q_src_off  [[buffer(3)]],\n"
    "    constant uint& k_src_off  [[buffer(4)]],\n"
    "    constant uint& q_dst_off  [[buffer(5)]],\n"
    "    constant uint& k_dst_off  [[buffer(6)]],\n"
    "    uint tid [[thread_position_in_grid]])\n"
    "{\n"
    "    const uint SEQ = " @METAL_SEQ_STR ";\n"
    "    const uint HD  = " @METAL_HD_STR ";\n"
    "    const uint HALF_HD = HD / 2;\n"
    "    uint t = tid % SEQ;\n"
    "    uint j = (tid / SEQ) % HALF_HD;\n"
    "    uint h = tid / (HALF_HD * SEQ);\n"
    "    uint row0 = (h * HD + 2 * j) * SEQ + t;\n"
    "    uint row1 = (h * HD + 2 * j + 1) * SEQ + t;\n"
    "    uint tbl_idx = (j * SEQ + t) * 2;\n"
    "    half cos_v = tbl[tbl_idx];\n"
    "    half sin_v = tbl[tbl_idx + 1];\n"
    "    // Inverse Q rotation\n"
    "    half dq0 = src[q_src_off + row0];\n"
    "    half dq1 = src[q_src_off + row1];\n"
    "    dst[q_dst_off + row0] =  dq0 * cos_v + dq1 * sin_v;\n"
    "    dst[q_dst_off + row1] = -dq0 * sin_v + dq1 * cos_v;\n"
    "    // Inverse K rotation\n"
    "    half dk0 = src[k_src_off + row0];\n"
    "    half dk1 = src[k_src_off + row1];\n"
    "    dst[k_dst_off + row0] =  dk0 * cos_v + dk1 * sin_v;\n"
    "    dst[k_dst_off + row1] = -dk0 * sin_v + dk1 * cos_v;\n"
    "}\n";

// Substitute compile-time constants into shader source
static NSString *metal_rope_substitute(NSString *src) {
    NSString *s = [src stringByReplacingOccurrencesOfString:@METAL_SEQ_STR
                       withString:[NSString stringWithFormat:@"%d", SEQ]];
    return [s stringByReplacingOccurrencesOfString:@METAL_HD_STR
                withString:[NSString stringWithFormat:@"%d", HD]];
}

static bool metal_init(void) {
    g_metal.device = MTLCreateSystemDefaultDevice();
    if (!g_metal.device) { printf("  [metal] No Metal device\n"); return false; }
    g_metal.queue = [g_metal.device newCommandQueue];

    // Compile forward RoPE shader
    NSError *err = nil;
    NSString *fwd_src = metal_rope_substitute(metal_rope_fwd_src);
    id<MTLLibrary> fwd_lib = [g_metal.device newLibraryWithSource:fwd_src options:nil error:&err];
    if (!fwd_lib) { printf("  [metal] RoPE fwd compile: %s\n", [[err localizedDescription] UTF8String]); return false; }
    id<MTLFunction> fwd_fn = [fwd_lib newFunctionWithName:@"rope_fwd"];
    g_metal.rope_fwd = [g_metal.device newComputePipelineStateWithFunction:fwd_fn error:&err];
    if (!g_metal.rope_fwd) { printf("  [metal] RoPE fwd pipeline: %s\n", [[err localizedDescription] UTF8String]); return false; }

    // Compile backward RoPE shader
    NSString *bwd_src = metal_rope_substitute(metal_rope_bwd_src);
    id<MTLLibrary> bwd_lib = [g_metal.device newLibraryWithSource:bwd_src options:nil error:&err];
    if (!bwd_lib) { printf("  [metal] RoPE bwd compile: %s\n", [[err localizedDescription] UTF8String]); return false; }
    id<MTLFunction> bwd_fn = [bwd_lib newFunctionWithName:@"rope_bwd"];
    g_metal.rope_bwd = [g_metal.device newComputePipelineStateWithFunction:bwd_fn error:&err];
    if (!g_metal.rope_bwd) { printf("  [metal] RoPE bwd pipeline: %s\n", [[err localizedDescription] UTF8String]); return false; }

    // Precompute cos/sin table as fp16: [HD/2][SEQ][2]
    // Size: 32 * 256 * 2 * 2 = 32768 bytes = 32 KB
    int half_hd = HD / 2;
    int tbl_elems = half_hd * SEQ * 2;
    _Float16 *tbl = (_Float16*)malloc(tbl_elems * sizeof(_Float16));
    for (int j = 0; j < half_hd; j++) {
        float freq = 1.0f / powf(10000.0f, (float)(2 * j) / HD);
        for (int t = 0; t < SEQ; t++) {
            float angle = t * freq;
            tbl[(j * SEQ + t) * 2]     = (_Float16)cosf(angle);
            tbl[(j * SEQ + t) * 2 + 1] = (_Float16)sinf(angle);
        }
    }
    g_metal.cos_sin_table = [g_metal.device newBufferWithBytes:tbl
                                                        length:tbl_elems * sizeof(_Float16)
                                                       options:MTLResourceStorageModeShared];
    free(tbl);

    printf("  [metal] GPU RoPE initialized: %s, table=%d bytes\n",
           [[g_metal.device name] UTF8String], tbl_elems * 2);
    return true;
}

// Dispatch RoPE forward: src IOSurface[Q,K] → dst IOSurface[Q_rope,K_rope]
// q_src_off, k_src_off: fp16 element offsets into src for Q and K channels
// q_dst_off, k_dst_off: fp16 element offsets into dst for Q_rope and K_rope channels
static void metal_rope_fwd(IOSurfaceRef src, IOSurfaceRef dst,
                           uint32_t q_src_off, uint32_t k_src_off,
                           uint32_t q_dst_off, uint32_t k_dst_off) {
    void *src_addr = IOSurfaceGetBaseAddress(src);
    void *dst_addr = IOSurfaceGetBaseAddress(dst);
    size_t src_len = IOSurfaceGetAllocSize(src);
    size_t dst_len = IOSurfaceGetAllocSize(dst);

    id<MTLBuffer> src_buf = [g_metal.device newBufferWithBytesNoCopy:src_addr length:src_len
                                                             options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> dst_buf = (src == dst) ? src_buf :
        [g_metal.device newBufferWithBytesNoCopy:dst_addr length:dst_len
                                         options:MTLResourceStorageModeShared deallocator:nil];

    id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_metal.rope_fwd];
    [enc setBuffer:src_buf offset:0 atIndex:0];
    [enc setBuffer:dst_buf offset:0 atIndex:1];
    [enc setBuffer:g_metal.cos_sin_table offset:0 atIndex:2];
    [enc setBytes:&q_src_off length:4 atIndex:3];
    [enc setBytes:&k_src_off length:4 atIndex:4];
    [enc setBytes:&q_dst_off length:4 atIndex:5];
    [enc setBytes:&k_dst_off length:4 atIndex:6];

    NSUInteger total = HEADS * (HD / 2) * SEQ;  // 12 * 32 * 256 = 98304
    NSUInteger tpg = MIN(g_metal.rope_fwd.maxTotalThreadsPerThreadgroup, 256);
    [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

// Dispatch RoPE backward (inverse rotation)
static void metal_rope_bwd(IOSurfaceRef src, IOSurfaceRef dst,
                           uint32_t q_src_off, uint32_t k_src_off,
                           uint32_t q_dst_off, uint32_t k_dst_off) {
    void *src_addr = IOSurfaceGetBaseAddress(src);
    void *dst_addr = IOSurfaceGetBaseAddress(dst);
    size_t src_len = IOSurfaceGetAllocSize(src);
    size_t dst_len = IOSurfaceGetAllocSize(dst);

    id<MTLBuffer> src_buf = [g_metal.device newBufferWithBytesNoCopy:src_addr length:src_len
                                                             options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> dst_buf = (src == dst) ? src_buf :
        [g_metal.device newBufferWithBytesNoCopy:dst_addr length:dst_len
                                         options:MTLResourceStorageModeShared deallocator:nil];

    id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_metal.rope_bwd];
    [enc setBuffer:src_buf offset:0 atIndex:0];
    [enc setBuffer:dst_buf offset:0 atIndex:1];
    [enc setBuffer:g_metal.cos_sin_table offset:0 atIndex:2];
    [enc setBytes:&q_src_off length:4 atIndex:3];
    [enc setBytes:&k_src_off length:4 atIndex:4];
    [enc setBytes:&q_dst_off length:4 atIndex:5];
    [enc setBytes:&k_dst_off length:4 atIndex:6];

    NSUInteger total = HEADS * (HD / 2) * SEQ;
    NSUInteger tpg = MIN(g_metal.rope_bwd.maxTotalThreadsPerThreadgroup, 256);
    [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}
