// infer_mil.h — MIL kernel generators for inference (parameterized, GQA-aware)
// Generates MIL text for ANE compilation. All kernels use channel-first [1, C, 1, S] layout.
#pragma once
#import <Foundation/Foundation.h>
#include "infer_config.h"
#include "stories_io.h"

// Shared MIL header and conv constants (same as stories_mil.h)
#define INFER_MIL_HDR \
    @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"
#define INFER_CONV_CONST \
    "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n" \
    "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n" \
    "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"

// ========== QKV Projection ==========
// Input:  [1, dim, 1, S] — x
// Output: [1, dim + 2*kv_dim, 1, S] — concat(Q, K, V)
// Weights: rms_att[dim], Wq[dim,dim], Wk[kv_dim,dim], Wv[kv_dim,dim]
// Blob layout: rms1.bin, wq.bin, wk.bin, wv.bin
static NSString *gen_infer_qkv(const InferConfig *c, int S) {
    float invd = 1.0f / (float)c->dim;
    NSMutableString *m = [NSMutableString string];
    [m appendString:INFER_MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", c->dim, S];
    // RMSNorm
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x)[name=string(\"sq\")];\n", c->dim, S];
    [m appendFormat:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendFormat:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string(\"ss\")];\n", S];
    [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];\n", S];
    [m appendFormat:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];\n", S];
    [m appendFormat:@"        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];\n", S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];\n", c->dim, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms1.bin\"), offset=uint64(64)))];\n", c->dim, c->dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];\n", c->dim, S];
    // QKV convolutions
    [m appendString:@INFER_CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];\n", c->dim, c->dim, c->dim, c->dim];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];\n", c->kv_dim, c->dim, c->kv_dim, c->dim];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];\n", c->kv_dim, c->dim, c->kv_dim, c->dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=xn)[name=string(\"cq\")];\n", c->dim, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wk,x=xn)[name=string(\"ck\")];\n", c->kv_dim, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wv,x=xn)[name=string(\"cv\")];\n", c->kv_dim, S];
    // Concat Q, K, V
    int out_ch = c->dim + 2 * c->kv_dim;
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(qf,kf,vf))[name=string(\"cat\")];\n", out_ch, S];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// ========== Attention Prefill (full causal) ==========
// Input: [1, dim + 2*kv_dim + dim, 1, S] — concat(Q_rope, K_rope, V, x_residual)
// Output: [1, dim, 1, S] — x2 (after Wo + residual)
// GQA: tile K,V from n_kv_heads to n_heads via reshape+tile
static NSString *gen_infer_attn_prefill(const InferConfig *c, int S) {
    float sc = 1.0f / sqrtf((float)c->head_dim);
    int in_ch = c->dim + 2 * c->kv_dim + c->dim;
    NSMutableString *m = [NSMutableString string];
    [m appendString:INFER_MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", in_ch, S];

    // Slice Q [dim], K [kv_dim], V [kv_dim], x_residual [dim]
    [m appendFormat:@"        tensor<int32, [4]> szq = const()[name=string(\"szq\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", c->dim, S];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = slice_by_size(x=x,begin=b0,size=szq)[name=string(\"sq\")];\n", c->dim, S];

    [m appendFormat:@"        tensor<int32, [4]> szk = const()[name=string(\"szk\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", c->kv_dim, S];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", c->dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = slice_by_size(x=x,begin=b1,size=szk)[name=string(\"sk\")];\n", c->kv_dim, S];

    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", c->dim + c->kv_dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = slice_by_size(x=x,begin=b2,size=szk)[name=string(\"sv\")];\n", c->kv_dim, S];

    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", c->dim + 2 * c->kv_dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xr = slice_by_size(x=x,begin=b3,size=szq)[name=string(\"sxr\")];\n", c->dim, S];

    // Reshape Q to [1, n_heads, S, head_dim]
    [m appendFormat:@"        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", c->n_heads, c->head_dim, S];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q4 = reshape(shape=qsh,x=qf)[name=string(\"rq\")];\n", c->n_heads, c->head_dim, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=q4)[name=string(\"tq\")];\n", c->n_heads, S, c->head_dim];

    // Reshape K,V to [1, n_kv_heads, S, head_dim], then tile to [1, n_heads, S, head_dim]
    [m appendFormat:@"        tensor<int32, [4]> kvsh = const()[name=string(\"kvsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", c->n_kv_heads, c->head_dim, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k4 = reshape(shape=kvsh,x=kf)[name=string(\"rk\")];\n", c->n_kv_heads, c->head_dim, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> kt = transpose(perm=pm,x=k4)[name=string(\"tk\")];\n", c->n_kv_heads, S, c->head_dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v4 = reshape(shape=kvsh,x=vf)[name=string(\"rv\")];\n", c->n_kv_heads, c->head_dim, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> vt = transpose(perm=pm,x=v4)[name=string(\"tv\")];\n", c->n_kv_heads, S, c->head_dim];

    if (c->kv_group > 1) {
        // GQA: tile KV heads to match Q heads
        // tile along head dimension: [1, n_kv_heads, S, HD] → [1, n_heads, S, HD]
        // MIL tile op: reps=[1, kv_group, 1, 1]
        [m appendFormat:@"        tensor<int32, [4]> treps = const()[name=string(\"treps\"), val=tensor<int32, [4]>([1,%d,1,1])];\n", c->kv_group];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = tile(reps=treps,x=kt)[name=string(\"tilk\")];\n", c->n_heads, S, c->head_dim];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = tile(reps=treps,x=vt)[name=string(\"tilv\")];\n", c->n_heads, S, c->head_dim];
    } else {
        // No GQA: rename
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = identity(x=kt)[name=string(\"idk\")];\n", c->n_heads, S, c->head_dim];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = identity(x=vt)[name=string(\"idv\")];\n", c->n_heads, S, c->head_dim];
    }

    // Q @ K^T, scale, mask, softmax, @ V
    [m appendString:@"        bool tx = const()[name=string(\"tx\"), val=bool(false)];\n"];
    [m appendString:@"        bool ty = const()[name=string(\"ty\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=tx,transpose_y=ty,x=q,y=k)[name=string(\"mm1\")];\n", c->n_heads, S, S];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];\n", c->n_heads, S, S];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];\n", S, S, S, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];\n", c->n_heads, S, S];
    [m appendString:@"        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> aw = softmax(axis=sax,x=ms)[name=string(\"sm\")];\n", c->n_heads, S, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> a4 = matmul(transpose_x=tx,transpose_y=tx,x=aw,y=v)[name=string(\"mm2\")];\n", c->n_heads, S, c->head_dim];

    // Transpose back, reshape, Wo conv, residual
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> at = transpose(perm=pm,x=a4)[name=string(\"ta\")];\n", c->n_heads, c->head_dim, S];
    [m appendFormat:@"        tensor<int32, [4]> os = const()[name=string(\"os\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", c->dim, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> af = reshape(shape=os,x=at)[name=string(\"ra\")];\n", c->dim, S];
    [m appendString:@INFER_CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];\n", c->dim, c->dim, c->dim, c->dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> oo = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wo,x=af)[name=string(\"co\")];\n", c->dim, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = add(x=xr,y=oo)[name=string(\"res\")];\n", c->dim, S];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// ========== Attention Decode (single-token, cached KV) ==========
// Input: [1, dim + 2*kv_dim*max_seq + dim + max_seq, 1, 1] — too large for IOSurface channels
// INSTEAD: use separate IOSurfaces. We'll do attention on CPU for decode.
// Actually — let's do a smarter approach:
// Input:  [1, dim + dim, 1, 1] — concat(Q_rope[dim,1], x_residual[dim,1])
// KV cache: passed as a second IOSurface [1, 2*kv_dim, 1, max_seq] — K,V stacked
// BUT: ANE only supports 1 input IOSurface per kernel.
//
// Strategy: Run attention decode on CPU with Accelerate (it's just one vector query)
// This is actually fast: Q[1,dim] @ K_cache[dim,T]^T = [1,T] scores — one BLAS call
// Then softmax, then scores[1,T] @ V_cache[T,dim] — another BLAS call
// For T=2048, dim=4096, this is ~2 matmul calls, ~0.1ms each on M-series NEON

// ========== FFN (inference-only, no backward taps) ==========
// Input:  [1, dim, 1, S]
// Output: [1, dim, 1, S] — x + W2(silu(W1(xn)) * W3(xn))
// Weights: rms2[dim], W1[hidden,dim], W3[hidden,dim], W2[dim,hidden]
static NSString *gen_infer_ffn(const InferConfig *c, int S) {
    float invd = 1.0f / (float)c->dim;
    NSMutableString *m = [NSMutableString string];
    [m appendString:INFER_MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", c->dim, S];
    // RMSNorm
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x)[name=string(\"sq\")];\n", c->dim, S];
    [m appendFormat:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendFormat:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string(\"ss\")];\n", S];
    [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];\n", S];
    [m appendFormat:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];\n", S];
    [m appendFormat:@"        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];\n", S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];\n", c->dim, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms2.bin\"), offset=uint64(64)))];\n", c->dim, c->dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];\n", c->dim, S];
    // FFN: W1, W3 (gate), silu, W2
    [m appendString:@INFER_CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W1 = const()[name=string(\"W1\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];\n", c->hidden_dim, c->dim, c->hidden_dim, c->dim];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W3 = const()[name=string(\"W3\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3.bin\"), offset=uint64(64)))];\n", c->hidden_dim, c->dim, c->hidden_dim, c->dim];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W2 = const()[name=string(\"W2\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];\n", c->dim, c->hidden_dim, c->dim, c->hidden_dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=xn)[name=string(\"c1\")];\n", c->hidden_dim, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3,x=xn)[name=string(\"c3\")];\n", c->hidden_dim, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1)[name=string(\"sg\")];\n", c->hidden_dim, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> silu = mul(x=h1,y=sig)[name=string(\"si\")];\n", c->hidden_dim, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> gate = mul(x=silu,y=h3)[name=string(\"gt\")];\n", c->hidden_dim, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2,x=gate)[name=string(\"c2\")];\n", c->dim, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = add(x=x,y=y)[name=string(\"res\")];\n", c->dim, S];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// ========== Final RMSNorm ==========
// Input:  [1, dim, 1, S]
// Output: [1, dim, 1, S]
static NSString *gen_infer_rms_final(const InferConfig *c, int S) {
    float invd = 1.0f / (float)c->dim;
    NSMutableString *m = [NSMutableString string];
    [m appendString:INFER_MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", c->dim, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x)[name=string(\"sq\")];\n", c->dim, S];
    [m appendFormat:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendFormat:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string(\"ss\")];\n", S];
    [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];\n", S];
    [m appendFormat:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];\n", S];
    [m appendFormat:@"        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];\n", S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];\n", c->dim, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_final.bin\"), offset=uint64(64)))];\n", c->dim, c->dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = mul(x=xr,y=rw)[name=string(\"out\")];\n", c->dim, S];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// ========== Causal mask builder ==========
// Returns fp16 causal mask [S, S] for baking into attention kernel
static NSData *build_causal_mask(int S) {
    _Float16 *mask = (_Float16 *)malloc(S * S * sizeof(_Float16));
    for (int t1 = 0; t1 < S; t1++)
        for (int t2 = 0; t2 < S; t2++)
            mask[t1 * S + t2] = (t2 <= t1) ? (_Float16)0.0f : (_Float16)-65504.0f;
    NSData *blob = build_blob_fp16(mask, S * S);
    free(mask);
    return blob;
}

// ========== CPU Attention Decode ==========
// Single-query attention against cached KV. Uses Accelerate BLAS.
// q: [dim] fp16 (post-RoPE)
// kv: KV cache layer (channel-first [kv_dim, max_seq])
// out: [dim] fp16
// scratch: caller-provided fp32 buffers
typedef struct {
    float *scores;      // [n_heads, T] attention scores
    float *q_f32;       // [dim] Q in fp32
    float *attn_f32;    // [dim] attention output in fp32
    float *out_f32;     // [dim] after Wo projection
    _Float16 *wo_f16;   // [dim, dim] Wo weights (persisted)
} AttnDecodeScratch;

static AttnDecodeScratch *attn_decode_scratch_alloc(const InferConfig *c) {
    AttnDecodeScratch *s = (AttnDecodeScratch *)calloc(1, sizeof(AttnDecodeScratch));
    s->scores = (float *)malloc(c->n_heads * c->max_seq * sizeof(float));
    s->q_f32 = (float *)malloc(c->dim * sizeof(float));
    s->attn_f32 = (float *)malloc(c->dim * sizeof(float));
    s->out_f32 = (float *)malloc(c->dim * sizeof(float));
    return s;
}

static void attn_decode_scratch_free(AttnDecodeScratch *s) {
    if (!s) return;
    free(s->scores); free(s->q_f32); free(s->attn_f32); free(s->out_f32);
    free(s);
}

// CPU decode attention for a single query position
// q_rope: [dim] fp16 (post-RoPE Q for this position)
// k_cache, v_cache: [kv_dim, max_seq] fp16 channel-first
// wo: [dim, dim] fp16 weights
// x_residual: [dim] fp16
// out: [dim] fp16 (x2 = x_residual + Wo @ attn)
// T: number of valid cached positions
static void cpu_attn_decode(const InferConfig *c, const _Float16 *q_rope,
                            const _Float16 *k_cache, const _Float16 *v_cache,
                            const _Float16 *wo, const _Float16 *x_residual,
                            _Float16 *out, int T, AttnDecodeScratch *scratch) {
    int nH = c->n_heads, nKVH = c->n_kv_heads, hd = c->head_dim, gqa = c->kv_group;
    float scale = 1.0f / sqrtf((float)hd);

    // For each head: scores[t] = sum_d(q[d] * k[d,t]) * scale
    // Then softmax, then attn[d] = sum_t(scores[t] * v[d,t])
    for (int h = 0; h < nH; h++) {
        int kv_h = h / gqa;  // GQA: map Q head to KV head
        float *sc = scratch->scores + h * T;

        // Compute attention scores for this head
        float max_score = -1e30f;
        for (int t = 0; t < T; t++) {
            float dot = 0;
            for (int d = 0; d < hd; d++) {
                int q_idx = h * hd + d;
                int k_idx = (kv_h * hd + d) * c->max_seq + t;
                dot += (float)q_rope[q_idx] * (float)k_cache[k_idx];
            }
            sc[t] = dot * scale;
            if (sc[t] > max_score) max_score = sc[t];
        }

        // Softmax
        float sum = 0;
        for (int t = 0; t < T; t++) {
            sc[t] = expf(sc[t] - max_score);
            sum += sc[t];
        }
        float inv_sum = 1.0f / sum;
        for (int t = 0; t < T; t++) sc[t] *= inv_sum;

        // Weighted sum of V
        for (int d = 0; d < hd; d++) {
            float val = 0;
            for (int t = 0; t < T; t++) {
                int v_idx = (kv_h * hd + d) * c->max_seq + t;
                val += sc[t] * (float)v_cache[v_idx];
            }
            scratch->attn_f32[h * hd + d] = val;
        }
    }

    // Wo projection: out = Wo @ attn (Wo is [dim, dim] row-major)
    int D = c->dim;
    for (int i = 0; i < D; i++) {
        float val = 0;
        for (int j = 0; j < D; j++)
            val += (float)wo[i * D + j] * scratch->attn_f32[j];
        out[i] = (_Float16)((float)x_residual[i] + val);
    }
}
