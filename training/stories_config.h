// stories_config.h — Stories110M model config and structures
#pragma once
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#import <Accelerate/Accelerate.h>
#include <math.h>
#include <unistd.h>
#include <dispatch/dispatch.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

// Stories110M config
#define DIM 768
#define HIDDEN 2048
#define HEADS 12
#define HD (DIM/HEADS)
#define SEQ 256
#define NLAYERS 12
#define VOCAB 32000
#define DEFAULT_ACCUM_STEPS 50
#define DEFAULT_MAX_COMPILES 200

// Per compile: 6 weight-bearing kernels per layer + 1 classifier = 6*12+1 = 73
// Plus 1 static (sdpaBwd2 per layer, no weights) = 12 more but those are weight-free
// Weight-bearing: qkvFwd(1) + attnFwd(1) + fwdFFN(1) + ffnBwd(1) + sdpaBwd1(1) + qkvBwd(1) = 6 per layer
// 6 * 12 = 72 weight-bearing compiles per batch
// With MAX_COMPILES=200, we get 2 batches per exec() cycle
#define KERNELS_PER_LAYER 6
#define TOTAL_WEIGHT_KERNELS (KERNELS_PER_LAYER * NLAYERS)
#define CLS_KERNELS 2

// Attention score channels for SDPA backward
#define SCORE_CH (HEADS*SEQ)

// Weight sizes per layer
#define WQ_SZ (DIM*DIM)
#define WO_SZ (DIM*DIM)
#define W1_SZ (HIDDEN*DIM)
#define W2_SZ (DIM*HIDDEN)
#define W3_SZ (HIDDEN*DIM)
#define LAYER_PARAMS (4*WQ_SZ + W1_SZ + W2_SZ + W3_SZ + 2*DIM)
#define TOTAL_PARAMS (NLAYERS * LAYER_PARAMS + DIM + VOCAB*DIM)  // +rms_final+embed

// Per-layer weight and optimizer state
typedef struct {
    float *Wq, *Wk, *Wv, *Wo;
    float *W1, *W2, *W3;
    float *rms_att, *rms_ffn;
} LayerWeights;

typedef struct {
    float *m, *v;
    size_t n;
} AdamState;

typedef struct {
    AdamState Wq, Wk, Wv, Wo;
    AdamState W1, W2, W3;
    AdamState rms_att, rms_ffn;
} LayerAdam;

// Per-layer activation buffers (saved for backward)
typedef struct {
    float *layer_in;    // [DIM, SEQ] input to this layer (for rmsnorm1 bwd)
    float *Q, *K, *V;  // [DIM, SEQ] QKV projections
    float *x2;          // [DIM, SEQ] residual after attn (fused on ANE)
    float *h1, *h3;     // [HIDDEN, SEQ] FFN intermediates
    float *rrms_att;    // [SEQ] cached rrms from rmsnorm1 (for dw)
    float *rrms_ffn;    // [SEQ] cached rrms from rmsnorm2 (for dw)
} LayerActs;

// Double-buffered dW capture slots (eliminates malloc+memcpy per step)
typedef struct {
    // Activations: io_read writes here directly in forward
    float *silu_out[2];  // [HIDDEN, SEQ]
    float *x2norm[2];    // [DIM, SEQ]
    float *attn_out[2];  // [DIM, SEQ]
    float *xnorm[2];     // [DIM, SEQ]
    // Gradients: io_read writes here directly in backward
    float *dh1[2];       // [HIDDEN, SEQ]
    float *dh3[2];       // [HIDDEN, SEQ]
    float *dq[2];        // [DIM, SEQ]
    float *dk[2];        // [DIM, SEQ]
    float *dv[2];        // [DIM, SEQ]
    // Gradients: memcpy from main-thread buffers
    float *dffn[2];      // [DIM, SEQ]
    float *dx2[2];       // [DIM, SEQ]
    dispatch_semaphore_t sem;
} LayerDWCap;

// Per-layer gradient accumulators
typedef struct {
    float *Wq, *Wk, *Wv, *Wo;
    float *W1, *W2, *W3;
    float *rms_att, *rms_ffn;
} LayerGrads;

// ANE kernels per layer
typedef struct { void *model; IOSurfaceRef ioIn, ioOut; void *request; void *tmpDir; } Kern;
typedef struct {
    Kern *qkvFwd, *attnFwd, *fwdFFN, *ffnBwd, *sdpaBwd1, *sdpaBwd2, *qkvBwd, *rmsBwd;
} LayerKernels;

// Checkpoint header
typedef struct {
    int magic;          // 0x424C5A54 "BLZT"
    int version;        // 2
    int step, total_steps;
    int n_layers, vocab_size, dim, hidden_dim, n_heads, seq_len;
    float lr, loss;
    double cum_compile, cum_train, cum_wall;
    int cum_steps, cum_batches;
    int adam_t;
    int pad[3];         // alignment
} CkptHdr;

// llama2.c model file header
typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} Llama2Config;

// Globals
static Class g_D, g_I, g_AR, g_AIO;
static mach_timebase_info_data_t g_tb;
static int g_compile_count = 0;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
}
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// Alloc helpers
static AdamState adam_alloc(size_t n) { AdamState s; s.m=(float*)calloc(n,4); s.v=(float*)calloc(n,4); s.n=n; return s; }
static void adam_free(AdamState *s) { free(s->m); free(s->v); }

static LayerWeights layer_weights_alloc(void) {
    LayerWeights w;
    w.Wq=(float*)malloc(WQ_SZ*4); w.Wk=(float*)malloc(WQ_SZ*4);
    w.Wv=(float*)malloc(WQ_SZ*4); w.Wo=(float*)malloc(WO_SZ*4);
    w.W1=(float*)malloc(W1_SZ*4); w.W2=(float*)malloc(W2_SZ*4); w.W3=(float*)malloc(W3_SZ*4);
    w.rms_att=(float*)malloc(DIM*4); w.rms_ffn=(float*)malloc(DIM*4);
    return w;
}
static void layer_weights_free(LayerWeights *w) {
    free(w->Wq);free(w->Wk);free(w->Wv);free(w->Wo);
    free(w->W1);free(w->W2);free(w->W3);
    free(w->rms_att);free(w->rms_ffn);
}
static LayerAdam layer_adam_alloc(void) {
    LayerAdam a;
    a.Wq=adam_alloc(WQ_SZ); a.Wk=adam_alloc(WQ_SZ); a.Wv=adam_alloc(WQ_SZ); a.Wo=adam_alloc(WO_SZ);
    a.W1=adam_alloc(W1_SZ); a.W2=adam_alloc(W2_SZ); a.W3=adam_alloc(W3_SZ);
    a.rms_att=adam_alloc(DIM); a.rms_ffn=adam_alloc(DIM);
    return a;
}
static void layer_adam_free(LayerAdam *a) {
    adam_free(&a->Wq);adam_free(&a->Wk);adam_free(&a->Wv);adam_free(&a->Wo);
    adam_free(&a->W1);adam_free(&a->W2);adam_free(&a->W3);
    adam_free(&a->rms_att);adam_free(&a->rms_ffn);
}
static LayerActs layer_acts_alloc(void) {
    LayerActs a;
    a.layer_in=(float*)malloc(SEQ*DIM*4);
    a.Q=(float*)malloc(SEQ*DIM*4);
    a.K=(float*)malloc(SEQ*DIM*4); a.V=(float*)malloc(SEQ*DIM*4);
    a.x2=(float*)malloc(SEQ*DIM*4);
    a.h1=(float*)malloc(SEQ*HIDDEN*4); a.h3=(float*)malloc(SEQ*HIDDEN*4);
    a.rrms_att=(float*)malloc(SEQ*4); a.rrms_ffn=(float*)malloc(SEQ*4);
    return a;
}
static void layer_acts_free(LayerActs *a) {
    free(a->layer_in);free(a->Q);free(a->K);free(a->V);
    free(a->x2);
    free(a->h1);free(a->h3);
    free(a->rrms_att);free(a->rrms_ffn);
}
static LayerDWCap layer_dwcap_alloc(void) {
    LayerDWCap c;
    for (int s=0; s<2; s++) {
        c.silu_out[s]=(float*)malloc(SEQ*HIDDEN*4);
        c.x2norm[s]=(float*)malloc(SEQ*DIM*4);
        c.attn_out[s]=(float*)malloc(SEQ*DIM*4);
        c.xnorm[s]=(float*)malloc(SEQ*DIM*4);
        c.dh1[s]=(float*)malloc(SEQ*HIDDEN*4);
        c.dh3[s]=(float*)malloc(SEQ*HIDDEN*4);
        c.dq[s]=(float*)malloc(SEQ*DIM*4);
        c.dk[s]=(float*)malloc(SEQ*DIM*4);
        c.dv[s]=(float*)malloc(SEQ*DIM*4);
        c.dffn[s]=(float*)malloc(SEQ*DIM*4);
        c.dx2[s]=(float*)malloc(SEQ*DIM*4);
    }
    c.sem = dispatch_semaphore_create(2);
    return c;
}
static void layer_dwcap_free(LayerDWCap *c) {
    for (int s=0; s<2; s++) {
        free(c->silu_out[s]);free(c->x2norm[s]);
        free(c->attn_out[s]);free(c->xnorm[s]);
        free(c->dh1[s]);free(c->dh3[s]);
        free(c->dq[s]);free(c->dk[s]);free(c->dv[s]);
        free(c->dffn[s]);free(c->dx2[s]);
    }
}
static LayerGrads layer_grads_alloc(void) {
    LayerGrads g;
    g.Wq=(float*)calloc(WQ_SZ,4); g.Wk=(float*)calloc(WQ_SZ,4);
    g.Wv=(float*)calloc(WQ_SZ,4); g.Wo=(float*)calloc(WO_SZ,4);
    g.W1=(float*)calloc(W1_SZ,4); g.W2=(float*)calloc(W2_SZ,4); g.W3=(float*)calloc(W3_SZ,4);
    g.rms_att=(float*)calloc(DIM,4); g.rms_ffn=(float*)calloc(DIM,4);
    return g;
}
static void layer_grads_zero(LayerGrads *g) {
    memset(g->Wq,0,WQ_SZ*4);memset(g->Wk,0,WQ_SZ*4);
    memset(g->Wv,0,WQ_SZ*4);memset(g->Wo,0,WO_SZ*4);
    memset(g->W1,0,W1_SZ*4);memset(g->W2,0,W2_SZ*4);memset(g->W3,0,W3_SZ*4);
    memset(g->rms_att,0,DIM*4);memset(g->rms_ffn,0,DIM*4);
}
static void layer_grads_free(LayerGrads *g) {
    free(g->Wq);free(g->Wk);free(g->Wv);free(g->Wo);
    free(g->W1);free(g->W2);free(g->W3);
    free(g->rms_att);free(g->rms_ffn);
}
