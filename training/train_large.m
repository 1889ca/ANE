// train_large.m — Train stories110M (12 layers, 768dim, 3072hidden) on ANE
// Uses pretokenized TinyStories data with cross-entropy loss
// 5 weight-bearing ANE kernels per layer × 12 layers + 2 classifier = 62 per compile batch
#include "stories_io.h"
#include "stories_mil.h"
#include "stories_cpu_ops.h"

#define CKPT_PATH "ane_stories110M_ckpt.bin"
#define MODEL_PATH "../../assets/models/stories110M.bin"
#define DATA_PATH "tinystories_data00.bin"

// ===== Weight loading from llama2.c format =====
static bool load_pretrained(LayerWeights *lw, float *rms_final, float *embed, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); return false; }
    Llama2Config cfg;
    fread(&cfg, sizeof(cfg), 1, f);
    printf("  Model config: dim=%d hidden=%d layers=%d heads=%d vocab=%d seq=%d\n",
           cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, abs(cfg.vocab_size), cfg.seq_len);
    if (cfg.dim != DIM || cfg.hidden_dim != HIDDEN || cfg.n_layers != NLAYERS) {
        printf("  ERROR: Config mismatch! Expected dim=%d hidden=%d layers=%d\n", DIM, HIDDEN, NLAYERS);
        fclose(f); return false;
    }
    int V = abs(cfg.vocab_size);
    bool shared = cfg.vocab_size > 0;

    // Read in llama2.c order: embed, rms_att[all], wq[all], wk[all], wv[all], wo[all],
    //                         rms_ffn[all], w1[all], w2[all], w3[all], rms_final, [wcls]
    fread(embed, 4, V * DIM, f);

    // rms_att weights for all layers (contiguous)
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].rms_att, 4, DIM, f);
    // wq for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wq, 4, WQ_SZ, f);
    // wk for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wk, 4, WQ_SZ, f);
    // wv for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wv, 4, WQ_SZ, f);
    // wo for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wo, 4, WO_SZ, f);
    // rms_ffn weights for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].rms_ffn, 4, DIM, f);
    // w1 for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].W1, 4, W1_SZ, f);
    // w2 for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].W2, 4, W2_SZ, f);
    // w3 for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].W3, 4, W3_SZ, f);
    // rms_final
    fread(rms_final, 4, DIM, f);
    // wcls = embed if shared (we just use embed pointer)

    fclose(f);
    printf("  Loaded pretrained weights (%s)\n", shared ? "shared embed/cls" : "separate cls");
    return true;
}

// ===== Compile one layer's kernels =====
static bool compile_layer_kernels(LayerKernels *lk, LayerWeights *w) {
    lk->fwdAttn = compile_kern_mil_w(gen_sdpa_fwd_taps(), (@{
        @"@model_path/weights/rms1.bin": @{@"offset":@0, @"data":build_blob(w->rms_att,1,DIM)},
        @"@model_path/weights/wq.bin": @{@"offset":@0, @"data":build_blob(w->Wq,DIM,DIM)},
        @"@model_path/weights/wk.bin": @{@"offset":@0, @"data":build_blob(w->Wk,DIM,DIM)},
        @"@model_path/weights/wv.bin": @{@"offset":@0, @"data":build_blob(w->Wv,DIM,DIM)},
        @"@model_path/weights/wo.bin": @{@"offset":@0, @"data":build_blob(w->Wo,DIM,DIM)},
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
    }), DIM*SEQ*2, (6*DIM+1)*SEQ*2);

    lk->fwdFFN = compile_kern_mil_w(gen_ffn_fwd_taps(), (@{
        @"@model_path/weights/rms2.bin": @{@"offset":@0, @"data":build_blob(w->rms_ffn,1,DIM)},
        @"@model_path/weights/w1.bin": @{@"offset":@0, @"data":build_blob(w->W1,HIDDEN,DIM)},
        @"@model_path/weights/w3.bin": @{@"offset":@0, @"data":build_blob(w->W3,HIDDEN,DIM)},
        @"@model_path/weights/w2.bin": @{@"offset":@0, @"data":build_blob(w->W2,DIM,HIDDEN)},
    }), DIM*SEQ*2, (2*DIM+3*HIDDEN+1)*SEQ*2);

    lk->ffnBwd = compile_kern_mil_w(gen_ffn_bwd(), (@{
        @"@model_path/weights/w2t.bin": @{@"offset":@0, @"data":build_blob_t(w->W2,DIM,HIDDEN)},
        @"@model_path/weights/w1t.bin": @{@"offset":@0, @"data":build_blob_t(w->W1,HIDDEN,DIM)},
        @"@model_path/weights/w3t.bin": @{@"offset":@0, @"data":build_blob_t(w->W3,HIDDEN,DIM)},
    }), (DIM+2*HIDDEN)*SEQ*2, (DIM+2*HIDDEN)*SEQ*2);

    lk->sdpaBwd1 = compile_kern_mil_w(gen_sdpa_bwd1(), (@{
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
        @"@model_path/weights/wot.bin": @{@"offset":@0, @"data":build_blob_t(w->Wo,DIM,DIM)},
    }), 4*DIM*SEQ*2, (DIM+2*SCORE_CH)*SEQ*2);

    lk->qkvBwd = compile_kern_mil_w(gen_qkvb(), (@{
        @"@model_path/weights/wqt.bin": @{@"offset":@0, @"data":build_blob_t(w->Wq,DIM,DIM)},
        @"@model_path/weights/wkt.bin": @{@"offset":@0, @"data":build_blob_t(w->Wk,DIM,DIM)},
        @"@model_path/weights/wvt.bin": @{@"offset":@0, @"data":build_blob_t(w->Wv,DIM,DIM)},
    }), 3*DIM*SEQ*2, DIM*SEQ*2);

    return lk->fwdAttn && lk->fwdFFN && lk->ffnBwd && lk->sdpaBwd1 && lk->qkvBwd;
}

// Compile weight-free rmsBwd (one per layer, no weights)
static Kern *compile_rms_bwd(void) {
    return compile_kern_mil_w(gen_rms_bwd(), @{},
        3*DIM*SEQ*2, DIM*SEQ*2);
}

// Compile weight-free sdpaBwd2 (only needs once, no weights)
static Kern *compile_sdpa_bwd2(void) {
    return compile_kern_mil_w(gen_sdpa_bwd2(), @{},
        (2*SCORE_CH+2*DIM)*SEQ*2, 2*DIM*SEQ*2);
}

static void free_layer_kernels(LayerKernels *lk) {
    free_kern(lk->fwdAttn); free_kern(lk->fwdFFN); free_kern(lk->ffnBwd);
    free_kern(lk->sdpaBwd1); free_kern(lk->qkvBwd);
    // sdpaBwd2, rmsBwd are shared/static, freed separately
    lk->fwdAttn = lk->fwdFFN = lk->ffnBwd = lk->sdpaBwd1 = lk->qkvBwd = NULL;
}

// ===== Checkpoint save/load =====
static void save_checkpoint(const char *path, int step, int total_steps, float lr, float loss,
                            double cc, double ct, double cw, int cs, int cb, int adam_t,
                            int accum_steps,
                            LayerWeights *lw, LayerAdam *la, float *rms_final, AdamState *arms_final,
                            float *embed, AdamState *aembed) {
    FILE *f = fopen(path, "wb");
    CkptHdr h = {0};
    h.magic = 0x424C5A54; h.version = 2;
    h.step = step; h.total_steps = total_steps;
    h.n_layers = NLAYERS; h.vocab_size = VOCAB; h.dim = DIM;
    h.hidden_dim = HIDDEN; h.n_heads = HEADS; h.seq_len = SEQ;
    h.lr = lr; h.loss = loss;
    h.cum_compile = cc; h.cum_train = ct; h.cum_wall = cw;
    h.cum_steps = cs; h.cum_batches = cb; h.adam_t = adam_t;
    h.pad[0] = accum_steps;
    fwrite(&h, sizeof(h), 1, f);
    // Per-layer weights + adam
    for (int L = 0; L < NLAYERS; L++) {
        fwrite(lw[L].Wq,4,WQ_SZ,f); fwrite(lw[L].Wk,4,WQ_SZ,f);
        fwrite(lw[L].Wv,4,WQ_SZ,f); fwrite(lw[L].Wo,4,WO_SZ,f);
        fwrite(lw[L].W1,4,W1_SZ,f); fwrite(lw[L].W2,4,W2_SZ,f); fwrite(lw[L].W3,4,W3_SZ,f);
        fwrite(lw[L].rms_att,4,DIM,f); fwrite(lw[L].rms_ffn,4,DIM,f);
        // Adam state
        fwrite(la[L].Wq.m,4,WQ_SZ,f); fwrite(la[L].Wq.v,4,WQ_SZ,f);
        fwrite(la[L].Wk.m,4,WQ_SZ,f); fwrite(la[L].Wk.v,4,WQ_SZ,f);
        fwrite(la[L].Wv.m,4,WQ_SZ,f); fwrite(la[L].Wv.v,4,WQ_SZ,f);
        fwrite(la[L].Wo.m,4,WO_SZ,f); fwrite(la[L].Wo.v,4,WO_SZ,f);
        fwrite(la[L].W1.m,4,W1_SZ,f); fwrite(la[L].W1.v,4,W1_SZ,f);
        fwrite(la[L].W2.m,4,W2_SZ,f); fwrite(la[L].W2.v,4,W2_SZ,f);
        fwrite(la[L].W3.m,4,W3_SZ,f); fwrite(la[L].W3.v,4,W3_SZ,f);
        fwrite(la[L].rms_att.m,4,DIM,f); fwrite(la[L].rms_att.v,4,DIM,f);
        fwrite(la[L].rms_ffn.m,4,DIM,f); fwrite(la[L].rms_ffn.v,4,DIM,f);
    }
    fwrite(rms_final,4,DIM,f);
    fwrite(arms_final->m,4,DIM,f); fwrite(arms_final->v,4,DIM,f);
    fwrite(embed,4,VOCAB*DIM,f);
    fwrite(aembed->m,4,VOCAB*DIM,f); fwrite(aembed->v,4,VOCAB*DIM,f);
    fclose(f);
}

static bool load_checkpoint(const char *path, int *step, int *total_steps, float *lr, float *loss,
                             double *cc, double *ct, double *cw, int *cs, int *cb, int *adam_t,
                             int *accum_steps,
                             LayerWeights *lw, LayerAdam *la, float *rms_final, AdamState *arms_final,
                             float *embed, AdamState *aembed) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    CkptHdr h;
    fread(&h, sizeof(h), 1, f);
    if (h.magic != 0x424C5A54 || h.version != 2) { fclose(f); return false; }
    *step = h.step; *total_steps = h.total_steps; *lr = h.lr; *loss = h.loss;
    *cc = h.cum_compile; *ct = h.cum_train; *cw = h.cum_wall;
    *cs = h.cum_steps; *cb = h.cum_batches; *adam_t = h.adam_t;
    if (h.pad[0] > 0) *accum_steps = h.pad[0];
    for (int L = 0; L < NLAYERS; L++) {
        fread(lw[L].Wq,4,WQ_SZ,f); fread(lw[L].Wk,4,WQ_SZ,f);
        fread(lw[L].Wv,4,WQ_SZ,f); fread(lw[L].Wo,4,WO_SZ,f);
        fread(lw[L].W1,4,W1_SZ,f); fread(lw[L].W2,4,W2_SZ,f); fread(lw[L].W3,4,W3_SZ,f);
        fread(lw[L].rms_att,4,DIM,f); fread(lw[L].rms_ffn,4,DIM,f);
        fread(la[L].Wq.m,4,WQ_SZ,f); fread(la[L].Wq.v,4,WQ_SZ,f);
        fread(la[L].Wk.m,4,WQ_SZ,f); fread(la[L].Wk.v,4,WQ_SZ,f);
        fread(la[L].Wv.m,4,WQ_SZ,f); fread(la[L].Wv.v,4,WQ_SZ,f);
        fread(la[L].Wo.m,4,WO_SZ,f); fread(la[L].Wo.v,4,WO_SZ,f);
        fread(la[L].W1.m,4,W1_SZ,f); fread(la[L].W1.v,4,W1_SZ,f);
        fread(la[L].W2.m,4,W2_SZ,f); fread(la[L].W2.v,4,W2_SZ,f);
        fread(la[L].W3.m,4,W3_SZ,f); fread(la[L].W3.v,4,W3_SZ,f);
        fread(la[L].rms_att.m,4,DIM,f); fread(la[L].rms_att.v,4,DIM,f);
        fread(la[L].rms_ffn.m,4,DIM,f); fread(la[L].rms_ffn.v,4,DIM,f);
    }
    fread(rms_final,4,DIM,f);
    fread(arms_final->m,4,DIM,f); fread(arms_final->v,4,DIM,f);
    fread(embed,4,VOCAB*DIM,f);
    fread(aembed->m,4,VOCAB*DIM,f); fread(aembed->v,4,VOCAB*DIM,f);
    fclose(f);
    return true;
}

// ===== Main =====
int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();
        mach_timebase_info(&g_tb);

        int total_steps = 10000;
        float lr = 3e-4f;
        float adam_b1=0.9f, adam_b2=0.999f, adam_eps=1e-8f;
        int adam_t = 0, start_step = 0;
        int accum_steps = DEFAULT_ACCUM_STEPS;
        int max_compiles = DEFAULT_MAX_COMPILES;

        // Parse args
        bool do_resume = false;
        for (int i=1; i<argc; i++) {
            if (strcmp(argv[i], "--resume") == 0) do_resume = true;
            else if (strcmp(argv[i], "--steps") == 0 && i+1<argc) total_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--lr") == 0 && i+1<argc) lr = atof(argv[++i]);
            else if (strcmp(argv[i], "--accum") == 0 && i+1<argc) accum_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--max-compiles") == 0 && i+1<argc) max_compiles = atoi(argv[++i]);
        }

        // Allocate per-layer state
        LayerWeights lw[NLAYERS];
        LayerAdam la[NLAYERS];
        LayerActs acts[NLAYERS];
        LayerGrads grads[NLAYERS];
        LayerKernels kern[NLAYERS];
        LayerDWCap dwcap[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            lw[L] = layer_weights_alloc();
            la[L] = layer_adam_alloc();
            acts[L] = layer_acts_alloc();
            grads[L] = layer_grads_alloc();
            dwcap[L] = layer_dwcap_alloc();
            memset(&kern[L], 0, sizeof(LayerKernels));
        }

        // Final RMSNorm + embedding + classifier
        float *rms_final = (float*)malloc(DIM*4);
        float *rrms_final = (float*)malloc(SEQ*4);
        float *embed = (float*)malloc(VOCAB*DIM*4);  // [VOCAB, DIM] row-major
        float *grms_final = (float*)calloc(DIM, 4);
        float *gembed_cls = (float*)calloc(VOCAB*DIM, 4);  // dense sgemm accumulator
        float *gembed_emb = (float*)calloc(VOCAB*DIM, 4);  // scatter-add accumulator
        // Double-buffered dlogits/x_final capture (eliminates 32MB malloc/free per step)
        float *capt_dlogits[2], *capt_xfinal[2];
        for (int s=0; s<2; s++) {
            capt_dlogits[s] = (float*)malloc((size_t)SEQ*VOCAB*4);
            capt_xfinal[s] = (float*)malloc(SEQ*DIM*4);
        }
        AdamState arms_final = adam_alloc(DIM);
        AdamState aembed = adam_alloc((size_t)VOCAB*DIM);

        double cum_compile=0, cum_train=0, cum_wall=0;
        int cum_steps=0, cum_batches=0;

        float resume_loss = 0;
        bool resuming = false;
        if (do_resume) {
            resuming = load_checkpoint(CKPT_PATH, &start_step, &total_steps, &lr, &resume_loss,
                &cum_compile, &cum_train, &cum_wall, &cum_steps, &cum_batches, &adam_t,
                &accum_steps,
                lw, la, rms_final, &arms_final, embed, &aembed);
            if (resuming) printf("[RESUMED step %d, loss=%.4f]\n", start_step, resume_loss);
        }
        if (!resuming) {
            printf("=== ANE Training: Stories110M (12 layers) ===\n");
            printf("dim=%d hidden=%d heads=%d seq=%d vocab=%d layers=%d\n", DIM, HIDDEN, HEADS, SEQ, VOCAB, NLAYERS);
            if (!load_pretrained(lw, rms_final, embed, MODEL_PATH)) {
                printf("Pretrained load failed, using random init\n");
                srand48(42);
                float scale_d=1.0f/sqrtf(DIM), scale_h=1.0f/sqrtf(HIDDEN);
                for (int L=0; L<NLAYERS; L++) {
                    for(size_t i=0;i<WQ_SZ;i++){lw[L].Wq[i]=scale_d*(2*drand48()-1);lw[L].Wk[i]=scale_d*(2*drand48()-1);}
                    for(size_t i=0;i<WQ_SZ;i++){lw[L].Wv[i]=scale_d*(2*drand48()-1);lw[L].Wo[i]=scale_d*(2*drand48()-1);}
                    for(size_t i=0;i<W1_SZ;i++) lw[L].W1[i]=scale_h*(2*drand48()-1);
                    for(size_t i=0;i<W2_SZ;i++) lw[L].W2[i]=scale_d*(2*drand48()-1);
                    for(size_t i=0;i<W3_SZ;i++) lw[L].W3[i]=scale_h*(2*drand48()-1);
                    for(int i=0;i<DIM;i++){lw[L].rms_att[i]=1.0f; lw[L].rms_ffn[i]=1.0f;}
                }
                for(int i=0;i<DIM;i++) rms_final[i]=1.0f;
                float escale = 0.02f;
                for(size_t i=0;i<(size_t)VOCAB*DIM;i++) embed[i]=escale*(2*drand48()-1);
            }
            size_t tp = (size_t)NLAYERS*LAYER_PARAMS + DIM + (size_t)VOCAB*DIM;
            double xfmr_params = (double)NLAYERS*LAYER_PARAMS;
            double embed_params = (double)VOCAB*DIM;
            printf("Params: %.2fM (transformer %.2fM + embed %.2fM)\n", tp/1e6, xfmr_params/1e6, embed_params/1e6);
            printf("Kernels: %d (%d weight-bearing + %d static sdpaBwd2)\n",
                   TOTAL_WEIGHT_KERNELS+NLAYERS, TOTAL_WEIGHT_KERNELS, NLAYERS);
            printf("Accum %d steps per recompile | Adam LR=%.1e b1=%.1f b2=%.3f\n", accum_steps, lr, adam_b1, adam_b2);
            double fwd_f = NLAYERS*(4.0*2*DIM*DIM*SEQ + 2.0*2*DIM*HIDDEN*SEQ + 2.0*HIDDEN*DIM*SEQ);
            double bwd_dx_f = fwd_f, bwd_dw_f = fwd_f;
            double sdpa_f = NLAYERS*2.0*HEADS*5*SEQ*SEQ*HD;
            double cls_f = 2.0*VOCAB*DIM*SEQ;
            double total_f = fwd_f + bwd_dx_f + bwd_dw_f + sdpa_f + cls_f*3;
            double ane_f = fwd_f + bwd_dx_f + sdpa_f;
            printf("FLOPs/step: fwd=%.0fM bwd_dx=%.0fM bwd_dW=%.0fM sdpa_bwd=%.0fM total=%.0fM\n",
                   fwd_f/1e6, bwd_dx_f/1e6, bwd_dw_f/1e6, sdpa_f/1e6, total_f/1e6);
            printf("ANE FLOPs/step: %.0fM (fwd+bwd_dx+sdpa_bwd+cls) | CPU: dW (cblas)\n\n", ane_f/1e6);
        }

        // mmap token data
        int data_fd = open(DATA_PATH, O_RDONLY);
        if (data_fd < 0) { printf("Cannot open %s\n", DATA_PATH); return 1; }
        struct stat st; fstat(data_fd, &st);
        size_t data_len = st.st_size;
        uint16_t *token_data = (uint16_t*)mmap(NULL, data_len, PROT_READ, MAP_PRIVATE, data_fd, 0);
        if (token_data == MAP_FAILED) { printf("mmap failed\n"); return 1; }
        size_t n_tokens = data_len / 2;
        printf("Token data: %zu tokens (%.1f MB)\n", n_tokens, data_len/1e6);

        // Gradient buffers shared across layers (reused each step)
        float *dy = (float*)malloc(SEQ*DIM*4);            // gradient flowing backward
        float *dx_ffn = (float*)malloc(SEQ*DIM*4);
        float *dx2 = (float*)malloc(SEQ*DIM*4);
        float *dx_attn = (float*)malloc(SEQ*DIM*4);

        // x buffer for input to each layer (channel-first [DIM, SEQ])
        float *x_cur = (float*)malloc(SEQ*DIM*4);
        float *x_final = (float*)malloc(SEQ*DIM*4);     // after final rmsnorm
        float *logits = (float*)malloc(SEQ*VOCAB*4);     // [VOCAB, SEQ] for cross-entropy
        float *dlogits = (float*)malloc(SEQ*VOCAB*4);

        // Compile static sdpaBwd2 kernels (no weights, one per layer)
        Kern *sdpaBwd2[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            sdpaBwd2[L] = compile_sdpa_bwd2();
            if (!sdpaBwd2[L]) { printf("sdpaBwd2 compile failed\n"); return 1; }
        }

        // Compile static rmsBwd kernels (no weights, one per layer + 1 final)
        for (int L=0; L<NLAYERS; L++) {
            kern[L].rmsBwd = compile_rms_bwd();
            if (!kern[L].rmsBwd) { printf("rmsBwd compile failed at layer %d\n", L); return 1; }
        }
        Kern *rmsBwdFinal = compile_rms_bwd();
        if (!rmsBwdFinal) { printf("rmsBwdFinal compile failed\n"); return 1; }

        // Classifier ANE kernels (not per-layer, recompiled each batch with embed weights)
        Kern *cls_fwd = NULL, *cls_bwd = NULL;
        bool use_ane_cls = true;

        // Softmax ANE kernel (weight-free, compiled once)
        Kern *softmax_kern = compile_kern_mil_w(gen_softmax(), @{},
            VOCAB*SEQ*2, VOCAB*SEQ*2);
        bool use_ane_softmax = (softmax_kern != NULL);
        if (!use_ane_softmax) printf("  [softmax] ANE compile failed, falling back to CPU\n");

        // Per-layer serial queues allow different layers' dW to run in parallel
        // while serializing same-layer dW across steps (protects gradient accumulators)
        dispatch_queue_t dw_layer_q[NLAYERS];
        for (int L=0; L<NLAYERS; L++)
            dw_layer_q[L] = dispatch_queue_create("dw_layer", DISPATCH_QUEUE_SERIAL);
        dispatch_queue_t dw_embed_q = dispatch_queue_create("dw_embed", DISPATCH_QUEUE_SERIAL);
        dispatch_group_t layer_dw_grp = dispatch_group_create();
        dispatch_group_t embed_dw_grp = dispatch_group_create();

        float last_loss = 999.0f;
        double total_compile_ms=0, total_train_ms=0;
        int total_steps_done=0, total_batches=0;
        uint64_t t_wall_start = mach_absolute_time();

        srand48(42 + start_step);

        int step = start_step;
        while (step < total_steps) {
            // Check compile budget
            if (g_compile_count + TOTAL_WEIGHT_KERNELS + (use_ane_cls ? CLS_KERNELS : 0) > max_compiles) {
                for (int L=0; L<NLAYERS; L++) { free_layer_kernels(&kern[L]); free_kern(sdpaBwd2[L]); free_kern(kern[L].rmsBwd); kern[L].rmsBwd = NULL; }
                free_kern(rmsBwdFinal); rmsBwdFinal = NULL;
                free_kern(cls_fwd); free_kern(cls_bwd); cls_fwd = cls_bwd = NULL;
                free_kern(softmax_kern); softmax_kern = NULL;
                double wall = tb_ms(mach_absolute_time() - t_wall_start);
                save_checkpoint(CKPT_PATH, step, total_steps, lr, last_loss,
                    total_compile_ms+cum_compile, total_train_ms+cum_train, wall+cum_wall,
                    total_steps_done+cum_steps, total_batches+cum_batches, adam_t,
                    accum_steps,
                    lw, la, rms_final, &arms_final, embed, &aembed);
                printf("[exec() restart step %d, %d compiles, loss=%.4f]\n", step, g_compile_count, last_loss);
                fflush(stdout);
                // Build new argv: original args + --resume (deduped)
                char **new_argv = (char**)calloc(argc + 3, sizeof(char*));
                int n = 0;
                new_argv[n++] = argv[0];
                bool has_resume = false;
                for (int i = 1; i < argc; i++) {
                    if (strcmp(argv[i], "--resume") == 0) { has_resume = true; }
                    new_argv[n++] = argv[i];
                }
                if (!has_resume) new_argv[n++] = "--resume";
                new_argv[n] = NULL;
                execv(argv[0], new_argv);
                perror("execv"); free(new_argv); return 1;
            }

            // Compile all layers' weight-bearing kernels
            uint64_t tc = mach_absolute_time();
            for (int L=0; L<NLAYERS; L++) free_layer_kernels(&kern[L]);

            __block bool compile_ok = true;
            dispatch_queue_t cq = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
            dispatch_semaphore_t csem = dispatch_semaphore_create(4);
            dispatch_group_t cgrp = dispatch_group_create();
            LayerKernels *kern_p = kern;
            LayerWeights *lw_p = lw;
            for (int L=0; L<NLAYERS; L++) {
                dispatch_group_async(cgrp, cq, ^{
                    dispatch_semaphore_wait(csem, DISPATCH_TIME_FOREVER);
                    if (compile_ok && !compile_layer_kernels(&kern_p[L], &lw_p[L])) {
                        printf("\nCompile failed at layer %d, restart\n", L);
                        compile_ok = false;
                    }
                    dispatch_semaphore_signal(csem);
                });
            }
            dispatch_group_wait(cgrp, DISPATCH_TIME_FOREVER);
            if (!compile_ok) { g_compile_count = max_compiles; continue; }

            // Re-compile sdpaBwd2 and rmsBwd if needed (after exec restart)
            for (int L=0; L<NLAYERS; L++) {
                if (!sdpaBwd2[L]) {
                    sdpaBwd2[L] = compile_sdpa_bwd2();
                    if (!sdpaBwd2[L]) { printf("sdpaBwd2 recompile failed\n"); return 1; }
                }
                if (!kern[L].rmsBwd) {
                    kern[L].rmsBwd = compile_rms_bwd();
                    if (!kern[L].rmsBwd) { printf("rmsBwd recompile failed\n"); return 1; }
                }
            }
            if (!rmsBwdFinal) {
                rmsBwdFinal = compile_rms_bwd();
                if (!rmsBwdFinal) { printf("rmsBwdFinal recompile failed\n"); return 1; }
            }
            if (!softmax_kern && use_ane_softmax) {
                softmax_kern = compile_kern_mil_w(gen_softmax(), @{},
                    VOCAB*SEQ*2, VOCAB*SEQ*2);
                if (!softmax_kern) { printf("  [softmax] recompile failed, falling back to CPU\n"); use_ane_softmax = false; }
            }

            // Compile classifier ANE kernels (embed weights change each batch)
            if (use_ane_cls) {
                free_kern(cls_fwd); free_kern(cls_bwd);
                cls_fwd = compile_kern_mil_w(gen_cls_fwd(), (@{
                    @"@model_path/weights/embed.bin": @{@"offset":@0, @"data":build_blob(embed, VOCAB, DIM)},
                }), DIM*SEQ*2, VOCAB*SEQ*2);
                cls_bwd = compile_kern_mil_w(gen_cls_bwd(), (@{
                    @"@model_path/weights/embed_t.bin": @{@"offset":@0, @"data":build_blob_t(embed, VOCAB, DIM)},
                }), VOCAB*SEQ*2, DIM*SEQ*2);
                if (!cls_fwd || !cls_bwd) {
                    printf("  [cls] ANE compile failed (VOCAB=%d may exceed channel limit), falling back to CPU\n", VOCAB);
                    free_kern(cls_fwd); free_kern(cls_bwd);
                    cls_fwd = cls_bwd = NULL;
                    use_ane_cls = false;
                }
            }

            int n_compiled = TOTAL_WEIGHT_KERNELS + (use_ane_cls ? CLS_KERNELS : 0);
            double cms = tb_ms(mach_absolute_time() - tc);
            total_compile_ms += cms;
            printf("  Compiled %d kernels in %.0fms%s\n", n_compiled, cms, use_ane_cls ? " (cls=ANE)" : " (cls=CPU)");

            // Zero gradient accumulators
            for (int L=0; L<NLAYERS; L++) layer_grads_zero(&grads[L]);
            memset(grms_final, 0, DIM*4);
            memset(gembed_cls, 0, (size_t)VOCAB*DIM*4);
            memset(gembed_emb, 0, (size_t)VOCAB*DIM*4);

            int steps_batch = 0;
            uint64_t tt = mach_absolute_time();
            double t_ane=0,t_io=0,t_rms=0,t_cls=0;
            double t_embed=0,t_resid=0,t_xent=0,t_memcpy=0,t_rms_bwd=0,t_embed_bwd=0;
            double t_bwd_ane=0,t_bwd_io=0,t_cls_bwd=0;
            // dW instrumentation: per-layer sgemm times (heap-alloc for block capture)
            double *dw_t_ffn = (double*)calloc(NLAYERS, sizeof(double));
            double *dw_t_wo  = (double*)calloc(NLAYERS, sizeof(double));
            double *dw_t_qkv = (double*)calloc(NLAYERS, sizeof(double));
            __block double dw_t_embed = 0;
            // Semaphore wait times (main thread only)
            double t_sem_wait = 0;
            double sem_wait_layer[NLAYERS]; memset(sem_wait_layer, 0, sizeof(sem_wait_layer));

            for (int a=0; a<accum_steps && step<total_steps; a++, step++) {
                int slot = a % 2;
                uint64_t t0,t1;
                // Sample random position in token data
                size_t max_pos = n_tokens - SEQ - 1;
                size_t pos = (size_t)(drand48() * max_pos);
                uint16_t *input_tokens = token_data + pos;
                uint16_t *target_tokens = token_data + pos + 1;

                // Embedding lookup → x_cur [DIM, SEQ] channel-first
                t0=mach_absolute_time();
                embed_lookup(x_cur, embed, input_tokens, DIM, SEQ);
                t1=mach_absolute_time(); t_embed+=tb_ms(t1-t0);

                // ===== FORWARD (12 layers) =====
                for (int L=0; L<NLAYERS; L++) {
                    LayerActs *ac = &acts[L];

                    // Save layer input for rmsnorm1 backward
                    t0=mach_absolute_time();
                    memcpy(ac->layer_in, x_cur, SEQ*DIM*4);
                    t1=mach_absolute_time(); t_memcpy+=tb_ms(t1-t0);
                    // Attention forward: x_cur → x2,Q,K,V,attn_out,xnorm (residual fused on ANE)
                    t0=mach_absolute_time();
                    io_write_fp16(kern[L].fwdAttn->ioIn, x_cur, DIM, SEQ);
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(kern[L].fwdAttn);
                    t1=mach_absolute_time(); t_ane+=tb_ms(t1-t0); t0=t1;
                    io_copy(kern[L].fwdFFN->ioIn, 0, kern[L].fwdAttn->ioOut, 0, DIM, SEQ);
                    io_read_fp16(kern[L].fwdAttn->ioOut, ac->x2,                  0,     DIM, SEQ);
                    io_read_fp16(kern[L].fwdAttn->ioOut, dwcap[L].attn_out[slot], 4*DIM, DIM, SEQ);
                    io_read_fp16(kern[L].fwdAttn->ioOut, dwcap[L].xnorm[slot],    5*DIM, DIM, SEQ);
                    io_read_fp16(kern[L].fwdAttn->ioOut, ac->rrms_att,           6*DIM, 1,   SEQ);
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;

                    // FFN forward (x2 already piped via io_copy)
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(kern[L].fwdFFN);
                    t1=mach_absolute_time(); t_ane+=tb_ms(t1-t0); t0=t1;
                    io_read_fp16(kern[L].fwdFFN->ioOut, x_cur,                   0,            DIM,    SEQ);
                    io_read_fp16(kern[L].fwdFFN->ioOut, ac->h1,                  DIM,          HIDDEN, SEQ);
                    io_read_fp16(kern[L].fwdFFN->ioOut, ac->h3,                  DIM+HIDDEN,   HIDDEN, SEQ);
                    io_read_fp16(kern[L].fwdFFN->ioOut, dwcap[L].silu_out[slot], DIM+2*HIDDEN, HIDDEN, SEQ);
                    io_read_fp16(kern[L].fwdFFN->ioOut, dwcap[L].x2norm[slot],   DIM+3*HIDDEN, DIM,    SEQ);
                    io_read_fp16(kern[L].fwdFFN->ioOut, ac->rrms_ffn,          2*DIM+3*HIDDEN, 1, SEQ);
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0);
                }

                // Final RMSNorm (CPU)
                t0=mach_absolute_time();
                rmsnorm(x_final, x_cur, rms_final, rrms_final, DIM, SEQ);
                t1=mach_absolute_time(); t_rms+=tb_ms(t1-t0); t0=t1;

                // Classifier forward: logits = embed @ x_final
                if (use_ane_cls) {
                    io_write_fp16(cls_fwd->ioIn, x_final, DIM, SEQ);
                    ane_eval(cls_fwd);
                } else {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                VOCAB, SEQ, DIM, 1.0f,
                                embed, DIM, x_final, SEQ, 0.0f, logits, SEQ);
                }
                t1=mach_absolute_time(); t_cls+=tb_ms(t1-t0); t0=t1;

                // Cross-entropy: softmax on ANE, loss+gradient on CPU
                float loss;
                if (use_ane_softmax && use_ane_cls) {
                    io_copy(softmax_kern->ioIn, 0, cls_fwd->ioOut, 0, VOCAB, SEQ);
                    ane_eval(softmax_kern);
                    io_read_fp16(softmax_kern->ioOut, dlogits, 0, VOCAB, SEQ);
                    // Loss: read target probabilities (256 lookups)
                    float total_loss = 0;
                    for (int t = 0; t < SEQ; t++)
                        total_loss -= logf(dlogits[target_tokens[t]*SEQ + t] + 1e-10f);
                    loss = total_loss / SEQ;
                    // Gradient: dlogits = probs/S, correct at target positions
                    float invS = 1.0f / SEQ;
                    vDSP_vsmul(dlogits, 1, &invS, dlogits, 1, (vDSP_Length)(VOCAB*SEQ));
                    for (int t = 0; t < SEQ; t++)
                        dlogits[target_tokens[t]*SEQ + t] -= invS;
                } else {
                    if (use_ane_cls)
                        io_read_fp16(cls_fwd->ioOut, logits, 0, VOCAB, SEQ);
                    loss = cross_entropy_loss(dlogits, logits, target_tokens, VOCAB, SEQ);
                }
                last_loss = loss;
                t1=mach_absolute_time(); t_xent+=tb_ms(t1-t0); t0=t1;

                // ===== BACKWARD =====
                // Classifier backward: dx_final = embed^T @ dlogits
                t0=mach_absolute_time();
                if (use_ane_cls) {
                    io_write_fp16(cls_bwd->ioIn, dlogits, VOCAB, SEQ);
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(cls_bwd);
                    t1=mach_absolute_time(); t_cls_bwd+=tb_ms(t1-t0); t0=t1;
                    io_read_fp16(cls_bwd->ioOut, dy, 0, DIM, SEQ);
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0);
                } else {
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                DIM, SEQ, VOCAB, 1.0f,
                                embed, DIM, dlogits, SEQ, 0.0f, dy, SEQ);
                    t1=mach_absolute_time(); t_cls_bwd+=tb_ms(t1-t0);
                }

                // dembed_cls[VOCAB,DIM] += dlogits[VOCAB,SEQ] @ x_final^T[SEQ,DIM]
                // Double-buffered capture (no malloc), tiled sgemm via dispatch_apply
                float *cd = capt_dlogits[slot], *cx = capt_xfinal[slot];
                memcpy(cd, dlogits, (size_t)SEQ*VOCAB*4);
                memcpy(cx, x_final, SEQ*DIM*4);
                dispatch_group_async(embed_dw_grp, dw_embed_q, ^{
                    uint64_t dw0=mach_absolute_time();
                    int ntiles = 8, chunk = VOCAB / ntiles;
                    dispatch_apply(ntiles, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
                        int start = (int)i * chunk;
                        int rows = ((int)i == ntiles-1) ? (VOCAB - start) : chunk;
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                    rows, DIM, SEQ, 1.0f,
                                    cd + (size_t)start*SEQ, SEQ, cx, SEQ,
                                    1.0f, gembed_cls + (size_t)start*DIM, DIM);
                    });
                    dw_t_embed+=tb_ms(mach_absolute_time()-dw0);
                });

                // Final RMSNorm backward: dw on CPU, dx on ANE
                t0=mach_absolute_time();
                rmsnorm_dw(grms_final, dy, x_cur, rrms_final, DIM, SEQ);
                t1=mach_absolute_time(); t_rms_bwd+=tb_ms(t1-t0); t0=t1;
                if (use_ane_cls)
                    io_copy(rmsBwdFinal->ioIn, 0, cls_bwd->ioOut, 0, DIM, SEQ);
                else
                    io_write_fp16_at(rmsBwdFinal->ioIn, 0, dy, DIM, SEQ);
                io_copy(rmsBwdFinal->ioIn, DIM, kern[NLAYERS-1].fwdFFN->ioOut, 0, DIM, SEQ);
                io_write_fp16_vec(rmsBwdFinal->ioIn, 2*DIM, rms_final, DIM, SEQ);
                t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0); t0=t1;
                ane_eval(rmsBwdFinal);
                t1=mach_absolute_time(); t_bwd_ane+=tb_ms(t1-t0); t0=t1;
                io_read_fp16(rmsBwdFinal->ioOut, dy, 0, DIM, SEQ);
                t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0);

                // ===== BACKWARD (12 layers, reverse) =====
                for (int L=NLAYERS-1; L>=0; L--) {
                    LayerActs *ac = &acts[L];
                    LayerGrads *gr = &grads[L];
                    LayerDWCap *cap = &dwcap[L];

                    // Acquire capture slot (blocks if slot still in use by async dW from 2 steps ago)
                    { uint64_t sw0=mach_absolute_time();
                    dispatch_semaphore_wait(cap->sem, DISPATCH_TIME_FOREVER);
                    double sw=tb_ms(mach_absolute_time()-sw0); t_sem_wait+=sw; sem_wait_layer[L]+=sw; }

                    // dy is the gradient at the output of this layer
                    t0=mach_absolute_time();
                    memcpy(cap->dffn[slot], dy, SEQ*DIM*4);
                    t1=mach_absolute_time(); t_memcpy+=tb_ms(t1-t0);

                    // FFN backward (ANE)
                    t0=mach_absolute_time();
                    io_write_fp16_at(kern[L].ffnBwd->ioIn, 0, cap->dffn[slot], DIM, SEQ);
                    io_copy(kern[L].ffnBwd->ioIn, DIM, kern[L].fwdFFN->ioOut, DIM, 2*HIDDEN, SEQ);
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(kern[L].ffnBwd);
                    t1=mach_absolute_time(); t_bwd_ane+=tb_ms(t1-t0); t0=t1;
                    io_read_fp16(kern[L].ffnBwd->ioOut, dx_ffn,         0,          DIM,    SEQ);
                    io_read_fp16(kern[L].ffnBwd->ioOut, cap->dh1[slot], DIM,        HIDDEN, SEQ);
                    io_read_fp16(kern[L].ffnBwd->ioOut, cap->dh3[slot], DIM+HIDDEN, HIDDEN, SEQ);
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0);

                    // dW FFN async (silu_out[slot], x2norm[slot] populated in forward)
                    dispatch_group_async(layer_dw_grp, dw_layer_q[L], ^{
                        uint64_t dw0=mach_absolute_time();
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, HIDDEN, SEQ,
                                    1.0f, cap->dffn[slot], SEQ, cap->silu_out[slot], SEQ, 1.0f, gr->W2, HIDDEN);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                    1.0f, cap->dh1[slot], SEQ, cap->x2norm[slot], SEQ, 1.0f, gr->W1, DIM);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                    1.0f, cap->dh3[slot], SEQ, cap->x2norm[slot], SEQ, 1.0f, gr->W3, DIM);
                        dw_t_ffn[L]+=tb_ms(mach_absolute_time()-dw0);
                    });

                    // RMSNorm2 backward: dw on CPU, dx on ANE
                    t0=mach_absolute_time();
                    rmsnorm_dw(gr->rms_ffn, dx_ffn, ac->x2, ac->rrms_ffn, DIM, SEQ);
                    t1=mach_absolute_time(); t_rms_bwd+=tb_ms(t1-t0); t0=t1;
                    io_copy(kern[L].rmsBwd->ioIn, 0,   kern[L].ffnBwd->ioOut, 0, DIM, SEQ);
                    io_copy(kern[L].rmsBwd->ioIn, DIM, kern[L].fwdFFN->ioIn, 0, DIM, SEQ);
                    io_write_fp16_vec(kern[L].rmsBwd->ioIn, 2*DIM, lw[L].rms_ffn, DIM, SEQ);
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(kern[L].rmsBwd);
                    t1=mach_absolute_time(); t_bwd_ane+=tb_ms(t1-t0); t0=t1;
                    io_read_fp16(kern[L].rmsBwd->ioOut, dx2, 0, DIM, SEQ);
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0);
                    // Add residual: dx2 += dy (from skip connection)
                    t0=mach_absolute_time();
                    for(int i=0;i<SEQ*DIM;i++) dx2[i] += dy[i];
                    t1=mach_absolute_time(); t_resid+=tb_ms(t1-t0);

                    // dWo async (attn_out[slot] populated in forward)
                    t0=mach_absolute_time();
                    memcpy(cap->dx2[slot], dx2, SEQ*DIM*4);
                    t1=mach_absolute_time(); t_memcpy+=tb_ms(t1-t0);
                    dispatch_group_async(layer_dw_grp, dw_layer_q[L], ^{
                        uint64_t dw0=mach_absolute_time();
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                    1.0f, cap->dx2[slot], SEQ, cap->attn_out[slot], SEQ, 1.0f, gr->Wo, DIM);
                        dw_t_wo[L]+=tb_ms(mach_absolute_time()-dw0);
                    });

                    // SDPA backward (ANE)
                    t0=mach_absolute_time();
                    io_copy(kern[L].sdpaBwd1->ioIn, 0, kern[L].fwdAttn->ioOut, DIM, 3*DIM, SEQ);
                    io_write_fp16_at(kern[L].sdpaBwd1->ioIn, 3*DIM, dx2, DIM, SEQ);
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(kern[L].sdpaBwd1);
                    t1=mach_absolute_time(); t_bwd_ane+=tb_ms(t1-t0); t0=t1;
                    io_copy(sdpaBwd2[L]->ioIn, 0, kern[L].sdpaBwd1->ioOut, DIM, 2*SCORE_CH, SEQ);
                    io_copy(sdpaBwd2[L]->ioIn, 2*SCORE_CH, kern[L].fwdAttn->ioOut, DIM, 2*DIM, SEQ);
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(sdpaBwd2[L]);
                    t1=mach_absolute_time(); t_bwd_ane+=tb_ms(t1-t0); t0=t1;

                    io_read_fp16(sdpaBwd2[L]->ioOut, cap->dq[slot], 0,   DIM, SEQ);
                    io_read_fp16(sdpaBwd2[L]->ioOut, cap->dk[slot], DIM, DIM, SEQ);
                    io_read_fp16(kern[L].sdpaBwd1->ioOut, cap->dv[slot], 0, DIM, SEQ);
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0);

                    // dWq/dWk/dWv async (xnorm[slot] populated in forward)
                    dispatch_group_async(layer_dw_grp, dw_layer_q[L], ^{
                        uint64_t dw0=mach_absolute_time();
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                    1.0f, cap->dq[slot], SEQ, cap->xnorm[slot], SEQ, 1.0f, gr->Wq, DIM);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                    1.0f, cap->dk[slot], SEQ, cap->xnorm[slot], SEQ, 1.0f, gr->Wk, DIM);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                    1.0f, cap->dv[slot], SEQ, cap->xnorm[slot], SEQ, 1.0f, gr->Wv, DIM);
                        dw_t_qkv[L]+=tb_ms(mach_absolute_time()-dw0);
                        dispatch_semaphore_signal(cap->sem);
                    });

                    // QKV backward (ANE)
                    t0=mach_absolute_time();
                    io_copy(kern[L].qkvBwd->ioIn, 0, sdpaBwd2[L]->ioOut, 0, 2*DIM, SEQ);
                    io_copy(kern[L].qkvBwd->ioIn, 2*DIM, kern[L].sdpaBwd1->ioOut, 0, DIM, SEQ);
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(kern[L].qkvBwd);
                    t1=mach_absolute_time(); t_bwd_ane+=tb_ms(t1-t0); t0=t1;
                    io_read_fp16(kern[L].qkvBwd->ioOut, dx_attn, 0, DIM, SEQ);
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0);

                    // RMSNorm1 backward: dw on CPU (while fp32 dy is in dx_attn), dx on ANE
                    t0=mach_absolute_time();
                    rmsnorm_dw(gr->rms_att, dx_attn, ac->layer_in, ac->rrms_att, DIM, SEQ);
                    t1=mach_absolute_time(); t_rms_bwd+=tb_ms(t1-t0); t0=t1;
                    io_copy(kern[L].rmsBwd->ioIn, 0,   kern[L].qkvBwd->ioOut, 0, DIM, SEQ);
                    io_copy(kern[L].rmsBwd->ioIn, DIM, kern[L].fwdAttn->ioIn, 0, DIM, SEQ);
                    io_write_fp16_vec(kern[L].rmsBwd->ioIn, 2*DIM, lw[L].rms_att, DIM, SEQ);
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(kern[L].rmsBwd);
                    t1=mach_absolute_time(); t_bwd_ane+=tb_ms(t1-t0); t0=t1;
                    io_read_fp16(kern[L].rmsBwd->ioOut, dx_attn, 0, DIM, SEQ);
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0);

                    // dy for previous layer = dx_attn (through rmsnorm1) + dx2 (skip connection)
                    t0=mach_absolute_time();
                    for(int i=0;i<SEQ*DIM;i++) dy[i] = dx_attn[i] + dx2[i];
                    t1=mach_absolute_time(); t_resid+=tb_ms(t1-t0);
                }

                // Embedding backward — scatter-add into separate accumulator (no wait needed)
                t0=mach_absolute_time();
                embed_backward(gembed_emb, dy, input_tokens, DIM, SEQ);
                t1=mach_absolute_time(); t_embed_bwd+=tb_ms(t1-t0);

                steps_batch++;
                if (step % 10 == 0 || step == start_step)
                    printf("step %-4d loss=%.4f\n", step, loss);
            }
            double tms = tb_ms(mach_absolute_time() - tt);
            total_train_ms += tms;
            total_steps_done += steps_batch;
            total_batches++;

            // Ensure all async dW finished
            uint64_t tw0=mach_absolute_time();
            dispatch_group_wait(layer_dw_grp, DISPATCH_TIME_FOREVER);
            dispatch_group_wait(embed_dw_grp, DISPATCH_TIME_FOREVER);
            double t_final_dw_wait = tb_ms(mach_absolute_time()-tw0);

            // Adam update (scale gradients by 1/steps_batch)
            float gsc = 1.0f / steps_batch;
            adam_t++;
            for (int L=0; L<NLAYERS; L++) {
                LayerGrads *g = &grads[L];
                for(size_t i=0;i<WQ_SZ;i++){g->Wq[i]*=gsc;g->Wk[i]*=gsc;g->Wv[i]*=gsc;g->Wo[i]*=gsc;}
                for(size_t i=0;i<W1_SZ;i++) g->W1[i]*=gsc;
                for(size_t i=0;i<W2_SZ;i++) g->W2[i]*=gsc;
                for(size_t i=0;i<W3_SZ;i++) g->W3[i]*=gsc;
                for(int i=0;i<DIM;i++){g->rms_att[i]*=gsc; g->rms_ffn[i]*=gsc;}

                adam_update(lw[L].Wq, g->Wq, &la[L].Wq, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update(lw[L].Wk, g->Wk, &la[L].Wk, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update(lw[L].Wv, g->Wv, &la[L].Wv, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update(lw[L].Wo, g->Wo, &la[L].Wo, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update(lw[L].W1, g->W1, &la[L].W1, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update(lw[L].W2, g->W2, &la[L].W2, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update(lw[L].W3, g->W3, &la[L].W3, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update(lw[L].rms_att, g->rms_att, &la[L].rms_att, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update(lw[L].rms_ffn, g->rms_ffn, &la[L].rms_ffn, adam_t, lr, adam_b1, adam_b2, adam_eps);
            }
            for(int i=0;i<DIM;i++) grms_final[i]*=gsc;
            adam_update(rms_final, grms_final, &arms_final, adam_t, lr, adam_b1, adam_b2, adam_eps);
            // Merge embed accumulators (cls dense + emb scatter), scale, and update
            for(size_t i=0;i<(size_t)VOCAB*DIM;i++) gembed_cls[i] = (gembed_cls[i] + gembed_emb[i]) * gsc;
            adam_update(embed, gembed_cls, &aembed, adam_t, lr, adam_b1, adam_b2, adam_eps);

            printf("  [batch %d: compile=%.0fms train=%.1fms (%.1fms/step) compiles=%d]\n",
                   steps_batch, cms, tms, tms/steps_batch, g_compile_count);
            double t_elem_total = t_embed+t_resid+t_xent+t_memcpy+t_rms_bwd+t_embed_bwd;
            printf("    fwd: ane=%.1f io=%.1f cls=%.1f rms_fwd=%.1f ms/step\n",
                   t_ane/steps_batch, t_io/steps_batch, t_cls/steps_batch,
                   t_rms/steps_batch);
            printf("    bwd: ane=%.1f io=%.1f cls_bwd=%.1f ms/step\n",
                   t_bwd_ane/steps_batch, t_bwd_io/steps_batch, t_cls_bwd/steps_batch);
            printf("    elem=%.1f [xent=%.1f memcpy=%.1f rms_bwd=%.1f resid=%.1f embed=%.1f embed_bwd=%.1f]\n",
                   t_elem_total/steps_batch, t_xent/steps_batch, t_memcpy/steps_batch,
                   t_rms_bwd/steps_batch, t_resid/steps_batch, t_embed/steps_batch,
                   t_embed_bwd/steps_batch);
            // dW instrumentation report
            double tot_ffn=0,tot_wo=0,tot_qkv=0;
            for(int L=0;L<NLAYERS;L++){tot_ffn+=dw_t_ffn[L];tot_wo+=dw_t_wo[L];tot_qkv+=dw_t_qkv[L];}
            printf("    dW sgemm: ffn=%.1f wo=%.1f qkv=%.1f embed=%.1f total=%.1f ms/step\n",
                   tot_ffn/steps_batch, tot_wo/steps_batch, tot_qkv/steps_batch,
                   dw_t_embed/steps_batch, (tot_ffn+tot_wo+tot_qkv+dw_t_embed)/steps_batch);
            printf("    sem_wait=%.1f final_dw_wait=%.1f ms/step\n",
                   t_sem_wait/steps_batch, t_final_dw_wait/steps_batch);
            // Per-layer semaphore wait breakdown (find the bottleneck layer)
            printf("    sem/layer:");
            for(int L=0;L<NLAYERS;L++) printf(" L%d=%.1f",L,sem_wait_layer[L]/steps_batch);
            printf("\n");
            free(dw_t_ffn); free(dw_t_wo); free(dw_t_qkv);
        }

        // Efficiency report
        double wall = tb_ms(mach_absolute_time() - t_wall_start);
        total_compile_ms += cum_compile; total_train_ms += cum_train;
        wall += cum_wall; total_steps_done += cum_steps; total_batches += cum_batches;
        double fwd_flops = NLAYERS * (4.0*2*DIM*DIM*SEQ + 2.0*2*DIM*HIDDEN*SEQ + 2.0*HIDDEN*DIM*SEQ);
        double sdpa_flops = NLAYERS * 2.0*HEADS*5*SEQ*SEQ*HD;
        double cls_flops = 2.0*VOCAB*DIM*SEQ;
        double total_flops = (fwd_flops*3 + sdpa_flops + cls_flops*3) * total_steps_done;
        double ane_cls_flops = use_ane_cls ? cls_flops*2 : 0;  // fwd + bwd_dx on ANE
        double ane_flops = (fwd_flops*2 + sdpa_flops + ane_cls_flops) * total_steps_done;
        printf("\n=== Efficiency Report ===\n");
        printf("Total steps:     %d\n", total_steps_done);
        printf("Wall time:       %.0f ms (%.1f s)\n", wall, wall/1000);
        printf("Compile time:    %.0f ms (%.1f%%)\n", total_compile_ms, 100*total_compile_ms/wall);
        printf("Train time:      %.0f ms (%.1f%%)\n", total_train_ms, 100*total_train_ms/wall);
        printf("Avg train:       %.1f ms/step\n", total_train_ms/total_steps_done);
        printf("ANE TFLOPS:      %.2f sustained\n", ane_flops / (total_train_ms * 1e9));
        printf("Total TFLOPS:    %.2f (ANE+CPU)\n", total_flops / (total_train_ms * 1e9));
        printf("ANE utilization: %.1f%% of 15.8 TFLOPS\n", 100*ane_flops/(total_train_ms*1e9)/15.8);

        // Cleanup
        free_kern(cls_fwd); free_kern(cls_bwd);
        free_kern(softmax_kern);
        free_kern(rmsBwdFinal);
        for (int L=0; L<NLAYERS; L++) {
            free_layer_kernels(&kern[L]);
            free_kern(sdpaBwd2[L]);
            free_kern(kern[L].rmsBwd);
            layer_weights_free(&lw[L]);
            layer_adam_free(&la[L]);
            layer_acts_free(&acts[L]);
            layer_grads_free(&grads[L]);
            layer_dwcap_free(&dwcap[L]);
        }
        munmap(token_data, data_len);
        close(data_fd);
        free(rms_final); free(rrms_final); free(embed); free(grms_final);
        free(gembed_cls); free(gembed_emb);
        for (int s=0; s<2; s++) { free(capt_dlogits[s]); free(capt_xfinal[s]); }
        adam_free(&arms_final); adam_free(&aembed);
        free(dy); free(dx_ffn); free(dx2); free(dx_attn);
        free(x_cur); free(x_final); free(logits); free(dlogits);
    }
    return 0;
}
