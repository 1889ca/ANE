// train_large.m — Train stories110M (12 layers, 768dim, 3072hidden) on ANE
// Uses pretokenized TinyStories data with cross-entropy loss
// 6 weight-bearing ANE kernels per layer × 12 layers + 2 classifier = 74 per compile batch
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
    lk->qkvFwd = compile_kern_mil_w(gen_qkv_fwd_taps(), (@{
        @"@model_path/weights/rms1.bin": @{@"offset":@0, @"data":build_blob(w->rms_att,1,DIM)},
        @"@model_path/weights/wq.bin": @{@"offset":@0, @"data":build_blob(w->Wq,DIM,DIM)},
        @"@model_path/weights/wk.bin": @{@"offset":@0, @"data":build_blob(w->Wk,DIM,DIM)},
        @"@model_path/weights/wv.bin": @{@"offset":@0, @"data":build_blob(w->Wv,DIM,DIM)},
    }), DIM*SEQ*2, (4*DIM+1)*SEQ*2);

    lk->attnFwd = compile_kern_mil_w(gen_attn_fwd(), (@{
        @"@model_path/weights/wo.bin": @{@"offset":@0, @"data":build_blob(w->Wo,DIM,DIM)},
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
    }), 4*DIM*SEQ*2, 2*DIM*SEQ*2);

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

    return lk->qkvFwd && lk->attnFwd && lk->fwdFFN && lk->ffnBwd && lk->sdpaBwd1 && lk->qkvBwd;
}

// Compile weight-free rmsBwd (one per layer, no weights)
static Kern *compile_rms_bwd(void) {
    return compile_kern_mil_w(gen_rms_bwd(), @{},
        3*DIM*SEQ*2, DIM*SEQ*2);
}
// Compile weight-free rmsBwd with fused residual add
static Kern *compile_rms_bwd_resid(void) {
    return compile_kern_mil_w(gen_rms_bwd_resid(), @{},
        4*DIM*SEQ*2, DIM*SEQ*2);
}

// Compile weight-free sdpaBwd2 (only needs once, no weights)
static Kern *compile_sdpa_bwd2(void) {
    return compile_kern_mil_w(gen_sdpa_bwd2(), @{},
        (2*SCORE_CH+2*DIM)*SEQ*2, 2*DIM*SEQ*2);
}

static void free_layer_kernels(LayerKernels *lk) {
    free_kern(lk->qkvFwd); free_kern(lk->attnFwd);
    free_kern(lk->fwdFFN); free_kern(lk->ffnBwd);
    free_kern(lk->sdpaBwd1); free_kern(lk->qkvBwd);
    // sdpaBwd2, rmsBwd1, rmsBwd2 are shared/static, freed separately
    lk->qkvFwd = lk->attnFwd = lk->fwdFFN = lk->ffnBwd = lk->sdpaBwd1 = lk->qkvBwd = NULL;
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
        bool neon_rope = false;  // set after arg parsing

        int total_steps = 10000;
        float lr = 3e-4f, lr_min = 0.0f;
        int warmup_steps = 0;
        float adam_b1=0.9f, adam_b2=0.999f, adam_eps=1e-8f;
        int adam_t = 0, start_step = 0;
        int accum_steps = DEFAULT_ACCUM_STEPS;
        int max_compiles = DEFAULT_MAX_COMPILES;

        // Parse args
        bool do_resume = false;
        bool use_cpu_rope = false;
        const char *data_path = DATA_PATH;
        const char *ckpt_path = CKPT_PATH;
        int freeze_below = 0;  // freeze layers 0..freeze_below-1 (no weight update)
        for (int i=1; i<argc; i++) {
            if (strcmp(argv[i], "--resume") == 0) do_resume = true;
            else if (strcmp(argv[i], "--cpu-rope") == 0) use_cpu_rope = true;
            else if (strcmp(argv[i], "--steps") == 0 && i+1<argc) total_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--lr") == 0 && i+1<argc) lr = atof(argv[++i]);
            else if (strcmp(argv[i], "--accum") == 0 && i+1<argc) accum_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--lr-min") == 0 && i+1<argc) lr_min = atof(argv[++i]);
            else if (strcmp(argv[i], "--warmup") == 0 && i+1<argc) warmup_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--max-compiles") == 0 && i+1<argc) max_compiles = atoi(argv[++i]);
            else if (strcmp(argv[i], "--data") == 0 && i+1<argc) data_path = argv[++i];
            else if (strcmp(argv[i], "--ckpt") == 0 && i+1<argc) ckpt_path = argv[++i];
            else if (strcmp(argv[i], "--freeze") == 0 && i+1<argc) freeze_below = atoi(argv[++i]);
        }

        // Initialize NEON fp16 RoPE table (unless --cpu-rope fallback)
        if (!use_cpu_rope) {
            rope_init_table();
            neon_rope = true;
            printf("  [rope] Using NEON fp16 RoPE\n");
        } else {
            printf("  [rope] Using CPU fp32 RoPE (--cpu-rope)\n");
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
        // Double-buffered dlogits/x_final capture
        _Float16 *capt_dlogits_f16[2];  // fp16 capture from softmax IOSurface
        float *capt_dlogits[2], *capt_xfinal[2];
        for (int s=0; s<2; s++) {
            capt_dlogits_f16[s] = (_Float16*)malloc((size_t)SEQ*VOCAB*2);
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
            resuming = load_checkpoint(ckpt_path, &start_step, &total_steps, &lr, &resume_loss,
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
            printf("Accum %d steps per recompile | Adam LR=%.1e", accum_steps, lr);
            if (warmup_steps > 0) printf(" warmup=%d", warmup_steps);
            if (lr_min > 0) printf(" lr_min=%.1e (cosine)", lr_min);
            printf(" b1=%.1f b2=%.3f\n", adam_b1, adam_b2);
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
        int data_fd = open(data_path, O_RDONLY);
        if (data_fd < 0) { printf("Cannot open %s\n", data_path); return 1; }
        struct stat st; fstat(data_fd, &st);
        size_t data_len = st.st_size;
        uint16_t *token_data = (uint16_t*)mmap(NULL, data_len, PROT_READ, MAP_PRIVATE, data_fd, 0);
        if (token_data == MAP_FAILED) { printf("mmap failed\n"); return 1; }
        size_t n_tokens = data_len / 2;
        if (n_tokens <= (size_t)(SEQ + 1)) {
            printf("Token data too short: need at least %d tokens, got %zu\n", SEQ + 2, n_tokens);
            munmap(token_data, data_len);
            close(data_fd);
            return 1;
        }
        printf("Token data: %zu tokens (%.1f MB)\n", n_tokens, data_len/1e6);

        // Gradient buffers shared across layers (reused each step)
        float *dy = (float*)malloc(SEQ*DIM*4);            // gradient flowing backward
        float *dx_ffn = (float*)malloc(SEQ*DIM*4);
        float *dx2 = (float*)malloc(SEQ*DIM*4);
        float *dx_attn = (float*)malloc(SEQ*DIM*4);

        // x buffer for input to each layer (channel-first [DIM, SEQ])
        float *x_cur = (float*)malloc(SEQ*DIM*4);
        float *x_final = (float*)malloc(SEQ*DIM*4);     // after final rmsnorm
        // RoPE temp buffers: fp32 Q,K for in-place rotation (channel-first [DIM, SEQ])
        float *rope_q = (float*)malloc(SEQ*DIM*4);
        float *rope_k = (float*)malloc(SEQ*DIM*4);
        float *logits = (float*)malloc(SEQ*VOCAB*4);     // [VOCAB, SEQ] for cross-entropy
        float *dlogits = (float*)malloc(SEQ*VOCAB*4);

        // Compile static sdpaBwd2 kernels (no weights, one per layer)
        Kern *sdpaBwd2[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            sdpaBwd2[L] = compile_sdpa_bwd2();
            if (!sdpaBwd2[L]) { printf("sdpaBwd2 compile failed\n"); return 1; }
        }

        // Compile rmsBwd kernels: resid variant per layer (fused residual add), plain for final
        for (int L=0; L<NLAYERS; L++) {
            kern[L].rmsBwd2 = compile_rms_bwd_resid();
            kern[L].rmsBwd1 = compile_rms_bwd_resid();
            if (!kern[L].rmsBwd2 || !kern[L].rmsBwd1) { printf("rmsBwd compile failed at layer %d\n", L); return 1; }
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
                for (int L=0; L<NLAYERS; L++) { free_layer_kernels(&kern[L]); free_kern(sdpaBwd2[L]); free_kern(kern[L].rmsBwd2); kern[L].rmsBwd2 = NULL; free_kern(kern[L].rmsBwd1); kern[L].rmsBwd1 = NULL; }
                free_kern(rmsBwdFinal); rmsBwdFinal = NULL;
                free_kern(cls_fwd); free_kern(cls_bwd); cls_fwd = cls_bwd = NULL;
                free_kern(softmax_kern); softmax_kern = NULL;
                double wall = tb_ms(mach_absolute_time() - t_wall_start);
                save_checkpoint(ckpt_path, step, total_steps, lr, last_loss,
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
                if (!kern[L].rmsBwd2) {
                    kern[L].rmsBwd2 = compile_rms_bwd_resid();
                    if (!kern[L].rmsBwd2) { printf("rmsBwd2 recompile failed\n"); return 1; }
                }
                if (!kern[L].rmsBwd1) {
                    kern[L].rmsBwd1 = compile_rms_bwd_resid();
                    if (!kern[L].rmsBwd1) { printf("rmsBwd1 recompile failed\n"); return 1; }
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
            // cls_bwd uses tiled 4×8K convs to avoid ANE pathology with 32K input channels
            if (use_ane_cls) {
                free_kern(cls_fwd); free_kern(cls_bwd);
                cls_fwd = compile_kern_mil_w(gen_cls_fwd(), (@{
                    @"@model_path/weights/embed.bin": @{@"offset":@0, @"data":build_blob(embed, VOCAB, DIM)},
                }), DIM*SEQ*2, VOCAB*SEQ*2);
                int chunk = VOCAB / 4;
                cls_bwd = compile_kern_mil_w(gen_cls_bwd_tiled(), (@{
                    @"@model_path/weights/wt0.bin": @{@"offset":@0, @"data":build_blob_t(embed + (size_t)0*chunk*DIM, chunk, DIM)},
                    @"@model_path/weights/wt1.bin": @{@"offset":@0, @"data":build_blob_t(embed + (size_t)1*chunk*DIM, chunk, DIM)},
                    @"@model_path/weights/wt2.bin": @{@"offset":@0, @"data":build_blob_t(embed + (size_t)2*chunk*DIM, chunk, DIM)},
                    @"@model_path/weights/wt3.bin": @{@"offset":@0, @"data":build_blob_t(embed + (size_t)3*chunk*DIM, chunk, DIM)},
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
                    // QKV forward: x_cur → Q,K,V,xnorm,rrms (RMSNorm + QKV projection on ANE)
                    t0=mach_absolute_time();
                    io_write_fp16(kern[L].qkvFwd->ioIn, x_cur, DIM, SEQ);
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(kern[L].qkvFwd);
                    t1=mach_absolute_time(); t_ane+=tb_ms(t1-t0); t0=t1;
                    if (neon_rope) {
                    // NEON fp16 path: single-pass RoPE + V copy + xnorm/rrms read
                    { _Float16 *attn_in = io_lock_rw(kern[L].attnFwd->ioIn);
                    const _Float16 *qkv_p = io_lock_ro(kern[L].qkvFwd->ioOut);
                    neon_rope_fwd_f16(qkv_p, attn_in, 0, DIM*SEQ, 0, DIM*SEQ);
                    memcpy(attn_in + 2*DIM*SEQ, qkv_p + 2*DIM*SEQ, DIM * SEQ * sizeof(_Float16));
                    memcpy(dwcap[L].xnorm_f16[slot],     qkv_p + 3*DIM*SEQ, DIM * SEQ * sizeof(_Float16));
                    cvt_f16_f32(ac->rrms_att,            qkv_p + 4*DIM*SEQ, 1 * SEQ);
                    io_unlock_ro(kern[L].qkvFwd->ioOut);
                    cvt_f32_f16(attn_in + 3*DIM*SEQ,   x_cur,   DIM * SEQ);
                    io_unlock_rw(kern[L].attnFwd->ioIn);
                    }
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;
                    } else {
                    // CPU fp32 fallback (--cpu-rope)
                    { const _Float16 *qkv_p = io_lock_ro(kern[L].qkvFwd->ioOut);
                    cvt_f16_f32(rope_q,                  qkv_p,              DIM * SEQ);
                    cvt_f16_f32(rope_k,                  qkv_p + DIM*SEQ,    DIM * SEQ);
                    memcpy(dwcap[L].xnorm_f16[slot],     qkv_p + 3*DIM*SEQ, DIM * SEQ * sizeof(_Float16));
                    cvt_f16_f32(ac->rrms_att,            qkv_p + 4*DIM*SEQ, 1 * SEQ);
                    io_unlock_ro(kern[L].qkvFwd->ioOut);
                    }
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;
                    cpu_rope_cf(rope_q, rope_k, SEQ, HEADS, HD);
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;
                    { _Float16 *attn_in = io_lock_rw(kern[L].attnFwd->ioIn);
                    const _Float16 *qkv_p = io_lock_ro(kern[L].qkvFwd->ioOut);
                    cvt_f32_f16(attn_in,               rope_q,  DIM * SEQ);
                    cvt_f32_f16(attn_in + DIM*SEQ,     rope_k,  DIM * SEQ);
                    memcpy(attn_in + 2*DIM*SEQ, qkv_p + 2*DIM*SEQ, DIM * SEQ * sizeof(_Float16));
                    io_unlock_ro(kern[L].qkvFwd->ioOut);
                    cvt_f32_f16(attn_in + 3*DIM*SEQ,   x_cur,   DIM * SEQ);
                    io_unlock_rw(kern[L].attnFwd->ioIn);
                    }
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;
                    }
                    ane_eval(kern[L].attnFwd);
                    t1=mach_absolute_time(); t_ane+=tb_ms(t1-t0); t0=t1;
                    { // Read attnFwd->ioOut: x2→fwdFFN->ioIn + ac->x2, attn_out→dwcap
                    const _Float16 *attn_out_p = io_lock_ro(kern[L].attnFwd->ioOut);
                    _Float16 *ffn_in_p = io_lock_rw(kern[L].fwdFFN->ioIn);
                    memcpy(ffn_in_p, attn_out_p, DIM * SEQ * sizeof(_Float16));
                    io_unlock_rw(kern[L].fwdFFN->ioIn);
                    cvt_f16_f32(ac->x2,                  attn_out_p,              DIM * SEQ);
                    memcpy(dwcap[L].attn_f16[slot],      attn_out_p + DIM*SEQ,    DIM * SEQ * sizeof(_Float16));
                    io_unlock_ro(kern[L].attnFwd->ioOut);
                    }
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;

                    // FFN forward (x2 already piped via memcpy above)
                    ane_eval(kern[L].fwdFFN);
                    t1=mach_absolute_time(); t_ane+=tb_ms(t1-t0); t0=t1;
                    { // Batch: fwdFFN->ioOut — defer dW captures as fp16 memcpy
                    const _Float16 *ffn_p = io_lock_ro(kern[L].fwdFFN->ioOut);
                    cvt_f16_f32(x_cur,                   ffn_p,                        DIM * SEQ);
                    memcpy(dwcap[L].silu_f16[slot],      ffn_p + (DIM+2*HIDDEN)*SEQ,   HIDDEN * SEQ * sizeof(_Float16));
                    memcpy(dwcap[L].x2norm_f16[slot],    ffn_p + (DIM+3*HIDDEN)*SEQ,   DIM * SEQ * sizeof(_Float16));
                    cvt_f16_f32(ac->rrms_ffn,            ffn_p + (2*DIM+3*HIDDEN)*SEQ,  1 * SEQ);
                    io_unlock_ro(kern[L].fwdFFN->ioOut);
                    }
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
                    // Compute loss + gradient in fp16 directly on softmax output
                    // Combined scale: (1/SEQ) * LOSS_SCALE = 1.0, so dlogits = probs with target -= 1
                    _Float16 *probs = io_lock_rw(softmax_kern->ioOut);
                    float total_loss = 0;
                    for (int t = 0; t < SEQ; t++) {
                        size_t idx = (size_t)target_tokens[t]*SEQ + t;
                        total_loss -= logf((float)probs[idx] + 1e-10f);
                        probs[idx] -= (_Float16)1.0f;
                    }
                    loss = total_loss / SEQ;
                    // Copy fp16 dlogits to cls_bwd input + embed dW capture
                    _Float16 *cls_in = io_lock_rw(cls_bwd->ioIn);
                    memcpy(cls_in, probs, (size_t)VOCAB * SEQ * sizeof(_Float16));
                    io_unlock_rw(cls_bwd->ioIn);
                    memcpy(capt_dlogits_f16[slot], probs, (size_t)VOCAB * SEQ * sizeof(_Float16));
                    io_unlock_rw(softmax_kern->ioOut);
                } else {
                    if (use_ane_cls)
                        io_read_fp16(cls_fwd->ioOut, logits, 0, VOCAB, SEQ);
                    loss = cross_entropy_loss(dlogits, logits, target_tokens, VOCAB, SEQ);
                    float ls = LOSS_SCALE;
                    vDSP_vsmul(dlogits, 1, &ls, dlogits, 1, (vDSP_Length)((size_t)VOCAB*SEQ));
                    // Fallback: copy fp32 dlogits to fp16 capture
                    cvt_f32_f16(capt_dlogits_f16[slot], dlogits, (int)((size_t)VOCAB*SEQ));
                }
                last_loss = loss;
                t1=mach_absolute_time(); t_xent+=tb_ms(t1-t0); t0=t1;

                // ===== BACKWARD =====
                // Classifier backward: dx_final = embed^T @ dlogits
                t0=mach_absolute_time();
                if (cls_bwd) {
                    if (!(use_ane_softmax && use_ane_cls)) {
                        // Non-ANE softmax path: write fp32 dlogits to cls_bwd
                        io_write_fp16(cls_bwd->ioIn, dlogits, VOCAB, SEQ);
                    }
                    // ANE softmax path: cls_bwd->ioIn already populated above
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(cls_bwd);
                    t1=mach_absolute_time(); t_cls_bwd+=tb_ms(t1-t0); t0=t1;
                    io_read_fp16(cls_bwd->ioOut, dy, 0, DIM, SEQ);
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0);
                } else {
                    // CPU cls_bwd (mode 2 or ANE fallback)
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                DIM, SEQ, VOCAB, 1.0f,
                                embed, DIM, dlogits, SEQ, 0.0f, dy, SEQ);
                    t1=mach_absolute_time(); t_cls_bwd+=tb_ms(t1-t0);
                }

                // dembed_cls[VOCAB,DIM] += dlogits[VOCAB,SEQ] @ x_final^T[SEQ,DIM]
                // fp16 dlogits captured above, converted in async path
                float *cx = capt_xfinal[slot];
                memcpy(cx, x_final, SEQ*DIM*4);
                float *cd = capt_dlogits[slot];
                _Float16 *cd16 = capt_dlogits_f16[slot];
                dispatch_group_async(embed_dw_grp, dw_embed_q, ^{
                    uint64_t dw0=mach_absolute_time();
                    cvt_f16_f32(cd, cd16, (int)((size_t)VOCAB * SEQ));
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
                if (cls_bwd)
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

                    // dy handoff: first layer from fp32, subsequent from fused rmsBwd1 output
                    t0=mach_absolute_time();

                    // FFN backward (ANE)
                    { _Float16 *bwd_in = io_lock_rw(kern[L].ffnBwd->ioIn);
                    const _Float16 *fwd_out = io_lock_ro(kern[L].fwdFFN->ioOut);
                    if (L == NLAYERS-1) {
                        // First backward layer: dy is fp32 from rmsBwdFinal
                        memcpy(cap->dffn[slot], dy, SEQ*DIM*4);
                        cvt_f32_f16(bwd_in, cap->dffn[slot], DIM * SEQ);
                    } else {
                        // Subsequent layers: dy is fp16 on rmsBwd1[L+1]->ioOut
                        const _Float16 *prev_dy = io_lock_ro(kern[L+1].rmsBwd1->ioOut);
                        memcpy(bwd_in, prev_dy, DIM * SEQ * sizeof(_Float16));
                        memcpy(cap->dffn_f16[slot], prev_dy, DIM * SEQ * sizeof(_Float16));
                        io_unlock_ro(kern[L+1].rmsBwd1->ioOut);
                    }
                    memcpy(bwd_in + DIM*SEQ, fwd_out + DIM*SEQ, 2*HIDDEN * SEQ * sizeof(_Float16));
                    io_unlock_ro(kern[L].fwdFFN->ioOut);
                    io_unlock_rw(kern[L].ffnBwd->ioIn);
                    }
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(kern[L].ffnBwd);
                    t1=mach_absolute_time(); t_bwd_ane+=tb_ms(t1-t0); t0=t1;
                    { // Batch: ffnBwd->ioOut — defer dh1/dh3 as fp16 memcpy
                    const _Float16 *bwd_out = io_lock_ro(kern[L].ffnBwd->ioOut);
                    cvt_f16_f32(dx_ffn,         bwd_out,                      DIM * SEQ);
                    memcpy(cap->dh1_f16[slot],  bwd_out + DIM*SEQ,            HIDDEN * SEQ * sizeof(_Float16));
                    memcpy(cap->dh3_f16[slot],  bwd_out + (DIM+HIDDEN)*SEQ,   HIDDEN * SEQ * sizeof(_Float16));
                    io_unlock_ro(kern[L].ffnBwd->ioOut);
                    }
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0);

                    // dW FFN async — deferred fp16→fp32 conversion + sgemm
                    dispatch_group_async(layer_dw_grp, dw_layer_q[L], ^{
                        uint64_t dw0=mach_absolute_time();
                        if (L < NLAYERS-1)
                            cvt_f16_f32(cap->dffn[slot], cap->dffn_f16[slot], DIM * SEQ);
                        cvt_f16_f32(cap->silu_out[slot], cap->silu_f16[slot], HIDDEN * SEQ);
                        cvt_f16_f32(cap->x2norm[slot],   cap->x2norm_f16[slot], DIM * SEQ);
                        cvt_f16_f32(cap->dh1[slot],      cap->dh1_f16[slot], HIDDEN * SEQ);
                        cvt_f16_f32(cap->dh3[slot],      cap->dh3_f16[slot], HIDDEN * SEQ);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, HIDDEN, SEQ,
                                    1.0f, cap->dffn[slot], SEQ, cap->silu_out[slot], SEQ, 1.0f, gr->W2, HIDDEN);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                    1.0f, cap->dh1[slot], SEQ, cap->x2norm[slot], SEQ, 1.0f, gr->W1, DIM);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                    1.0f, cap->dh3[slot], SEQ, cap->x2norm[slot], SEQ, 1.0f, gr->W3, DIM);
                        dw_t_ffn[L]+=tb_ms(mach_absolute_time()-dw0);
                    });

                    // RMSNorm2 backward: dw on CPU, dx+resid fused on ANE (rmsBwd2)
                    t0=mach_absolute_time();
                    rmsnorm_dw(gr->rms_ffn, dx_ffn, ac->x2, ac->rrms_ffn, DIM, SEQ);
                    t1=mach_absolute_time(); t_rms_bwd+=tb_ms(t1-t0); t0=t1;
                    { // Pack rmsBwd2: dx_ffn, x, w, dy_skip (fused residual add)
                    _Float16 *rms_in = io_lock_rw(kern[L].rmsBwd2->ioIn);
                    const _Float16 *ffn_bwd_p = io_lock_ro(kern[L].ffnBwd->ioOut);
                    const _Float16 *ffn_fwd_p = io_lock_ro(kern[L].fwdFFN->ioIn);
                    const _Float16 *ffn_bwd_in = io_lock_ro(kern[L].ffnBwd->ioIn);
                    memcpy(rms_in, ffn_bwd_p, DIM * SEQ * sizeof(_Float16));
                    memcpy(rms_in + DIM*SEQ, ffn_fwd_p, DIM * SEQ * sizeof(_Float16));
                    for (int c = 0; c < DIM; c++)
                        rms_in[(2*DIM + c) * SEQ] = (_Float16)lw[L].rms_ffn[c];
                    memcpy(rms_in + 3*DIM*SEQ, ffn_bwd_in, DIM * SEQ * sizeof(_Float16));
                    io_unlock_ro(kern[L].ffnBwd->ioIn);
                    io_unlock_ro(kern[L].fwdFFN->ioIn);
                    io_unlock_ro(kern[L].ffnBwd->ioOut);
                    io_unlock_rw(kern[L].rmsBwd2->ioIn);
                    }
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(kern[L].rmsBwd2);
                    t1=mach_absolute_time(); t_bwd_ane+=tb_ms(t1-t0); t0=t1;

                    // dWo async — dx2 captured as fp16 from fused rmsBwd2 output
                    { const _Float16 *rms_out = io_lock_ro(kern[L].rmsBwd2->ioOut);
                    memcpy(cap->dx2_f16[slot], rms_out, DIM * SEQ * sizeof(_Float16));
                    io_unlock_ro(kern[L].rmsBwd2->ioOut);
                    }
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0);
                    dispatch_group_async(layer_dw_grp, dw_layer_q[L], ^{
                        uint64_t dw0=mach_absolute_time();
                        cvt_f16_f32(cap->dx2[slot],  cap->dx2_f16[slot], DIM * SEQ);
                        cvt_f16_f32(cap->attn_out[slot], cap->attn_f16[slot], DIM * SEQ);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                    1.0f, cap->dx2[slot], SEQ, cap->attn_out[slot], SEQ, 1.0f, gr->Wo, DIM);
                        dw_t_wo[L]+=tb_ms(mach_absolute_time()-dw0);
                    });

                    // SDPA backward (ANE) — dx2 from fused rmsBwd2 output (fp16 direct)
                    t0=mach_absolute_time();
                    { // Batch: sdpaBwd1->ioIn + attnFwd->ioIn (Q,K) + qkvFwd->ioOut (V) + rmsBwd2->ioOut (dx2)
                    _Float16 *bwd1_in = io_lock_rw(kern[L].sdpaBwd1->ioIn);
                    const _Float16 *attn_in = io_lock_ro(kern[L].attnFwd->ioIn);
                    const _Float16 *qkv_out = io_lock_ro(kern[L].qkvFwd->ioOut);
                    const _Float16 *rms_out = io_lock_ro(kern[L].rmsBwd2->ioOut);
                    memcpy(bwd1_in,            attn_in,                 2*DIM * SEQ * sizeof(_Float16));
                    memcpy(bwd1_in + 2*DIM*SEQ, qkv_out + 2*DIM*SEQ,  DIM * SEQ * sizeof(_Float16));
                    memcpy(bwd1_in + 3*DIM*SEQ, rms_out,               DIM * SEQ * sizeof(_Float16));
                    io_unlock_ro(kern[L].rmsBwd2->ioOut);
                    io_unlock_ro(kern[L].qkvFwd->ioOut);
                    io_unlock_ro(kern[L].attnFwd->ioIn);
                    io_unlock_rw(kern[L].sdpaBwd1->ioIn);
                    }
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(kern[L].sdpaBwd1);
                    t1=mach_absolute_time(); t_bwd_ane+=tb_ms(t1-t0); t0=t1;
                    { // Batch: sdpaBwd2->ioIn + sdpaBwd1->ioOut (probs,dp) + attnFwd->ioIn (RoPE'd Q,K)
                    _Float16 *bwd2_in = io_lock_rw(sdpaBwd2[L]->ioIn);
                    const _Float16 *bwd1_p = io_lock_ro(kern[L].sdpaBwd1->ioOut);
                    const _Float16 *attn_in = io_lock_ro(kern[L].attnFwd->ioIn);
                    memcpy(bwd2_in, bwd1_p + DIM*SEQ, 2*SCORE_CH * SEQ * sizeof(_Float16));
                    memcpy(bwd2_in + 2*SCORE_CH*SEQ, attn_in, 2*DIM * SEQ * sizeof(_Float16));
                    io_unlock_ro(kern[L].attnFwd->ioIn);
                    io_unlock_ro(kern[L].sdpaBwd1->ioOut);
                    io_unlock_rw(sdpaBwd2[L]->ioIn);
                    }
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(sdpaBwd2[L]);
                    t1=mach_absolute_time(); t_bwd_ane+=tb_ms(t1-t0); t0=t1;

                    if (neon_rope) {
                    // NEON fp16 path: single-pass inverse RoPE → fp16 (qkvBwd) + fp32 (dW sgemm)
                    // dV folded into same lock (eliminates extra lock/unlock pair)
                    { _Float16 *qkv_in = io_lock_rw(kern[L].qkvBwd->ioIn);
                    const _Float16 *bwd2_p = io_lock_ro(sdpaBwd2[L]->ioOut);
                    const _Float16 *bwd1_p = io_lock_ro(kern[L].sdpaBwd1->ioOut);
                    neon_rope_bwd_f16(bwd2_p, qkv_in, cap->dq[slot], cap->dk[slot],
                                       0, DIM*SEQ, 0, DIM*SEQ);
                    memcpy(qkv_in + 2*DIM*SEQ, bwd1_p, DIM * SEQ * sizeof(_Float16));
                    memcpy(cap->dv_f16[slot],  bwd1_p, DIM * SEQ * sizeof(_Float16));
                    io_unlock_ro(kern[L].sdpaBwd1->ioOut);
                    io_unlock_ro(sdpaBwd2[L]->ioOut);
                    io_unlock_rw(kern[L].qkvBwd->ioIn);
                    }
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0);
                    } else {
                    // CPU fallback: read dQ,dK → rotate → write back
                    { const _Float16 *bwd2_p = io_lock_ro(sdpaBwd2[L]->ioOut);
                    cvt_f16_f32(cap->dq[slot], bwd2_p,            DIM * SEQ);
                    cvt_f16_f32(cap->dk[slot], bwd2_p + DIM*SEQ,  DIM * SEQ);
                    io_unlock_ro(sdpaBwd2[L]->ioOut);
                    }
                    io_read_fp16(kern[L].sdpaBwd1->ioOut, cap->dv[slot], 0, DIM, SEQ);
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0);
                    t0=mach_absolute_time();
                    cpu_rope_backward_cf(cap->dq[slot], cap->dk[slot], SEQ, HEADS, HD);
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0);
                    }

                    // dWq/dWk/dWv async — deferred fp16→fp32 conversion + sgemm
                    dispatch_group_async(layer_dw_grp, dw_layer_q[L], ^{
                        uint64_t dw0=mach_absolute_time();
                        cvt_f16_f32(cap->xnorm[slot], cap->xnorm_f16[slot], DIM * SEQ);
                        if (neon_rope)
                            cvt_f16_f32(cap->dv[slot], cap->dv_f16[slot], DIM * SEQ);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                    1.0f, cap->dq[slot], SEQ, cap->xnorm[slot], SEQ, 1.0f, gr->Wq, DIM);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                    1.0f, cap->dk[slot], SEQ, cap->xnorm[slot], SEQ, 1.0f, gr->Wk, DIM);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                    1.0f, cap->dv[slot], SEQ, cap->xnorm[slot], SEQ, 1.0f, gr->Wv, DIM);
                        dw_t_qkv[L]+=tb_ms(mach_absolute_time()-dw0);
                        dispatch_semaphore_signal(cap->sem);
                    });

                    // QKV backward (ANE): un-RoPE'd dQ,dK + dV already in qkvBwd->ioIn
                    t0=mach_absolute_time();
                    if (!neon_rope) {
                    // CPU path: write fp32 dQ,dK to qkvBwd->ioIn + copy dV
                    { _Float16 *qkv_in = io_lock_rw(kern[L].qkvBwd->ioIn);
                    cvt_f32_f16(qkv_in,              cap->dq[slot], DIM * SEQ);
                    cvt_f32_f16(qkv_in + DIM*SEQ,    cap->dk[slot], DIM * SEQ);
                    const _Float16 *bwd1_p = io_lock_ro(kern[L].sdpaBwd1->ioOut);
                    memcpy(qkv_in + 2*DIM*SEQ, bwd1_p, DIM * SEQ * sizeof(_Float16));
                    io_unlock_ro(kern[L].sdpaBwd1->ioOut);
                    io_unlock_rw(kern[L].qkvBwd->ioIn);
                    }
                    } // NEON path: qkvBwd->ioIn already fully populated above
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(kern[L].qkvBwd);
                    t1=mach_absolute_time(); t_bwd_ane+=tb_ms(t1-t0); t0=t1;
                    io_read_fp16(kern[L].qkvBwd->ioOut, dx_attn, 0, DIM, SEQ);
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0);

                    // RMSNorm1 backward: dw on CPU, dx+resid fused on ANE (rmsBwd1)
                    t0=mach_absolute_time();
                    rmsnorm_dw(gr->rms_att, dx_attn, ac->layer_in, ac->rrms_att, DIM, SEQ);
                    t1=mach_absolute_time(); t_rms_bwd+=tb_ms(t1-t0); t0=t1;
                    { // Pack rmsBwd1: dx_attn, x, w, dx2 (from rmsBwd2 fused output)
                    _Float16 *rms_in = io_lock_rw(kern[L].rmsBwd1->ioIn);
                    const _Float16 *qkv_p = io_lock_ro(kern[L].qkvBwd->ioOut);
                    const _Float16 *qkv_fwd_in = io_lock_ro(kern[L].qkvFwd->ioIn);
                    const _Float16 *rms2_out = io_lock_ro(kern[L].rmsBwd2->ioOut);
                    memcpy(rms_in, qkv_p, DIM * SEQ * sizeof(_Float16));
                    memcpy(rms_in + DIM*SEQ, qkv_fwd_in, DIM * SEQ * sizeof(_Float16));
                    for (int c = 0; c < DIM; c++)
                        rms_in[(2*DIM + c) * SEQ] = (_Float16)lw[L].rms_att[c];
                    memcpy(rms_in + 3*DIM*SEQ, rms2_out, DIM * SEQ * sizeof(_Float16));
                    io_unlock_ro(kern[L].rmsBwd2->ioOut);
                    io_unlock_ro(kern[L].qkvFwd->ioIn);
                    io_unlock_ro(kern[L].qkvBwd->ioOut);
                    io_unlock_rw(kern[L].rmsBwd1->ioIn);
                    }
                    t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(kern[L].rmsBwd1);
                    t1=mach_absolute_time(); t_bwd_ane+=tb_ms(t1-t0);
                    // Fused output = dx_rms + dx2 → stays on rmsBwd1->ioOut for next layer
                }

                // Read final dy from rmsBwd1[0] fused output (fp16→fp32 for embed backward)
                t0=mach_absolute_time();
                io_read_fp16(kern[0].rmsBwd1->ioOut, dy, 0, DIM, SEQ);
                t1=mach_absolute_time(); t_bwd_io+=tb_ms(t1-t0);
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

            // Cosine LR schedule with optional warmup (use end-of-batch step)
            float cur_lr = lr;
            if (warmup_steps > 0 && step <= warmup_steps) {
                cur_lr = lr * ((float)step / warmup_steps);
            } else if (lr_min > 0) {
                float progress = (float)(step - warmup_steps) / fmaxf(1.0f, (float)(total_steps - warmup_steps));
                cur_lr = lr_min + 0.5f * (lr - lr_min) * (1.0f + cosf(M_PI * progress));
            }

            // Adam update (scale gradients by 1/(steps_batch * LOSS_SCALE), then clip grad norm)
            float gsc = 1.0f / (steps_batch * LOSS_SCALE);
            adam_t++;
            for (int L=0; L<NLAYERS; L++) {
                LayerGrads *g = &grads[L];
                for(size_t i=0;i<WQ_SZ;i++){g->Wq[i]*=gsc;g->Wk[i]*=gsc;g->Wv[i]*=gsc;g->Wo[i]*=gsc;}
                for(size_t i=0;i<W1_SZ;i++) g->W1[i]*=gsc;
                for(size_t i=0;i<W2_SZ;i++) g->W2[i]*=gsc;
                for(size_t i=0;i<W3_SZ;i++) g->W3[i]*=gsc;
                for(int i=0;i<DIM;i++){g->rms_att[i]*=gsc; g->rms_ffn[i]*=gsc;}
            }
            for(int i=0;i<DIM;i++) grms_final[i]*=gsc;
            for(size_t i=0;i<(size_t)VOCAB*DIM;i++) gembed_cls[i] = (gembed_cls[i] + gembed_emb[i]) * gsc;

            // Zero out frozen layer gradients
            for (int L=0; L<freeze_below && L<NLAYERS; L++) {
                LayerGrads *g = &grads[L];
                memset(g->Wq, 0, WQ_SZ*4); memset(g->Wk, 0, WQ_SZ*4);
                memset(g->Wv, 0, WQ_SZ*4); memset(g->Wo, 0, WQ_SZ*4);
                memset(g->W1, 0, W1_SZ*4); memset(g->W2, 0, W2_SZ*4);
                memset(g->W3, 0, W3_SZ*4);
                memset(g->rms_att, 0, DIM*4); memset(g->rms_ffn, 0, DIM*4);
            }

            // Global gradient norm clipping (max_grad_norm=1.0)
            float grad_norm_sq = 0;
            for (int L=0; L<NLAYERS; L++) {
                LayerGrads *g = &grads[L];
                float s; vDSP_Length n;
                n=WQ_SZ; vDSP_dotpr(g->Wq,1,g->Wq,1,&s,n); grad_norm_sq+=s;
                vDSP_dotpr(g->Wk,1,g->Wk,1,&s,n); grad_norm_sq+=s;
                vDSP_dotpr(g->Wv,1,g->Wv,1,&s,n); grad_norm_sq+=s;
                vDSP_dotpr(g->Wo,1,g->Wo,1,&s,n); grad_norm_sq+=s;
                n=W1_SZ; vDSP_dotpr(g->W1,1,g->W1,1,&s,n); grad_norm_sq+=s;
                n=W2_SZ; vDSP_dotpr(g->W2,1,g->W2,1,&s,n); grad_norm_sq+=s;
                n=W3_SZ; vDSP_dotpr(g->W3,1,g->W3,1,&s,n); grad_norm_sq+=s;
                n=DIM; vDSP_dotpr(g->rms_att,1,g->rms_att,1,&s,n); grad_norm_sq+=s;
                vDSP_dotpr(g->rms_ffn,1,g->rms_ffn,1,&s,n); grad_norm_sq+=s;
            }
            { float s; vDSP_dotpr(grms_final,1,grms_final,1,&s,DIM); grad_norm_sq+=s; }
            { float s; vDSP_dotpr(gembed_cls,1,gembed_cls,1,&s,(vDSP_Length)((size_t)VOCAB*DIM)); grad_norm_sq+=s; }
            float grad_norm = sqrtf(grad_norm_sq);
            float max_grad_norm = 1.0f;
            float clip_coef = (grad_norm > max_grad_norm) ? max_grad_norm / grad_norm : 1.0f;
            if (clip_coef < 1.0f) {
                for (int L=0; L<NLAYERS; L++) {
                    LayerGrads *g = &grads[L];
                    vDSP_vsmul(g->Wq,1,&clip_coef,g->Wq,1,WQ_SZ);
                    vDSP_vsmul(g->Wk,1,&clip_coef,g->Wk,1,WQ_SZ);
                    vDSP_vsmul(g->Wv,1,&clip_coef,g->Wv,1,WQ_SZ);
                    vDSP_vsmul(g->Wo,1,&clip_coef,g->Wo,1,WQ_SZ);
                    vDSP_vsmul(g->W1,1,&clip_coef,g->W1,1,W1_SZ);
                    vDSP_vsmul(g->W2,1,&clip_coef,g->W2,1,W2_SZ);
                    vDSP_vsmul(g->W3,1,&clip_coef,g->W3,1,W3_SZ);
                    vDSP_vsmul(g->rms_att,1,&clip_coef,g->rms_att,1,DIM);
                    vDSP_vsmul(g->rms_ffn,1,&clip_coef,g->rms_ffn,1,DIM);
                }
                vDSP_vsmul(grms_final,1,&clip_coef,grms_final,1,DIM);
                vDSP_vsmul(gembed_cls,1,&clip_coef,gembed_cls,1,(vDSP_Length)((size_t)VOCAB*DIM));
                printf("    [grad_clip: norm=%.1f → %.1f]\n", grad_norm, max_grad_norm);
            }

            float wd = 0.1f;
            for (int L=0; L<NLAYERS; L++) {
                LayerGrads *g = &grads[L];
                adam_update(lw[L].Wq, g->Wq, &la[L].Wq, adam_t, cur_lr, adam_b1, adam_b2, adam_eps, wd);
                adam_update(lw[L].Wk, g->Wk, &la[L].Wk, adam_t, cur_lr, adam_b1, adam_b2, adam_eps, wd);
                adam_update(lw[L].Wv, g->Wv, &la[L].Wv, adam_t, cur_lr, adam_b1, adam_b2, adam_eps, wd);
                adam_update(lw[L].Wo, g->Wo, &la[L].Wo, adam_t, cur_lr, adam_b1, adam_b2, adam_eps, wd);
                adam_update(lw[L].W1, g->W1, &la[L].W1, adam_t, cur_lr, adam_b1, adam_b2, adam_eps, wd);
                adam_update(lw[L].W2, g->W2, &la[L].W2, adam_t, cur_lr, adam_b1, adam_b2, adam_eps, wd);
                adam_update(lw[L].W3, g->W3, &la[L].W3, adam_t, cur_lr, adam_b1, adam_b2, adam_eps, wd);
                adam_update(lw[L].rms_att, g->rms_att, &la[L].rms_att, adam_t, cur_lr, adam_b1, adam_b2, adam_eps, 0.0f);
                adam_update(lw[L].rms_ffn, g->rms_ffn, &la[L].rms_ffn, adam_t, cur_lr, adam_b1, adam_b2, adam_eps, 0.0f);
            }
            adam_update(rms_final, grms_final, &arms_final, adam_t, cur_lr, adam_b1, adam_b2, adam_eps, 0.0f);
            adam_update(embed, gembed_cls, &aembed, adam_t, cur_lr, adam_b1, adam_b2, adam_eps, wd);

            printf("  [batch %d: compile=%.0fms train=%.1fms (%.1fms/step) compiles=%d lr=%.2e]\n",
                   steps_batch, cms, tms, tms/steps_batch, g_compile_count, cur_lr);
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
            free_kern(kern[L].rmsBwd2); free_kern(kern[L].rmsBwd1);
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
        for (int s=0; s<2; s++) { free(capt_dlogits_f16[s]); free(capt_dlogits[s]); free(capt_xfinal[s]); }
        adam_free(&arms_final); adam_free(&aembed);
        free(dy); free(dx_ffn); free(dx2); free(dx_attn);
        free(x_cur); free(x_final); free(logits); free(dlogits);
    }
    return 0;
}
