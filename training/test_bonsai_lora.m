// test_bonsai_lora.m — Test LoRA kernel generators for Bonsai-8B on ANE
// Validates: (1) correctness vs CPU, (2) LoRA overhead vs plain conv, (3) Q1 dequant
#import <Foundation/Foundation.h>
#import "stories_io.h"
#import "bonsai_lora.h"
#include <mach/mach_time.h>
#include <Accelerate/Accelerate.h>

static mach_timebase_info_data_t tb;
static double tms(uint64_t t) { return (double)t * tb.numer / tb.denom / 1e6; }

// CPU reference: y = W@x + A@(B@x)
static void cpu_lora_matmul(float *y, const float *W, const float *A, const float *B,
                             const float *x, int out_ch, int in_ch, int rank, int spatial) {
    // W@x → y [out_ch, spatial]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                out_ch, spatial, in_ch, 1.0f, W, in_ch, x, spatial, 0.0f, y, spatial);
    // B@x → tmp [rank, spatial]
    float *tmp = (float *)malloc((size_t)rank * spatial * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rank, spatial, in_ch, 1.0f, B, in_ch, x, spatial, 0.0f, tmp, spatial);
    // A@tmp → y += [out_ch, spatial]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                out_ch, spatial, rank, 1.0f, A, rank, tmp, spatial, 1.0f, y, spatial);
    free(tmp);
}

// Compile and benchmark using stories_io compile_kern_mil_w
typedef struct {
    double compile_ms;
    double eval_ms;
    bool ok;
} BenchResult;

static BenchResult compile_and_bench_w(NSString *mil, NSDictionary *weights,
                                        int in_ch, int out_ch, int spatial,
                                        const float *input, float *output,
                                        int warmup, int iters) {
    BenchResult r = {0, 0, false};
    int ic_bytes = in_ch * spatial * 2;   // fp16
    int oc_bytes = out_ch * spatial * 2;

    uint64_t t0 = mach_absolute_time();
    Kern *k = compile_kern_mil_w(mil, weights, ic_bytes, oc_bytes);
    r.compile_ms = tms(mach_absolute_time() - t0);
    if (!k) { fprintf(stderr, "  Compile failed\n"); return r; }

    // Write input (fp32 → fp16 via io_write_fp16)
    io_write_fp16(k->ioIn, input, in_ch, spatial);

    // Warmup
    for (int i = 0; i < warmup; i++) ane_eval(k);

    // Benchmark
    t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) ane_eval(k);
    r.eval_ms = tms(mach_absolute_time() - t0) / iters;

    // Read output (fp16 → fp32)
    io_read_fp16(k->ioOut, output, 0, out_ch, spatial);

    free_kern(k);
    r.ok = true;
    return r;
}

static void compare_outputs(const float *ane, const float *cpu, int n, const char *name) {
    float max_err = 0, sum_err = 0;
    int max_idx = 0;
    for (int i = 0; i < n; i++) {
        float err = fabsf(ane[i] - cpu[i]);
        sum_err += err;
        if (err > max_err) { max_err = err; max_idx = i; }
    }
    float avg_err = sum_err / n;
    // Relative error vs max output magnitude
    float max_val = 0;
    for (int i = 0; i < n; i++) if (fabsf(cpu[i]) > max_val) max_val = fabsf(cpu[i]);
    printf("  [%s] max_err=%.6f (idx=%d) avg_err=%.6f rel_err=%.4f%%\n",
           name, max_err, max_idx, avg_err, max_val > 0 ? 100.0f * max_err / max_val : 0);
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        mach_timebase_info(&tb);
        ane_init();

        printf("=== Bonsai-8B LoRA Kernel Test ===\n\n");

        // ===== Test 1: Q1_0_g128 dequantization =====
        printf("--- Test 1: Q1_0_g128 dequantization ---\n");
        {
            // Create synthetic Q1 blocks
            int n_weights = 256;  // 2 blocks of 128
            int n_blocks = n_weights / BONSAI_Q1_GROUP;
            Q1Block blocks[2];
            blocks[0].scale = (_Float16)0.0234f;
            blocks[1].scale = (_Float16)0.0187f;
            // Pattern: alternating signs
            for (int b = 0; b < n_blocks; b++)
                for (int i = 0; i < 16; i++)
                    blocks[b].signs[i] = 0xAA;  // 10101010 = alternating

            float out[256];
            q1_dequantize(out, blocks, n_weights);

            // Verify: even indices should be +scale, odd should be -scale
            bool ok = true;
            for (int i = 0; i < 8; i++) {
                float expected_even = (float)blocks[i/128].scale;
                float expected_odd = -(float)blocks[i/128].scale;
                if (fabsf(out[i*2] - expected_even) > 1e-4f) ok = false;
                if (fabsf(out[i*2+1] - expected_odd) > 1e-4f) ok = false;
            }
            printf("  Q1 dequant: %s\n", ok ? "PASS" : "FAIL");
            printf("  Sample: [%.4f, %.4f, %.4f, %.4f] (expect [+s, -s, +s, -s])\n",
                   out[0], out[1], out[2], out[3]);
        }

        // ===== Test 2: LoRA kernel correctness at small scale =====
        printf("\n--- Test 2: LoRA correctness (256×256, rank=4, seq=32) ---\n");
        {
            int in_ch = 256, out_ch = 256, rank = 4, spatial = 32;
            size_t w_sz = (size_t)out_ch * in_ch;
            size_t a_sz = (size_t)out_ch * rank;
            size_t b_sz = (size_t)rank * in_ch;

            float *W = (float *)malloc(w_sz * 4);
            float *A = (float *)malloc(a_sz * 4);
            float *B = (float *)malloc(b_sz * 4);
            float *x = (float *)malloc((size_t)in_ch * spatial * 4);

            // Binary weights
            srand48(42);
            for (size_t i = 0; i < w_sz; i++) W[i] = 0.02f * ((drand48() > 0.5) ? 1.0f : -1.0f);
            lora_init(A, B, out_ch, in_ch, rank);
            // Non-zero B for testing (lora_init zeros B, but we want to verify the path works)
            for (size_t i = 0; i < b_sz; i++) B[i] = 0.01f * (2.0f * drand48() - 1.0f);
            for (size_t i = 0; i < (size_t)in_ch * spatial; i++) x[i] = 0.1f * (2.0f * drand48() - 1.0f);

            // CPU reference
            float *y_cpu = (float *)calloc((size_t)out_ch * spatial, 4);
            cpu_lora_matmul(y_cpu, W, A, B, x, out_ch, in_ch, rank, spatial);

            // ANE LoRA kernel
            NSString *mil = bonsai_gen_lora_conv(in_ch, out_ch, rank, spatial);
            NSDictionary *wdict = bonsai_lora_weights(W, out_ch, in_ch, B, A, rank);
            float *y_ane = (float *)calloc((size_t)out_ch * spatial, 4);

            BenchResult r = compile_and_bench_w(mil, wdict, in_ch, out_ch, spatial,
                x, y_ane, 5, 50);

            if (r.ok) {
                compare_outputs(y_ane, y_cpu, out_ch * spatial, "lora_small");
                printf("  compile=%.1fms eval=%.3fms\n", r.compile_ms, r.eval_ms);
            }

            free(W); free(A); free(B); free(x); free(y_cpu); free(y_ane);
        }

        // ===== Test 3: Bonsai-8B attention projection dimensions =====
        printf("\n--- Test 3: Bonsai Q projection (4096×4096, rank=16, seq=256) ---\n");
        {
            int in_ch = BONSAI_DIM, out_ch = BONSAI_DIM, rank = 16, spatial = 256;
            size_t w_sz = (size_t)out_ch * in_ch;

            float *W = (float *)malloc(w_sz * 4);
            float *A = (float *)malloc((size_t)out_ch * rank * 4);
            float *B = (float *)malloc((size_t)rank * in_ch * 4);
            float *x = (float *)malloc((size_t)in_ch * spatial * 4);

            srand48(42);
            for (size_t i = 0; i < w_sz; i++) W[i] = 0.02f * ((drand48() > 0.5) ? 1.0f : -1.0f);
            lora_init(A, B, out_ch, in_ch, rank);
            for (size_t i = 0; i < (size_t)rank * in_ch; i++) B[i] = 0.01f * (2.0f * drand48() - 1.0f);
            for (size_t i = 0; i < (size_t)in_ch * spatial; i++) x[i] = 0.1f * (2.0f * drand48() - 1.0f);

            // LoRA kernel
            NSString *mil_lora = bonsai_gen_lora_conv(in_ch, out_ch, rank, spatial);
            NSDictionary *wdict_lora = bonsai_lora_weights(W, out_ch, in_ch, B, A, rank);
            float *y_lora = (float *)calloc((size_t)out_ch * spatial, 4);

            BenchResult r_lora = compile_and_bench_w(mil_lora, wdict_lora, in_ch, out_ch, spatial,
                x, y_lora, 5, 50);

            // Plain conv (no LoRA) for comparison
            NSString *mil_plain = bonsai_gen_conv(in_ch, out_ch, spatial);
            NSDictionary *wdict_plain = @{
                @"@model_path/weights/w.bin": @{@"offset": @0, @"data": bonsai_build_blob(W, out_ch, in_ch)}
            };
            float *y_plain = (float *)calloc((size_t)out_ch * spatial, 4);

            BenchResult r_plain = compile_and_bench_w(mil_plain, wdict_plain, in_ch, out_ch, spatial,
                x, y_plain, 5, 50);

            double flops = 2.0 * out_ch * in_ch * spatial;
            double lora_flops = flops + 2.0 * rank * in_ch * spatial + 2.0 * out_ch * rank * spatial;

            if (r_lora.ok && r_plain.ok) {
                printf("  Plain conv:  compile=%6.1fms  eval=%.3fms  TFLOPS=%.2f\n",
                       r_plain.compile_ms, r_plain.eval_ms, flops / r_plain.eval_ms / 1e9);
                printf("  LoRA conv:   compile=%6.1fms  eval=%.3fms  TFLOPS=%.2f\n",
                       r_lora.compile_ms, r_lora.eval_ms, lora_flops / r_lora.eval_ms / 1e9);
                printf("  LoRA overhead: +%.3fms (+%.1f%%)\n",
                       r_lora.eval_ms - r_plain.eval_ms,
                       100.0 * (r_lora.eval_ms - r_plain.eval_ms) / r_plain.eval_ms);
            }

            free(W); free(A); free(B); free(x); free(y_lora); free(y_plain);
        }

        // ===== Test 4: GQA K/V projection (smaller output dim) =====
        printf("\n--- Test 4: Bonsai GQA K/V projection (4096→512, rank=16, seq=256) ---\n");
        {
            int in_ch = BONSAI_DIM, out_ch = BONSAI_KV_DIM, rank = 16, spatial = 256;
            size_t w_sz = (size_t)out_ch * in_ch;

            float *W = (float *)malloc(w_sz * 4);
            float *A = (float *)malloc((size_t)out_ch * rank * 4);
            float *B = (float *)malloc((size_t)rank * in_ch * 4);
            float *x = (float *)malloc((size_t)in_ch * spatial * 4);

            srand48(42);
            for (size_t i = 0; i < w_sz; i++) W[i] = 0.02f * ((drand48() > 0.5) ? 1.0f : -1.0f);
            lora_init(A, B, out_ch, in_ch, rank);
            for (size_t i = 0; i < (size_t)rank * in_ch; i++) B[i] = 0.01f * (2.0f * drand48() - 1.0f);
            for (size_t i = 0; i < (size_t)in_ch * spatial; i++) x[i] = 0.1f * (2.0f * drand48() - 1.0f);

            NSString *mil_lora = bonsai_gen_lora_conv(in_ch, out_ch, rank, spatial);
            NSDictionary *wdict_lora = bonsai_lora_weights(W, out_ch, in_ch, B, A, rank);
            float *y_lora = (float *)calloc((size_t)out_ch * spatial, 4);

            NSString *mil_plain = bonsai_gen_conv(in_ch, out_ch, spatial);
            NSDictionary *wdict_plain = @{
                @"@model_path/weights/w.bin": @{@"offset": @0, @"data": bonsai_build_blob(W, out_ch, in_ch)}
            };
            float *y_plain = (float *)calloc((size_t)out_ch * spatial, 4);

            BenchResult r_lora = compile_and_bench_w(mil_lora, wdict_lora, in_ch, out_ch, spatial,
                x, y_lora, 5, 50);
            BenchResult r_plain = compile_and_bench_w(mil_plain, wdict_plain, in_ch, out_ch, spatial,
                x, y_plain, 5, 50);

            if (r_lora.ok && r_plain.ok) {
                printf("  Plain conv:  compile=%6.1fms  eval=%.3fms\n", r_plain.compile_ms, r_plain.eval_ms);
                printf("  LoRA conv:   compile=%6.1fms  eval=%.3fms\n", r_lora.compile_ms, r_lora.eval_ms);
                printf("  LoRA overhead: +%.3fms (+%.1f%%)\n",
                       r_lora.eval_ms - r_plain.eval_ms,
                       100.0 * (r_lora.eval_ms - r_plain.eval_ms) / r_plain.eval_ms);
            }

            free(W); free(A); free(B); free(x); free(y_lora); free(y_plain);
        }

        // ===== Summary =====
        printf("\n=== Summary ===\n");
        printf("Bonsai-8B LoRA on ANE:\n");
        printf("  - Q1_0_g128 dequant: trivial (sign × scale)\n");
        printf("  - LoRA fused into single ANE kernel: conv(W,x) + conv(A,conv(B,x))\n");
        printf("  - 4 target projections per layer (q/k/v/o) × 28 layers = 112 LoRA kernels\n");
        printf("  - Adapter size: rank-16 × 4 projections × 28 layers = ~29 MB\n");
    }
    return 0;
}
