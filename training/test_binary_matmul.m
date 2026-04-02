// test_binary_matmul.m — Benchmark 1-bit binary matmul on ANE
// Tests whether ANE can efficiently handle Bonsai-8B style 1-bit weights
// Three approaches: (1) baked fp16 conv, (2) matmul with weights as input, (3) binary dequantized
// Dimensions: 768 (Stories110M) and 4096 (Bonsai-8B)
#import <Foundation/Foundation.h>
#import "ane_runtime.h"
#import "ane_mil_gen.h"
#include <mach/mach_time.h>
#include <math.h>

static mach_timebase_info_data_t tb;
static double ms(uint64_t t) { return (double)t * tb.numer / tb.denom / 1e6; }

// Generate 1-bit weights: sign × group_scale (Q1_0_g128 format)
// signs: ±1 randomly, scale: absmax of original fp16 group
static void gen_binary_weights(float *out, int rows, int cols, int group_size) {
    srand48(42);
    for (int r = 0; r < rows; r++) {
        for (int g = 0; g < cols / group_size; g++) {
            float scale = 0.02f + 0.01f * drand48(); // realistic scale range
            for (int i = 0; i < group_size; i++) {
                float sign = (drand48() > 0.5) ? 1.0f : -1.0f;
                out[r * cols + g * group_size + i] = sign * scale;
            }
        }
    }
}

// Generate random fp16-range weights for baseline
static void gen_random_weights(float *out, int rows, int cols) {
    srand48(42);
    float scale = 1.0f / sqrtf(cols);
    for (int i = 0; i < rows * cols; i++)
        out[i] = scale * (2.0f * drand48() - 1.0f);
}

// Generate random input
static void gen_random_input(float *out, int ch, int spatial) {
    srand48(123);
    for (int i = 0; i < ch * spatial; i++)
        out[i] = 0.1f * (2.0f * drand48() - 1.0f);
}

// ===== MIL generator for binary matmul using element-wise approach =====
// Strategy: decompose into sign-multiply + group-reduce + scale-multiply
// signs[out_ch, in_ch] as ±1 fp16, scales[out_ch, groups] as fp16
// y_i = Σ_g scale_ig × Σ_j∈group(sign_ij × x_j)
// But this decomposes poorly on ANE. Instead, just use conv with dequantized weights.
// The real test is: does ANE care that weights are ±scale vs random?

typedef struct {
    const char *name;
    double compile_ms;
    double eval_ms;
    int evals;
} BenchResult;

static BenchResult bench_baked_conv(int dim, int spatial, float *weights, const char *name) {
    BenchResult r = {name, 0, 0, 0};

    // Build weight blob
    NSData *wblob = mil_build_weight_blob(weights, dim, dim);

    // Generate MIL
    NSString *mil = mil_gen_conv(dim, dim, spatial);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

    // Compile
    size_t inBytes = (size_t)dim * spatial * 4;   // fp32 input [1, dim, 1, spatial]
    size_t outBytes = (size_t)dim * spatial * 4;   // fp32 output

    uint64_t t0 = mach_absolute_time();
    // Use the lower-level compile path with weight dict
    NSDictionary *wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wblob}};

    NSError *e = nil;
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
        milData, wdict, nil);
    if (!desc) { printf("  [%s] Descriptor creation failed\n", name); return r; }

    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
        g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);

    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [wblob writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        printf("  [%s] Compile failed: %s\n", name, [[e description] UTF8String]);
        [fm removeItemAtPath:td error:nil];
        return r;
    }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        printf("  [%s] Load failed: %s\n", name, [[e description] UTF8String]);
        [fm removeItemAtPath:td error:nil];
        return r;
    }
    uint64_t t1 = mach_absolute_time();
    r.compile_ms = ms(t1 - t0);

    // Create IOSurfaces
    IOSurfaceRef ioIn = ane_create_surface(inBytes);
    IOSurfaceRef ioOut = ane_create_surface(outBytes);

    // Fill input
    float *input = malloc(dim * spatial * 4);
    gen_random_input(input, dim, spatial);
    IOSurfaceLock(ioIn, 0, NULL);
    // Convert to fp32 [1, dim, 1, spatial] layout
    float *base = IOSurfaceGetBaseAddress(ioIn);
    memcpy(base, input, dim * spatial * 4);
    IOSurfaceUnlock(ioIn, 0, NULL);

    // Build request
    id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioIn);
    id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioOut);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
        g_ANEReq, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

    // Warmup
    for (int i = 0; i < 5; i++)
        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);

    // Benchmark
    int iters = 100;
    t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++)
        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    t1 = mach_absolute_time();
    r.eval_ms = ms(t1 - t0) / iters;
    r.evals = iters;

    // Read back a few values for sanity check
    IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
    float *out = IOSurfaceGetBaseAddress(ioOut);
    printf("  [%s] sample output: [%.4f, %.4f, %.4f, %.4f]\n", name, out[0], out[1], out[2], out[3]);
    IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);

    // Cleanup
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
    CFRelease(ioIn); CFRelease(ioOut);
    [fm removeItemAtPath:td error:nil];
    free(input);

    return r;
}

static BenchResult bench_matmul_input(int dim, int spatial, float *weights, const char *name) {
    BenchResult r = {name, 0, 0, 0};

    NSString *mil = mil_gen_matmul(dim, dim, spatial);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

    size_t xBytes = (size_t)dim * spatial * 4;      // [1, dim, spatial]
    size_t wBytes = (size_t)dim * dim * 4;           // [1, dim, dim]
    size_t outBytes = (size_t)dim * spatial * 4;

    size_t inSizes[2] = {xBytes, wBytes};
    size_t outSizes[1] = {outBytes};

    uint64_t t0 = mach_absolute_time();
    ANEKernel *k = ane_compile(milData, nil, 2, inSizes, 1, outSizes);
    uint64_t t1 = mach_absolute_time();
    if (!k) { printf("  [%s] Compile failed\n", name); return r; }
    r.compile_ms = ms(t1 - t0);

    // Fill inputs
    float *input = malloc(dim * spatial * 4);
    gen_random_input(input, dim, spatial);
    ane_write_input(k, 0, input, xBytes);
    ane_write_input(k, 1, weights, wBytes);

    // Warmup
    for (int i = 0; i < 5; i++) ane_eval(k);

    // Benchmark
    int iters = 100;
    t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) ane_eval(k);
    t1 = mach_absolute_time();
    r.eval_ms = ms(t1 - t0) / iters;
    r.evals = iters;

    // Sanity check
    float out[4];
    ane_read_output(k, 0, out, 16);
    printf("  [%s] sample output: [%.4f, %.4f, %.4f, %.4f]\n", name, out[0], out[1], out[2], out[3]);

    ane_free(k);
    free(input);
    return r;
}

// Verify numerical correctness: compare ANE output to CPU reference
static void verify_output(float *ane_out, float *weights, float *input, int dim, int spatial, const char *name) {
    // CPU reference: y = W @ x  (W[dim,dim] × x[dim,spatial] → y[dim,spatial])
    float max_err = 0;
    double sum_err = 0;
    int n = 0;
    for (int i = 0; i < dim && i < 8; i++) {  // check first 8 output channels
        for (int s = 0; s < spatial && s < 8; s++) {
            float ref = 0;
            for (int j = 0; j < dim; j++)
                ref += weights[i * dim + j] * input[j * spatial + s];
            float err = fabsf(ane_out[i * spatial + s] - ref);
            if (err > max_err) max_err = err;
            sum_err += err;
            n++;
        }
    }
    printf("  [%s] accuracy: max_err=%.6f avg_err=%.6f (vs CPU fp32)\n", name, max_err, sum_err/n);
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        mach_timebase_info(&tb);
        ane_init();

        printf("=== 1-Bit Binary Matmul on ANE: Feasibility Benchmark ===\n\n");

        // Test configurations
        typedef struct { int dim; int spatial; const char *label; } TestConfig;
        TestConfig configs[] = {
            {768,  256, "Stories110M (768×768, seq=256)"},
            {2048, 256, "Mid-scale (2048×2048, seq=256)"},
            {4096, 256, "Bonsai-8B (4096×4096, seq=256)"},
        };
        int nconfigs = sizeof(configs) / sizeof(configs[0]);

        for (int c = 0; c < nconfigs; c++) {
            int dim = configs[c].dim;
            int spatial = configs[c].spatial;
            printf("--- %s ---\n", configs[c].label);
            printf("  Matmul: [%d, %d] × [%d, %d] = [%d, %d]\n", dim, dim, dim, spatial, dim, spatial);
            printf("  Weight memory: %.1f MB (fp16) vs %.1f MB (1-bit Q1_0_g128)\n",
                   (double)dim*dim*2/1e6, (double)dim*dim*(1.125/8)/1e6);
            printf("  FLOPs per eval: %.1f MFLOP\n\n", 2.0*dim*dim*spatial/1e6);

            // Allocate weights
            float *w_random = malloc((size_t)dim * dim * 4);
            float *w_binary = malloc((size_t)dim * dim * 4);
            gen_random_weights(w_random, dim, dim);
            gen_binary_weights(w_binary, dim, dim, 128);

            // Test 1: Baked conv with random fp16 weights (baseline)
            BenchResult r1 = bench_baked_conv(dim, spatial, w_random, "fp16_baked");

            // Test 2: Baked conv with binary-dequantized weights
            BenchResult r2 = bench_baked_conv(dim, spatial, w_binary, "binary_baked");

            // Test 3: Matmul with random weights as input (dynamic path)
            BenchResult r3 = bench_matmul_input(dim, spatial, w_random, "fp16_dynamic");

            // Test 4: Matmul with binary weights as input (dynamic path)
            BenchResult r4 = bench_matmul_input(dim, spatial, w_binary, "binary_dynamic");

            printf("\n  Results:\n");
            printf("  %-20s compile=%7.1f ms  eval=%6.3f ms  TFLOPS=%.2f\n",
                   r1.name, r1.compile_ms, r1.eval_ms,
                   r1.eval_ms > 0 ? 2.0*dim*dim*spatial/r1.eval_ms/1e9 : 0);
            printf("  %-20s compile=%7.1f ms  eval=%6.3f ms  TFLOPS=%.2f\n",
                   r2.name, r2.compile_ms, r2.eval_ms,
                   r2.eval_ms > 0 ? 2.0*dim*dim*spatial/r2.eval_ms/1e9 : 0);
            printf("  %-20s compile=%7.1f ms  eval=%6.3f ms  TFLOPS=%.2f\n",
                   r3.name, r3.compile_ms, r3.eval_ms,
                   r3.eval_ms > 0 ? 2.0*dim*dim*spatial/r3.eval_ms/1e9 : 0);
            printf("  %-20s compile=%7.1f ms  eval=%6.3f ms  TFLOPS=%.2f\n",
                   r4.name, r4.compile_ms, r4.eval_ms,
                   r4.eval_ms > 0 ? 2.0*dim*dim*spatial/r4.eval_ms/1e9 : 0);

            // Binary vs fp16 speedup
            if (r1.eval_ms > 0 && r2.eval_ms > 0)
                printf("\n  Binary vs fp16 (baked): %.2fx %s\n",
                       r1.eval_ms / r2.eval_ms,
                       r2.eval_ms < r1.eval_ms ? "FASTER" : "slower");
            if (r3.eval_ms > 0 && r4.eval_ms > 0)
                printf("  Binary vs fp16 (dynamic): %.2fx %s\n",
                       r3.eval_ms / r4.eval_ms,
                       r4.eval_ms < r3.eval_ms ? "FASTER" : "slower");
            if (r1.eval_ms > 0 && r3.eval_ms > 0)
                printf("  Baked vs dynamic (fp16): %.2fx %s\n",
                       r3.eval_ms / r1.eval_ms,
                       r1.eval_ms < r3.eval_ms ? "FASTER" : "slower");

            printf("\n");
            free(w_random);
            free(w_binary);
        }

        printf("=== Summary ===\n");
        printf("If binary_baked ≈ fp16_baked: ANE doesn't optimize for ±1 weights (expected)\n");
        printf("  → 1-bit advantage is purely in weight storage (14x smaller) and compile time\n");
        printf("If 4096×4096 works at all: Bonsai-8B scale is feasible on ANE\n");
        printf("If dynamic eval is fast: LoRA adapters via input weights are practical\n");
    }
    return 0;
}
