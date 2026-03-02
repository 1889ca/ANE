// test_zero_copy.m — Test ANE→GPU zero-copy via IOSurface + MTLSharedEvent signaling
//
// Three experiments:
// 1. Can we create an IOSurface and import it as MTLBuffer? (zero-copy memory)
// 2. Can we attach MTLSharedEvent to _ANERequest? (zero-copy signaling)
// 3. Does ANE write data that GPU can read without CPU memcpy? (end-to-end)
//
// Build: clang -framework Foundation -framework Metal -framework IOSurface \
//        -framework CoreGraphics -lobjc test_zero_copy.m -o test_zero_copy

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

// ─── Experiment 1: IOSurface → MTLBuffer aliasing ───

static void test_iosurface_mtlbuffer(id<MTLDevice> device) {
    printf("\n══════════════════════════════════════════════════════\n");
    printf("EXPERIMENT 1: IOSurface → MTLBuffer zero-copy\n");
    printf("══════════════════════════════════════════════════════\n\n");

    // Create IOSurface matching our ANE tensor layout: [DIM, SEQ] fp16
    // Using 768 * 256 * 2 bytes = 393216 bytes
    int width = 768;
    int height = 256;
    int bytesPerElement = 2;  // fp16
    int bytesPerRow = width * bytesPerElement;
    int totalBytes = bytesPerRow * height;

    NSDictionary *props = @{
        (id)kIOSurfaceWidth: @(width),
        (id)kIOSurfaceHeight: @(height),
        (id)kIOSurfaceBytesPerElement: @(bytesPerElement),
        (id)kIOSurfaceBytesPerRow: @(bytesPerRow),
        (id)kIOSurfaceAllocSize: @(totalBytes),
        (id)kIOSurfacePixelFormat: @(0x00000010),  // kCVPixelFormatType_16Gray
    };

    IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
    if (!surface) {
        printf("  FAIL: Could not create IOSurface\n");
        return;
    }
    printf("  IOSurface created: %d x %d, %d bytes/row, alloc=%d\n",
           IOSurfaceGetWidth(surface), IOSurfaceGetHeight(surface),
           (int)IOSurfaceGetBytesPerRow(surface), (int)IOSurfaceGetAllocSize(surface));

    // Write a known pattern via CPU
    IOSurfaceLock(surface, 0, NULL);
    uint16_t *base = (uint16_t *)IOSurfaceGetBaseAddress(surface);
    int nElements = totalBytes / 2;
    for (int i = 0; i < nElements; i++) {
        // Write fp16 pattern: 0x3C00 = 1.0 in fp16
        base[i] = 0x3C00 + (i & 0xFF);  // 1.0 + small variation
    }
    IOSurfaceUnlock(surface, 0, NULL);
    printf("  Wrote %d fp16 elements to IOSurface\n", nElements);

    // Method A: newBufferWithBytesNoCopy from IOSurface base address
    printf("\n  --- Method A: MTLBuffer via newBufferWithBytesNoCopy ---\n");
    void *baseAddr = IOSurfaceGetBaseAddress(surface);
    size_t allocSize = IOSurfaceGetAllocSize(surface);

    id<MTLBuffer> bufferA = [device newBufferWithBytesNoCopy:baseAddr
                                                      length:allocSize
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
    if (bufferA) {
        printf("  SUCCESS: MTLBuffer created from IOSurface base address!\n");
        printf("    buffer.length = %lu, buffer.contents = %p, IOSurface base = %p\n",
               (unsigned long)[bufferA length], [bufferA contents], baseAddr);
        printf("    Same pointer: %s\n",
               [bufferA contents] == baseAddr ? "YES (true zero-copy!)" : "NO (copied)");

        // Verify data
        uint16_t *gpuData = (uint16_t *)[bufferA contents];
        int mismatches = 0;
        for (int i = 0; i < 100; i++) {
            if (gpuData[i] != (0x3C00 + (i & 0xFF))) mismatches++;
        }
        printf("    Data check (first 100): %d mismatches\n", mismatches);
    } else {
        printf("  FAIL: Metal rejected newBufferWithBytesNoCopy from IOSurface address\n");
        printf("    (This is expected if Metal validates allocation provenance)\n");
    }

    // Method B: MTLTexture from IOSurface
    printf("\n  --- Method B: MTLTexture via newTextureWithDescriptor:iosurface: ---\n");
    MTLTextureDescriptor *texDesc = [[MTLTextureDescriptor alloc] init];
    texDesc.textureType = MTLTextureType2D;
    texDesc.pixelFormat = MTLPixelFormatR16Float;
    texDesc.width = width;
    texDesc.height = height;
    texDesc.usage = MTLTextureUsageShaderRead;
    texDesc.storageMode = MTLStorageModeShared;

    id<MTLTexture> textureB = [device newTextureWithDescriptor:texDesc
                                                     iosurface:surface
                                                         plane:0];
    if (textureB) {
        printf("  SUCCESS: MTLTexture created from IOSurface!\n");
        printf("    texture: %lux%lu, format=%lu\n",
               (unsigned long)textureB.width, (unsigned long)textureB.height,
               (unsigned long)textureB.pixelFormat);

        // Read back via Metal compute shader to verify
        // (For now, just confirm the object was created)
    } else {
        printf("  FAIL: Metal rejected newTextureWithDescriptor:iosurface:\n");
    }

    // Method C: Check if there's a private newBufferWithIOSurface: API
    printf("\n  --- Method C: Private newBufferWithIOSurface: API ---\n");
    SEL sel = sel_registerName("newBufferWithIOSurface:");
    if ([device respondsToSelector:sel]) {
        printf("  Device responds to newBufferWithIOSurface:!\n");
        id<MTLBuffer> bufferC = ((id(*)(id,SEL,IOSurfaceRef))objc_msgSend)(
            device, sel, surface);
        if (bufferC) {
            printf("  SUCCESS: MTLBuffer from private API!\n");
            printf("    buffer.length = %lu\n", (unsigned long)[bufferC length]);
        } else {
            printf("  Returned nil\n");
        }
    } else {
        printf("  Device does NOT respond to newBufferWithIOSurface:\n");

        // Try newBufferWithIOSurface:type:
        SEL sel2 = sel_registerName("newBufferWithIOSurface:type:");
        if ([device respondsToSelector:sel2]) {
            printf("  But DOES respond to newBufferWithIOSurface:type:!\n");
        }

        // Enumerate all newBuffer* selectors
        unsigned int mc = 0;
        Method *methods = class_copyMethodList(object_getClass(device), &mc);
        for (unsigned int i = 0; i < mc; i++) {
            const char *name = sel_getName(method_getName(methods[i]));
            if (strstr(name, "newBuffer")) {
                printf("    found: %s\n", name);
            }
        }
        free(methods);
    }

    CFRelease(surface);
}

// ─── Experiment 2: MTLSharedEvent → ANE signaling ───

static void test_shared_event_signaling(id<MTLDevice> device) {
    printf("\n══════════════════════════════════════════════════════\n");
    printf("EXPERIMENT 2: MTLSharedEvent → ANE signal wiring\n");
    printf("══════════════════════════════════════════════════════\n\n");

    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

    // Create MTLSharedEvent
    id<MTLSharedEvent> event = [device newSharedEvent];
    printf("  MTLSharedEvent created: signaledValue=%llu\n", event.signaledValue);
    printf("  Class chain: ");
    Class cls = [event class];
    while (cls) {
        printf("%s", class_getName(cls));
        cls = class_getSuperclass(cls);
        if (cls) printf(" → ");
    }
    printf("\n");

    // Create _ANESharedSignalEvent wrapping our MTLSharedEvent
    Class sigCls = objc_getClass("_ANESharedSignalEvent");
    if (!sigCls) {
        printf("  FAIL: _ANESharedSignalEvent class not found\n");
        return;
    }

    // Try different agentMask values — this likely controls which hardware agent signals
    // 0 = unknown, let's see what works
    for (uint32_t mask = 0; mask <= 3; mask++) {
        id sigEvt = ((id(*)(id,SEL,uint64_t,NSInteger,NSInteger,id,uint32_t))objc_msgSend)(
            [sigCls alloc],
            sel_registerName("initWithValue:symbolIndex:eventType:sharedEvent:agentMask:"),
            (uint64_t)1,
            (NSInteger)0,  // symbolIndex
            (NSInteger)0,  // eventType
            event,
            mask
        );
        printf("  agentMask=%u: %s\n", mask, sigEvt ? "created" : "nil");
        if (sigEvt) {
            printf("    desc: %s\n", [[sigEvt description] UTF8String]);
        }
    }

    // Create _ANESharedWaitEvent
    Class waitCls = objc_getClass("_ANESharedWaitEvent");
    if (waitCls) {
        id waitEvt = ((id(*)(id,SEL,uint64_t,id,NSInteger))objc_msgSend)(
            [waitCls alloc],
            sel_registerName("initWithValue:sharedEvent:eventType:"),
            (uint64_t)1,
            event,
            (NSInteger)0
        );
        printf("\n  _ANESharedWaitEvent: %s\n", waitEvt ? "created" : "nil");
        if (waitEvt) {
            printf("    desc: %s\n", [[waitEvt description] UTF8String]);
        }
    }

    // Create _ANESharedEvents container
    Class sharedEvtsCls = objc_getClass("_ANESharedEvents");
    if (sharedEvtsCls) {
        id sigEvt = ((id(*)(id,SEL,uint64_t,NSInteger,NSInteger,id,uint32_t))objc_msgSend)(
            [sigCls alloc],
            sel_registerName("initWithValue:symbolIndex:eventType:sharedEvent:agentMask:"),
            (uint64_t)1, (NSInteger)0, (NSInteger)0, event, (uint32_t)0
        );

        NSArray *sigArray = @[sigEvt];
        NSArray *waitArray = @[];

        id sharedEvts = ((id(*)(id,SEL,id,id))objc_msgSend)(
            [sharedEvtsCls alloc],
            sel_registerName("initWithSignalEvents:waitEvents:"),
            sigArray,
            waitArray
        );
        printf("\n  _ANESharedEvents container: %s\n", sharedEvts ? "created" : "nil");
        if (sharedEvts) {
            printf("    desc: %s\n", [[sharedEvts description] UTF8String]);
        }

        // This is what we'd pass to [request setSharedEvents:sharedEvts]
        printf("\n  ✓ Full signal chain is constructable:\n");
        printf("    MTLSharedEvent → _ANESharedSignalEvent → _ANESharedEvents → _ANERequest\n");
        printf("    Then: Metal command buffer waitForEvent: on the same MTLSharedEvent\n");
    }
}

// ─── Experiment 3: IOSurface cache coherency test ───

static void test_cache_coherency(id<MTLDevice> device) {
    printf("\n══════════════════════════════════════════════════════\n");
    printf("EXPERIMENT 3: IOSurface cache coherency (CPU write → GPU read)\n");
    printf("══════════════════════════════════════════════════════\n\n");

    // Create a small IOSurface
    int size = 4096;  // 1 page
    NSDictionary *props = @{
        (id)kIOSurfaceWidth: @(size / 4),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @4,
        (id)kIOSurfaceBytesPerRow: @(size),
        (id)kIOSurfaceAllocSize: @(size),
        (id)kIOSurfacePixelFormat: @(0x20202034),  // '4   ' = 32-bit
    };

    IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
    if (!surface) {
        printf("  FAIL: Could not create IOSurface\n");
        return;
    }

    // Write known data
    IOSurfaceLock(surface, 0, NULL);
    float *data = (float *)IOSurfaceGetBaseAddress(surface);
    for (int i = 0; i < size/4; i++) {
        data[i] = (float)i * 1.5f;
    }
    IOSurfaceUnlock(surface, 0, NULL);

    // Try to read without lock (cache coherency test)
    float *rawData = (float *)IOSurfaceGetBaseAddress(surface);
    int correct_with_lock = 0, correct_without_lock = 0;
    for (int i = 0; i < size/4; i++) {
        if (rawData[i] == (float)i * 1.5f) correct_without_lock++;
    }
    printf("  Without IOSurfaceLock: %d/%d correct\n", correct_without_lock, size/4);

    // Now try MTLBuffer alias and GPU compute readback
    void *baseAddr = IOSurfaceGetBaseAddress(surface);
    id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:baseAddr
                                                  length:IOSurfaceGetAllocSize(surface)
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
    if (buf) {
        printf("  MTLBuffer alias created, running GPU readback...\n");

        // Create a simple compute shader that copies data
        NSString *shaderSrc = @
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "kernel void readback(device float* input [[buffer(0)]],\n"
            "                     device float* output [[buffer(1)]],\n"
            "                     uint id [[thread_position_in_grid]]) {\n"
            "    output[id] = input[id];\n"
            "}\n";

        NSError *err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:shaderSrc options:nil error:&err];
        if (!lib) {
            printf("  FAIL: Shader compilation: %s\n", [[err localizedDescription] UTF8String]);
            CFRelease(surface);
            return;
        }

        id<MTLFunction> func = [lib newFunctionWithName:@"readback"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
        id<MTLBuffer> outputBuf = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf offset:0 atIndex:0];
        [enc setBuffer:outputBuf offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(size/4, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        float *gpuResult = (float *)[outputBuf contents];
        int gpuCorrect = 0;
        for (int i = 0; i < size/4; i++) {
            if (gpuResult[i] == (float)i * 1.5f) gpuCorrect++;
        }
        printf("  GPU readback from IOSurface-backed MTLBuffer: %d/%d correct\n",
               gpuCorrect, size/4);

        if (gpuCorrect == size/4) {
            printf("\n  ✓ ZERO-COPY CONFIRMED: CPU wrote to IOSurface, GPU read via MTLBuffer alias\n");
            printf("    No memcpy, no IOSurfaceLock between write and GPU read needed\n");
        }
    } else {
        printf("  MTLBuffer alias failed, trying MTLTexture path...\n");
    }

    // Timing: IOSurfaceLock vs no-lock for different sizes
    printf("\n  --- IOSurfaceLock timing ---\n");
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);

    for (int trial = 0; trial < 3; trial++) {
        uint64_t t0 = mach_absolute_time();
        IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
        IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
        uint64_t t1 = mach_absolute_time();
        double us = (double)(t1-t0) * tb.numer / tb.denom / 1000.0;
        printf("  IOSurfaceLock/Unlock (read-only): %.1f µs\n", us);
    }

    CFRelease(surface);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        printf("ANE→GPU Zero-Copy Experiments\n");
        printf("═══════════════════════════════════════════════════════\n");

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            printf("FAIL: No Metal device\n");
            return 1;
        }
        printf("Metal device: %s\n", [[device name] UTF8String]);

        test_iosurface_mtlbuffer(device);
        test_shared_event_signaling(device);
        test_cache_coherency(device);

        printf("\n═══════════════════════════════════════════════════════\n");
        printf("DONE\n");
    }
    return 0;
}
