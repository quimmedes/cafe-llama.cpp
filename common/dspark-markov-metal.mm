#include "dspark-markov-metal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>

static const char * source = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Parameters { uint vocabulary; uint rank; uint mask; uint slot; uint groups; };

kernel void markov_partial(
    constant Parameters & p [[buffer(0)]],
    device const float * w1 [[buffer(1)]], device const float * w2 [[buffer(2)]],
    device const float * base [[buffer(3)]], device const int * previous [[buffer(4)]],
    device float * values [[buffer(5)]], device int * indices [[buffer(6)]],
    device atomic_uint * error [[buffer(7)]], threadgroup float * anchor [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    for (uint r = tid; r < p.rank; r += 128) {
        anchor[r] = w1[ulong(previous[0]) * p.rank + r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float best = -INFINITY;
    int index = INT_MAX;
    for (uint v = group * 128 + simd; v < min((group + 1) * 128, p.vocabulary); v += 4) {
        float sum = 0;
        for (uint r = lane; r < p.rank; r += 32) {
            sum += anchor[r] * w2[ulong(v) * p.rank + r];
        }
        sum = simd_sum(sum);
        const float score = base[ulong(p.slot) * p.vocabulary + v] + sum;
        if (v != p.mask) {
            if (!isfinite(score)) {
                if (lane == 0) atomic_store_explicit(error, 1u, memory_order_relaxed);
            } else if (score > best || (score == best && int(v) < index)) {
                best = score;
                index = int(v);
            }
        }
    }
    threadgroup float sv[4];
    threadgroup int si[4];
    if (lane == 0) { sv[simd] = best; si[simd] = index; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        best = lane < 4 ? sv[lane] : -INFINITY;
        index = lane < 4 ? si[lane] : INT_MAX;
        float maximum = simd_max(best);
        int winner = simd_min(best == maximum ? index : INT_MAX);
        if (lane == 0) { values[group] = maximum; indices[group] = winner; }
    }
}

kernel void markov_finish(
    constant Parameters & p [[buffer(0)]], device const float * values [[buffer(1)]],
    device const int * indices [[buffer(2)]], device int * previous [[buffer(3)]],
    device int * output [[buffer(4)]], device atomic_uint * error [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    float best = -INFINITY;
    int index = INT_MAX;
    for (uint i = tid; i < p.groups; i += 256) {
        if (values[i] > best || (values[i] == best && indices[i] < index)) {
            best = values[i]; index = indices[i];
        }
    }
    float maximum = simd_max(best);
    int winner = simd_min(best == maximum ? index : INT_MAX);
    threadgroup float sv[8];
    threadgroup int si[8];
    if (lane == 0) { sv[simd] = maximum; si[simd] = winner; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        best = lane < 8 ? sv[lane] : -INFINITY;
        index = lane < 8 ? si[lane] : INT_MAX;
        maximum = simd_max(best);
        winner = simd_min(best == maximum ? index : INT_MAX);
        if (lane == 0) {
            if (winner == INT_MAX) {
                atomic_store_explicit(error, 1u, memory_order_relaxed);
                winner = 0;
            }
            output[p.slot] = winner;
            previous[0] = winner;
        }
    }
}
)metal";

struct dspark_markov_metal {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLComputePipelineState> partial;
    id<MTLComputePipelineState> finish;
    id<MTLBuffer>               w1, w2, base, output, state, values, indices;
    uint32_t                    vocabulary = 0, rank = 0, mask = 0, groups = 0;
    int32_t                     capacity = 0;
};

dspark_markov_metal * dspark_markov_metal_init(const float * w1,
                                               const float * w2,
                                               int64_t       vocabulary,
                                               int64_t       rank,
                                               int32_t       mask_id) {
    if (!w1 || !w2 || vocabulary < 2 || vocabulary > INT32_MAX - 128 || rank < 1 || rank > 4096 || mask_id < 0 ||
        mask_id >= vocabulary || uint64_t(vocabulary) > SIZE_MAX / sizeof(float) / rank) {
        return nullptr;
    }
    const size_t elements = size_t(vocabulary) * rank;
    for (size_t i = 0; i < elements; ++i) {
        if (!std::isfinite(w1[i]) || !std::isfinite(w2[i])) {
            return nullptr;
        }
    }
    @autoreleasepool {
        auto ctx    = std::make_unique<dspark_markov_metal>();
        ctx->device = MTLCreateSystemDefaultDevice();
        if (!ctx->device || !ctx->device.hasUnifiedMemory) {
            return nullptr;
        }
        ctx->queue                  = [ctx->device newCommandQueue];
        MTLCompileOptions * options = [MTLCompileOptions new];
        options.fastMathEnabled     = NO;
        options.languageVersion     = MTLLanguageVersion2_4;
        NSError *      error        = nil;
        id<MTLLibrary> library =
            [ctx->device newLibraryWithSource:[NSString stringWithUTF8String:source] options:options error:&error];
        if (!library) {
            fprintf(stderr, "Metal Markov compilation: %s\n", error.localizedDescription.UTF8String);
            return nullptr;
        }
        ctx->partial = [ctx->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"markov_partial"]
                                                                  error:&error];
        ctx->finish  = [ctx->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"markov_finish"]
                                                                  error:&error];
        if (!ctx->queue || !ctx->partial || !ctx->finish || ctx->partial.threadExecutionWidth != 32 ||
            ctx->finish.threadExecutionWidth != 32 || ctx->partial.maxTotalThreadsPerThreadgroup < 128 ||
            ctx->finish.maxTotalThreadsPerThreadgroup < 256 ||
            ctx->device.maxThreadgroupMemoryLength < ((size_t(rank) * sizeof(float) + 15) & ~size_t(15)) + 32) {
            return nullptr;
        }
        ctx->vocabulary = uint32_t(vocabulary);
        ctx->rank       = uint32_t(rank);
        ctx->mask       = uint32_t(mask_id);
        ctx->groups     = (uint32_t(vocabulary) + 127) / 128;
        ctx->w1 =
            [ctx->device newBufferWithBytes:w1 length:elements * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->w2 =
            [ctx->device newBufferWithBytes:w2 length:elements * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->state = [ctx->device newBufferWithLength:8 options:MTLResourceStorageModeShared];
        ctx->values =
            [ctx->device newBufferWithLength:ctx->groups * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->indices =
            [ctx->device newBufferWithLength:ctx->groups * sizeof(int32_t) options:MTLResourceStorageModeShared];
        if (!ctx->w1 || !ctx->w2 || !ctx->state || !ctx->values || !ctx->indices) {
            return nullptr;
        }
        return ctx.release();
    }
}

void dspark_markov_metal_free(dspark_markov_metal * ctx) {
    delete ctx;
}

bool dspark_markov_metal_resample(dspark_markov_metal * ctx,
                                  const float *         logits,
                                  int32_t               anchor,
                                  int32_t               slots,
                                  int32_t *             output) {
    if (!ctx || !logits || !output || slots < 1 || slots > 64 || anchor < 0 || uint32_t(anchor) >= ctx->vocabulary ||
        uint32_t(anchor) == ctx->mask) {
        return false;
    }
    @autoreleasepool {
        if (slots > ctx->capacity) {
            ctx->base = [ctx->device newBufferWithLength:size_t(slots) * ctx->vocabulary * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
            ctx->output =
                [ctx->device newBufferWithLength:size_t(slots) * sizeof(int32_t) options:MTLResourceStorageModeShared];
            if (!ctx->base || !ctx->output) {
                ctx->capacity = 0;
                return false;
            }
            ctx->capacity = slots;
        }
        std::memcpy(ctx->base.contents, logits, size_t(slots) * ctx->vocabulary * sizeof(float));
        auto state                           = static_cast<int32_t *>(ctx->state.contents);
        state[0]                             = anchor;
        state[1]                             = 0;
        id<MTLCommandBuffer>         command = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) {
            return false;
        }

        struct Parameters {
            uint32_t vocabulary, rank, mask, slot, groups;
        };

        for (int32_t slot = 0; slot < slots; ++slot) {
            Parameters p{ ctx->vocabulary, ctx->rank, ctx->mask, uint32_t(slot), ctx->groups };
            [encoder setComputePipelineState:ctx->partial];
            [encoder setBytes:&p length:sizeof(p) atIndex:0];
            [encoder setBuffer:ctx->w1 offset:0 atIndex:1];
            [encoder setBuffer:ctx->w2 offset:0 atIndex:2];
            [encoder setBuffer:ctx->base offset:0 atIndex:3];
            [encoder setBuffer:ctx->state offset:0 atIndex:4];
            [encoder setBuffer:ctx->values offset:0 atIndex:5];
            [encoder setBuffer:ctx->indices offset:0 atIndex:6];
            [encoder setBuffer:ctx->state offset:4 atIndex:7];
            [encoder setThreadgroupMemoryLength:((ctx->rank * sizeof(float) + 15) & ~size_t(15)) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(ctx->groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [encoder setComputePipelineState:ctx->finish];
            [encoder setBytes:&p length:sizeof(p) atIndex:0];
            [encoder setBuffer:ctx->values offset:0 atIndex:1];
            [encoder setBuffer:ctx->indices offset:0 atIndex:2];
            [encoder setBuffer:ctx->state offset:0 atIndex:3];
            [encoder setBuffer:ctx->output offset:0 atIndex:4];
            [encoder setBuffer:ctx->state offset:4 atIndex:5];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted || state[1]) {
            return false;
        }
        std::memcpy(output, ctx->output.contents, size_t(slots) * sizeof(int32_t));
        return true;
    }
}
