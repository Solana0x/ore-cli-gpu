#include <stdint.h>
#include <stdio.h>
#include "drillx.h"
#include "equix.h"
#include "hashx.h"
#include "equix/src/context.h"
#include "equix/src/solver.h"
#include "equix/src/solver_heap.h"
#include "hashx/src/context.h"

const int BATCH_SIZE = 4096;
const int STREAM_COUNT = 4;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err); \
        } \
    } while (0)

extern "C" void hash(uint8_t *challenge, uint8_t *nonce, uint64_t *out) {
    hashx_ctx** ctxs;
    uint64_t** hash_space;

    CUDA_CHECK(cudaMallocManaged(&ctxs, BATCH_SIZE * sizeof(hashx_ctx*)));
    CUDA_CHECK(cudaMallocManaged(&hash_space, BATCH_SIZE * sizeof(uint64_t*)));

    for (int i = 0; i < BATCH_SIZE; i++) {
        CUDA_CHECK(cudaMallocManaged(&hash_space[i], INDEX_SPACE * sizeof(uint64_t)));
    }

    uint8_t seed[40];
    memcpy(seed, challenge, 32);

    // Create multiple CUDA streams for overlapping operations
    cudaStream_t streams[STREAM_COUNT];
    for (int i = 0; i < STREAM_COUNT; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    int completed_batches = 0;
    while (completed_batches < TOTAL_BATCHES) {  // TOTAL_BATCHES should be defined appropriately
        for (int stream_id = 0; stream_id < STREAM_COUNT; ++stream_id) {
            int batch_offset = completed_batches * STREAM_COUNT + stream_id;
            if (batch_offset >= TOTAL_BATCHES) break;

            for (int i = 0; i < BATCH_SIZE; i++) {
                uint64_t nonce_offset = *((uint64_t*)nonce) + i + batch_offset * BATCH_SIZE;
                memcpy(seed + 32, &nonce_offset, 8);
                ctxs[i] = hashx_alloc(HASHX_INTERPRETED);
                if (!ctxs[i] || !hashx_make(ctxs[i], seed, 40)) {
                    for (int j = 0; j <= i; j++) {
                        hashx_free(ctxs[j]);
                    }
                    return;
                }
            }

            dim3 threadsPerBlock(1024);
            dim3 blocksPerGrid((BATCH_SIZE * INDEX_SPACE + threadsPerBlock.x - 1) / threadsPerBlock.x);

            // Launch kernel asynchronously in a stream
            do_hash_stage0i<<<blocksPerGrid, threadsPerBlock, 0, streams[stream_id]>>>(ctxs, hash_space);
            CUDA_CHECK(cudaGetLastError()); // Check for launch errors
        }

        for (int stream_id = 0; stream_id < STREAM_COUNT; ++stream_id) {
            // Synchronize the stream to ensure the kernel has completed
            CUDA_CHECK(cudaStreamSynchronize(streams[stream_id]));

            // Asynchronously copy the results back to the host
            for (int i = 0; i < BATCH_SIZE; i++) {
                int batch_offset = completed_batches * STREAM_COUNT + stream_id;
                if (batch_offset >= TOTAL_BATCHES) break;
                CUDA_CHECK(cudaMemcpyAsync(out + (batch_offset * BATCH_SIZE + i) * INDEX_SPACE, hash_space[i], INDEX_SPACE * sizeof(uint64_t), cudaMemcpyDeviceToHost, streams[stream_id]));
            }
        }

        completed_batches += STREAM_COUNT;
    }

    // Ensure all asynchronous operations are completed
    for (int i = 0; i < STREAM_COUNT; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    // Free resources at the end
    for (int i = 0; i < BATCH_SIZE; i++) {
        hashx_free(ctxs[i]);
        CUDA_CHECK(cudaFree(hash_space[i]));
    }
    CUDA_CHECK(cudaFree(ctxs));
}

__global__ void do_hash_stage0i(hashx_ctx** ctxs, uint64_t** hash_space) {
    uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t batch_idx = item / INDEX_SPACE;
    uint32_t i = item % INDEX_SPACE;
    if (batch_idx < BATCH_SIZE) {
        hash_stage0i(ctxs[batch_idx], hash_space[batch_idx], i);
    }
}
