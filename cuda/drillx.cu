#include <stdint.h>
#include <stdio.h>
#include "drillx.h"
#include "equix.h"
#include "hashx.h"
#include "equix/src/context.h"
#include "equix/src/solver.h"
#include "equix/src/solver_heap.h"
#include "hashx/src/context.h"

const int BATCH_SIZE = 8192; // Increased batch size

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err); \
        } \
    } while (0)

// Memory pool struct to handle memory more efficiently
struct MemoryPool {
    uint64_t** hash_space;
    hashx_ctx** ctxs;

    MemoryPool(int batchSize) {
        CUDA_CHECK(cudaHostAlloc(&ctxs, batchSize * sizeof(hashx_ctx*), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc(&hash_space, batchSize * sizeof(uint64_t*), cudaHostAllocMapped));
        for (int i = 0; i < batchSize; i++) {
            CUDA_CHECK(cudaMalloc(&hash_space[i], INDEX_SPACE * sizeof(uint64_t)));
        }
    }

    ~MemoryPool() {
        for (int i = 0; i < BATCH_SIZE; i++) {
            CUDA_CHECK(cudaFree(hash_space[i]));
        }
        CUDA_CHECK(cudaFreeHost(hash_space));
        CUDA_CHECK(cudaFreeHost(ctxs));
    }
};

extern "C" void hash(uint8_t *challenge, uint8_t *nonce, uint64_t *out) {
    // Create a memory pool instance to handle memory allocation
    MemoryPool memPool(BATCH_SIZE);

    uint8_t seed[40];
    memcpy(seed, challenge, 32);

    for (int i = 0; i < BATCH_SIZE; i++) {
        uint64_t nonce_offset = *((uint64_t*)nonce) + i;
        memcpy(seed + 32, &nonce_offset, 8);
        memPool.ctxs[i] = hashx_alloc(HASHX_INTERPRETED);
        if (!memPool.ctxs[i] || !hashx_make(memPool.ctxs[i], seed, 40)) {
            return;
        }
    }

    dim3 threadsPerBlock(1024);
    dim3 blocksPerGrid((BATCH_SIZE * INDEX_SPACE + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Using CUDA streams for overlapping computation and memory transfers
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    do_hash_stage0i<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(memPool.ctxs, memPool.hash_space);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Efficient memory copying with streams
    for (int i = 0; i < BATCH_SIZE; i++) {
        CUDA_CHECK(cudaMemcpyAsync(out + i * INDEX_SPACE, memPool.hash_space[i], INDEX_SPACE * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

__global__ void do_hash_stage0i(hashx_ctx** ctxs, uint64_t** hash_space) {
    uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item < BATCH_SIZE * INDEX_SPACE) {
        uint32_t batch_idx = item / INDEX_SPACE;
        uint32_t i = item % INDEX_SPACE;
        hash_stage0i(ctxs[batch_idx], hash_space[batch_idx], i);
    }
}

extern "C" void solve_all_stages(uint64_t *hashes, uint8_t *out, uint32_t *sols, int num_sets) {
    uint64_t *d_hashes;
    solver_heap *d_heaps;
    equix_solution *d_solutions;
    uint32_t *d_num_sols;

    CUDA_CHECK(cudaMalloc(&d_hashes, num_sets * INDEX_SPACE * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_heaps, num_sets * sizeof(solver_heap)));
    CUDA_CHECK(cudaMalloc(&d_solutions, num_sets * EQUIX_MAX_SOLS * sizeof(equix_solution)));
    CUDA_CHECK(cudaMalloc(&d_num_sols, num_sets * sizeof(uint32_t)));

    equix_solution *h_solutions;
    uint32_t *h_num_sols;
    CUDA_CHECK(cudaHostAlloc(&h_solutions, num_sets * EQUIX_MAX_SOLS * sizeof(equix_solution), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_num_sols, num_sets * sizeof(uint32_t), cudaHostAllocDefault));

    CUDA_CHECK(cudaMemcpy(d_hashes, hashes, num_sets * INDEX_SPACE * sizeof(uint64_t), cudaMemcpyHostToDevice));

    int threadsPerBlock = 512;
    int blocksPerGrid = (num_sets + threadsPerBlock - 1) / threadsPerBlock;

    solve_all_stages_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_hashes, d_heaps, d_solutions, d_num_sols);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_solutions, d_solutions, num_sets * EQUIX_MAX_SOLS * sizeof(equix_solution), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_num_sols, d_num_sols, num_sets * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_sets; i++) {
        sols[i] = h_num_sols[i];
        if (h_num_sols[i] > 0) {
            memcpy(out + i * sizeof(equix_solution), &h_solutions[i * EQUIX_MAX_SOLS], sizeof(equix_solution));
        }
    }

    CUDA_CHECK(cudaFree(d_hashes));
    CUDA_CHECK(cudaFree(d_heaps));
    CUDA_CHECK(cudaFree(d_solutions));
    CUDA_CHECK(cudaFree(d_num_sols));

    CUDA_CHECK(cudaFreeHost(h_solutions));
    CUDA_CHECK(cudaFreeHost(h_num_sols));
}
