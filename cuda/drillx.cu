#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "drillx.h"
#include "equix.h"
#include "hashx.h"
#include "equix/src/context.h"
#include "equix/src/solver.h"
#include "equix/src/solver_heap.h"
#include "hashx/src/context.h"

const int BATCH_SIZE = 4096;
const int THREADS_PER_BLOCK = 1024;
const int BUFFER_SIZE = 2; // Double buffering

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err); \
        } \
    } while (0)

// Kernel for processing the hashing stage
__global__ void do_hash_stage0i(hashx_ctx** ctxs, uint64_t* hash_space, int index_space_size) {
    extern __shared__ uint64_t shared_hashes[];

    uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;

    if (item < index_space_size) {
        uint32_t batch_idx = item / index_space_size;
        uint32_t i = item % index_space_size;

        // Perform the hashing operation and store the result in shared memory
        hash_stage0i(ctxs[batch_idx], &shared_hashes[threadIdx.x], i);

        __syncthreads();  // Ensure all threads have completed their operations

        // Copy results from shared memory to global memory
        hash_space[item] = shared_hashes[threadIdx.x];
    }
}

extern "C" void hash(uint8_t *challenge, uint8_t *nonce, uint64_t *out) {
    hashx_ctx** ctxs;
    uint64_t* hash_space[BUFFER_SIZE]; // Double buffering
    cudaStream_t streams[BUFFER_SIZE];

    CUDA_CHECK(cudaMallocHost(&ctxs, BATCH_SIZE * sizeof(hashx_ctx*)));
    for (int i = 0; i < BUFFER_SIZE; i++) {
        CUDA_CHECK(cudaMalloc(&hash_space[i], BATCH_SIZE * INDEX_SPACE * sizeof(uint64_t)));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    uint8_t seed[40];
    memcpy(seed, challenge, 32);

    for (int i = 0; i < BATCH_SIZE; i++) {
        uint64_t nonce_offset = *((uint64_t*)nonce) + i;
        memcpy(seed + 32, &nonce_offset, 8);
        ctxs[i] = hashx_alloc(HASHX_INTERPRETED);
        if (!ctxs[i] || !hashx_make(ctxs[i], seed, 40)) {
            for (int j = 0; j <= i; j++) {
                hashx_free(ctxs[j]);
            }
            CUDA_CHECK(cudaFreeHost(ctxs));
            for (int j = 0; j < BUFFER_SIZE; j++) {
                CUDA_CHECK(cudaFree(hash_space[j]));
                CUDA_CHECK(cudaStreamDestroy(streams[j]));
            }
            return;
        }
    }

    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    int blocksPerGrid = (BATCH_SIZE * INDEX_SPACE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Persistent kernel launch
    for (int iter = 0; iter < 1000; ++iter) {
        int bufferIndex = iter % BUFFER_SIZE;

        do_hash_stage0i<<<blocksPerGrid, threadsPerBlock, THREADS_PER_BLOCK * sizeof(uint64_t), streams[bufferIndex]>>>(ctxs, hash_space[bufferIndex], BATCH_SIZE * INDEX_SPACE);
        CUDA_CHECK(cudaGetLastError());

        // Transfer the data to host memory asynchronously
        CUDA_CHECK(cudaMemcpyAsync(out + bufferIndex * BATCH_SIZE * INDEX_SPACE, hash_space[bufferIndex], BATCH_SIZE * INDEX_SPACE * sizeof(uint64_t), cudaMemcpyDeviceToHost, streams[bufferIndex]));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < BATCH_SIZE; i++) {
        hashx_free(ctxs[i]);
    }

    CUDA_CHECK(cudaFreeHost(ctxs));
    for (int i = 0; i < BUFFER_SIZE; i++) {
        CUDA_CHECK(cudaFree(hash_space[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
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

    int threadsPerBlock = 1024;
    int blocksPerGrid = (num_sets + threadsPerBlock - 1) / threadsPerBlock;
    solve_all_stages_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_hashes, d_heaps, d_solutions, d_num_sols);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the results back to the CPU
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
