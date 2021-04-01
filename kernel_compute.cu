#include "kernel_shared.cuh"
#include "kernel_compute.cuh"
#include "cuda_globals.cuh"
#include "globals.cuh"
#include "const.cuh"

#include <stdio.h>
#include <chrono>

__device__ static char getFieldAtIsHead(int x, int y, int width, int height, char* field)
{
    int offset = getFieldOffsetAt(x, y, width, height);
    if (offset < 0) {
        return 0;
    }
    return field[offset] == CELL_ELECTRON_HEAD;
}

__global__ void computeCell(int width, int height, char* field, char* outfield)
{
    char tmp;
    int offset = getFieldOffsetAt(0, 0, width, 0);

    switch (field[offset])
    {
        case CELL_ELECTRON_HEAD:
            outfield[offset] = CELL_ELECTRON_TAIL;
            break;
        case CELL_ELECTRON_TAIL:
            outfield[offset] = CELL_CONDUCTOR;
            break;
        case CELL_CONDUCTOR:
            tmp = 
                getFieldAtIsHead(-1, -1, width, height, field) +
                getFieldAtIsHead(-1, 0, width, height, field) +
                getFieldAtIsHead(-1, 1, width, height, field) +
                getFieldAtIsHead(0, -1, width, height, field) +
                getFieldAtIsHead(0, 1, width, height, field) +
                getFieldAtIsHead(1, -1, width, height, field) +
                getFieldAtIsHead(1, 0, width, height, field) +
                getFieldAtIsHead(1, 1, width, height, field);
            if (tmp == 1 || tmp == 2) {
                outfield[offset] = CELL_ELECTRON_HEAD;
                break;
            }
            outfield[offset] = CELL_CONDUCTOR;
            break;
    }
}

void runComputeCell(int iterations)
{
    for (int i = 0; i < iterations; i++) {
        computeCell<<<numBlocks, threadsPerBlock>>>(width, height, d_field, d_outfield);
        std::swap(d_outfield, d_field);
    }
    cudaDeviceSynchronize();
}

static int timedIterations = 100;
void runComputeCellFor(float msTarget)
{
    auto t1 = std::chrono::high_resolution_clock::now();
    runComputeCell(timedIterations);
    auto t2 = std::chrono::high_resolution_clock::now();

    float msActual = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;
    timedIterations = timedIterations * (msTarget / msActual);
    if (timedIterations < 1) {
        timedIterations = 1;
    }
}
