#include "kernel_shared.cuh"
#include "kernel_compute.cuh"
#include "cuda_globals.cuh"
#include "globals.cuh"
#include "const.cuh"

#include <stdio.h>
#include <chrono>

#define FIELD_AT_IS_HEAD(O) (field[O] == CELL_ELECTRON_HEAD)

__global__ void computeCell(const int width, const char* field, char* outfield)
{
    const int offset = getFieldOffsetAt(0, 0, width);

    switch (field[offset])
    {
        case CELL_ELECTRON_HEAD:
            outfield[offset] = CELL_ELECTRON_TAIL;
            break;
        case CELL_ELECTRON_TAIL:
            outfield[offset] = CELL_CONDUCTOR;
            break;
        case CELL_CONDUCTOR: {
            int cubeOffset = offset - (1 + width);
            char neighbourElectronHeads =
                FIELD_AT_IS_HEAD(cubeOffset) +
                FIELD_AT_IS_HEAD(cubeOffset + 1) +
                FIELD_AT_IS_HEAD(cubeOffset + 2);
        
            cubeOffset += width;
            neighbourElectronHeads +=
                FIELD_AT_IS_HEAD(cubeOffset) +
                FIELD_AT_IS_HEAD(cubeOffset + 2);
        
            cubeOffset += width;
            neighbourElectronHeads +=
                FIELD_AT_IS_HEAD(cubeOffset) +
                FIELD_AT_IS_HEAD(cubeOffset + 1) +
                FIELD_AT_IS_HEAD(cubeOffset + 2);

            outfield[offset] = (neighbourElectronHeads == 1 || neighbourElectronHeads == 2) ? CELL_ELECTRON_HEAD : CELL_CONDUCTOR;
            break;
        }
    }
}

void runComputeCell(int iterations)
{
    for (int i = 0; i < iterations; i++) {
        std::swap(d_outfield, d_field);
        computeCell<<<numBlocks, threadsPerBlock>>>(width, d_field, d_outfield);
    }
    cudaDeviceSynchronize();
}

static int timedIterations = 100;
void runComputeCellFor(float msTarget)
{
    runComputeCell(1);
    auto t1 = std::chrono::high_resolution_clock::now();
    runComputeCell(timedIterations);
    auto t2 = std::chrono::high_resolution_clock::now();

    float msActual = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;
    timedIterations = (int)(timedIterations * (msTarget / msActual));
    if (timedIterations < 1) {
        timedIterations = 1;
    }
}
