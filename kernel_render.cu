#include "kernel_shared.cuh"
#include "kernel_render.cuh"
#include "cuda_globals.cuh"
#include "globals.cuh"
#include "const.cuh"

static uchar4* d_displayBufferCuda = NULL;

__device__ uchar4 COLORS[] = {
    {0,   0,   0,   255}, // CELL_EMPTY
    {0,   0,   255, 255}, // CELL_ELECTRON_HEAD
    {255, 0,   0,   255}, // CELL_ELECTRON_TAIL
    {255, 255, 0,   255}, // CELL_CONDUCTOR
};

__global__ void drawCell(int width, char* field, uchar4* drawBuffer)
{
    int offset = getFieldOffsetAt(0, 0, width);
    drawBuffer[offset] = COLORS[field[offset]];
}

void runDrawCell(void)
{
    cudaGraphicsMapResources(1, &displayBufferCuda, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_displayBufferCuda, &num_bytes, displayBufferCuda);
    drawCell<<<numBlocks, threadsPerBlock>>>(width, d_outfield, d_displayBufferCuda);
    cudaGraphicsUnmapResources(1, &displayBufferCuda, 0);

    cudaDeviceSynchronize();
}
