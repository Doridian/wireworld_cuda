#include "kernel_shared.cuh"
#include "kernel_render.cuh"
#include "cuda_globals.cuh"
#include "globals.cuh"
#include "const.cuh"

static uchar4* d_displayBufferCuda = NULL;

__global__ void drawCell(int width, char* field, uchar4* drawBuffer)
{
    int offset = getFieldOffsetAt(0, 0, width);
    switch (field[offset])
    {
        case CELL_ELECTRON_HEAD:
            drawBuffer[offset].x = 0;
            drawBuffer[offset].y = 0;
            drawBuffer[offset].z = 255;
            break;
        case CELL_ELECTRON_TAIL:
            drawBuffer[offset].x = 255;
            drawBuffer[offset].y = 0;
            drawBuffer[offset].z = 0;
            break;
        case CELL_CONDUCTOR:
            drawBuffer[offset].x = 255;
            drawBuffer[offset].y = 255;
            drawBuffer[offset].z = 0;
            break;
        default:
            drawBuffer[offset].x = 0;
            drawBuffer[offset].y = 0;
            drawBuffer[offset].z = 0;
            break;
    }
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
