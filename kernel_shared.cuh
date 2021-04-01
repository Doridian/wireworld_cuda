#pragma once

__device__ static inline int getFieldOffsetAt(int x, int y, int width)
{
    int posX = x + (blockIdx.x * blockDim.x) + threadIdx.x;
    int posY = y + (blockIdx.y * blockDim.y) + threadIdx.y;
    return posX + (posY * width);
}
