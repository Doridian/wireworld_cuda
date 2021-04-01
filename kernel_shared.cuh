#pragma once

__device__ static int getFieldOffsetAt(int x, int y, int width, int height)
{
    int posX = x + (blockIdx.x * blockDim.x) + threadIdx.x;
    int posY = y + (blockIdx.y * blockDim.y) + threadIdx.y;
    if (height && (posX < 0 || posY < 0 || posX >= width || posY >= height)) {
        return -1;
    }
    return posX + (posY * width);
}
