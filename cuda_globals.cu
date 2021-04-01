#include "cuda_globals.cuh"

struct cudaGraphicsResource *displayBufferCuda;
char* d_field = NULL;
char* d_outfield = NULL;
dim3 threadsPerBlock(16,16);
dim3 numBlocks;
