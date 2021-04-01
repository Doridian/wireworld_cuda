#include <stdio.h>

#include "cuda_globals.cuh"

#include "gl.cuh"
#include "const.cuh"
#include "globals.cuh"

static char* field = NULL;

static char* getFieldPtrAt(int x, int y)
{
    if (x > width || y > height) {
        return NULL;
    }
    return field + (x + (y * width));
}

static void unloadField()
{
    if (d_outfield) {
        cudaFree(d_outfield);
        d_outfield = NULL;
    }
    if (d_field) {
        cudaFree(d_field);
        d_field = NULL;
    }
    if (field) {
        free(field);
        field = NULL;
    }
}

static int loadFile(const char* fileName)
{
    unloadField();

    int fileWidth, fileHeight;

    FILE* fd = fopen(fileName, "rb");
    if (!fd) {
        return 1;
    }
    fscanf(fd, "%d %d", &fileWidth, &fileHeight);

    printf("Got file dimensions %d / %d\n", fileWidth, fileHeight);

    width = fileWidth + 2;
    height = fileHeight + 2;

    if (width % threadsPerBlock.x) {
        width += threadsPerBlock.x - (width % threadsPerBlock.x);
    }
    if (height % threadsPerBlock.y) {
        height += threadsPerBlock.y - (height % threadsPerBlock.y);
    }

    numBlocks.x = width / threadsPerBlock.x;
    numBlocks.y = height / threadsPerBlock.y;

    size_t memorySize = (size_t)(width * height);

    field = (char*)malloc(memorySize);
    memset(field, CELL_EMPTY, memorySize);

    int charIn;
    char charOut;
    int x = 0;
    int y = 0;
    do {
        charIn = fgetc(fd);
        if (charIn < 0) {
            break;
        }
        if (charIn == '\r' || charIn == '\n') {
            continue;
        }

        switch (charIn) {
            case FILE_CELL_CONDUCTOR:
                charOut = CELL_CONDUCTOR;
                break;
            case FILE_CELL_ELECTRON_HEAD:
                charOut = CELL_ELECTRON_HEAD;
                break;
            case FILE_CELL_ELECTRON_TAIL:
                charOut = CELL_ELECTRON_TAIL;
                break;
            default:
                charOut = CELL_EMPTY;
                break;
        }

        *getFieldPtrAt(x + 1, y + 1) = charOut;
        if (++x >= fileWidth) {
            y++;
            x = 0;
        }
    } while (!feof(fd));

    printf("File loaded: %s\n", fileName);
    fclose(fd);

    cudaMalloc(&d_field, memorySize);
    cudaMalloc(&d_outfield, memorySize);
    cudaMemcpy(d_field, field, memorySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_outfield, field, memorySize, cudaMemcpyHostToDevice);

    return 0;
}

int main(int argc, char** argv)
{
    if (loadFile(argv[1])) {
        unloadField();
        return 1;
    }

    if (initGL(&argc, argv)) {
        unloadField();
        return 1;
    }

    deinitGL();
    unloadField();
}
