#include "const.cuh"
#include "globals.cuh"
#include "cuda_globals.cuh"
#include "kernel_render.cuh"
#include "kernel_compute.cuh"

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

static uchar4* displayBuffer = NULL;
static GLuint displayBufferTexturePtr = 0;
static GLuint displayBufferPtr = 0;

static void renderFrame()
{
    glColor3f(1.0f, 1.0f, 1.0f);
    glBindTexture(GL_TEXTURE_2D, displayBufferTexturePtr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, displayBufferPtr);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(float(width), 0.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(float(width), float(height));
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0.0f, float(height));
    glEnd();

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glutSwapBuffers();
    glutPostRedisplay();
}

static void displayCallback()
{
    runDrawCell();
    renderFrame();
}

static void idleCallback()
{
    runComputeCellFor(10);
    glutPostRedisplay();
}

int initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("WireWorld CUDA");
    glutDisplayFunc(displayCallback);
    glutIdleFunc(idleCallback);
    
    glewInit();

    // default initialization
    glViewport(0, 0, width, height);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, width, height, 0.0, -1.0, 1.0);

    // texture
    displayBuffer = new uchar4[width * height];
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &displayBufferTexturePtr);
    glBindTexture(GL_TEXTURE_2D, displayBufferTexturePtr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, displayBuffer);

    glGenBuffers(1, &displayBufferPtr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, displayBufferPtr);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(uchar4), displayBuffer, GL_STREAM_COPY);
   
    cudaGraphicsGLRegisterBuffer(&displayBufferCuda, displayBufferPtr, cudaGraphicsMapFlagsWriteDiscard);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

    glutMainLoop();

    return 0;
}

void deinitGL()
{
    if (displayBufferCuda) {
        cudaGraphicsUnregisterResource(displayBufferCuda);
        displayBufferCuda = NULL;
    }
    if (displayBufferPtr) {
        glDeleteBuffers(1, &displayBufferPtr);
        displayBufferPtr = 0;
    }
    if (displayBufferTexturePtr) {
        glDeleteTextures(1, &displayBufferTexturePtr);
        displayBufferTexturePtr = 0;
    }
}
