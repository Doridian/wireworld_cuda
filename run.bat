del wireworld.exe wireworld
nvcc -O3 -o wireworld globals.cu cuda_globals.cu kernel_render.cu kernel_compute.cu main.cu gl.cu -IC:\OpenGL\include -LC:\OpenGL\lib\x64 -lfreeglut -lglew32
wireworld.exe primes.wi
