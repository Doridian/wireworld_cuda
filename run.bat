del wireworld.exe wireworld
nvcc -o wireworld globals.cu cuda_globals.cu kernel_render.cu kernel_compute.cu main.cu gl.cu -lfreeglut -lglew32
wireworld.exe primes.wi

