# Texture Elevation Mapping by Parallel Ray Marching

Ray marching technique for generating images of a 3D function with an input image texture using `C++` and `CUDA`.

squares_cpu_8192           | lena_ripples
:-------------------------:|:-------------------------:
<img src='./data/squares_cpu_8192.png' align="left" width=256> | <img src='./data/lena_ripples.png' align="right" width=256>

### Build Source
To build the project from source, be in the top level directory of this repository and run
```
make
```
This project requires the `C` library `libpng`. Compile and link flags can be updated in the `Makefile`. See [libpng](http://www.libpng.org/pub/png/libpng.html) for handy PNG handling in C.

GPU architecture flags may need to be updated for your setup as well. Again, this can be done in the `Makefile`.

### Running the Program
After building, run the program using the following command format
```
./main <input texture png file> <output file> <output file height> <output file rows> <do serial>
```
The `do serial` flag tells the program to either run the serial CPU implementation, or the parallel GPU implementation. Pass `1` for CPU, `0` for the screamin' demon GPU implementation.  

To run the program with the provided test image `squares.png` using the GPU implementation to get an output image that was 1024x1024 pixels, you would run

```
./main ./data/squares.png output.png 1024 1024 0
```

#### Have Fun

You can edit the 3D camera and ray marching parameters in `util.h`, and the GPU settings in both the `Makefile` or `march_kernel.h`. Elevation functions can be defined in `march_kernel.h` and `march_host.cpp`.
