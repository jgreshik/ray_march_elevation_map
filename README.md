# Texture Elevation Mapping by Parallel Ray Marching

Ray marching technique for generating images of a 3D functions with an input image texture using `C` and `CUDA`.

<img src='./data/squares_cpu_8192.png' align="center" width=1024>

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

To run the program with the provided test image, `squares.png`, using the GPU implementation, you would run

```
./main ./data/squares.png output.png 0
```

#### Have Fun

I really enjoyed making this and trying all sorts of 3D functions and textures out. You can edit the 3D camera and ray marching parameters in `util.h`, and the GPU settings in both the `Makefile` or `march_kernel.h`.
