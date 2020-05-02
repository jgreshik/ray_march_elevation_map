#pragma once

#define NUM_THREADS 32 

__device__ double elev_function(double x, double z);

__device__ double3 raymarch(double3 pix_pos,double3 pix_dir);

__global__ void color_pixels(unsigned int*in_png, int in_width, int in_height, 
        unsigned int*out_png, int out_width, int out_height);
