#pragma once

__device__ double elev_function(double x, double z);

__device__ double3 raymarch(double3 pix_pos,double3 pix_dir);

__global__ void color_pixels(uint1*in_png, int in_width, int in_height, 
        uint1*out_png, int out_width, int out_height);
