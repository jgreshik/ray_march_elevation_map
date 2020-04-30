#include <cuda_runtime_api.h>
#include <png.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>

#include "png_proc.h"
#include "march_host.h"
#include "march_kernel.h"

int main(int argc, char *argv[]) {
    if(argc != 6) abort();

    PNGProc input;
    int out_height = atoi(argv[3]);
    int out_width = atoi(argv[4]);
    PNGProc output;

    input.read_png_file(argv[1]);
    output.read_png_file(argv[1]);
    output.width=out_width;
    output.height=out_height;
    
    output.allo_mem();

    if(atoi(argv[5])==1){
        printf("Running serial version.\n");
        color_out(input,output);
        output.write_png_file(argv[2]);
        return 0;
    }

    printf("Running parallel version.\n");
    
    unsigned int*out_data=nullptr;
    out_data=(unsigned int *)malloc(sizeof(unsigned int) * output.width * output.height);
    
    unsigned int*in_data=nullptr; 
    in_data=(unsigned int *)malloc(sizeof(unsigned int) * input.width * input.height);
    input.png_to_uint_array(in_data);

    unsigned int*out_data_m;
    unsigned int*in_data_m;

    cudaMalloc(&out_data_m,sizeof(unsigned int) * output.width * output.height);
    cudaMalloc(&in_data_m,sizeof(unsigned int) * input.width * input.height);

    cudaMemcpy(in_data_m,in_data,sizeof(unsigned int) * input.width * input.height,cudaMemcpyHostToDevice);
    
    dim3 threads_per_block(NUM_THREADS,NUM_THREADS,1);
    dim3 num_blocks(
            (out_width+threads_per_block.x-1)/threads_per_block.x,
            (out_height+threads_per_block.y-1)/threads_per_block.y,
            1);

    color_pixels<<<num_blocks,threads_per_block>>>(in_data_m, input.width, input.height, 
            out_data_m, output.width, output.height);

    cudaMemcpy(out_data,out_data_m,sizeof(unsigned int) * output.width * output.height,cudaMemcpyDeviceToHost);

    output.uint_array_to_png(out_data);
    output.write_png_file(argv[2]);

    return 0;
}
