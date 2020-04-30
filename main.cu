#include <png.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>

#include "png_proc.h"
#include "march_host.h"
#include "march_kernel.h"

int main(int argc, char *argv[]) {
    if(argc != 5) abort();

    PNGProc input;
    int height = atoi(argv[3]);
    int width = atoi(argv[4]);
    PNGProc output;

    input.read_png_file(argv[1]);
    output.read_png_file(argv[1]);
    output.width=width;
    output.height=height;
    
    output.allo_mem();
    
    unsigned int*out_data;
    
    unsigned int*in_data=nullptr; 
    in_data=(unsigned int *)malloc(sizeof(unsigned int) * input.width * input.height);

    for (int i = 0; i< 4; ++i) std::cout << input.row_pointers[i] << std::endl;
    std::cout << std::endl;
    for (int i = 0; i< 4; ++i) std::cout << in_data[i] << std::endl;
    std::cout << std::endl;
    input.png_to_uint_array(in_data);
    for (int i = 0; i< 4; ++i) std::cout << in_data[i] << std::endl;

//    int size=input.width*input.height;
//
//    int num_blocks=2;
//    int threads_per_block=2;
//    
//    int*k=new int[1];
//    k[0]=0;
//    int*k_m;
//
//    cudaMalloc(&k_m,sizeof(int));
//    cudaMemcpy(k_m,k,sizeof(int),cudaMemcpyHostToDevice);
//
//    dummy<<<num_blocks,threads_per_block>>>(k_m);
//
//    cudaMemcpy(k, k_m, sizeof(int), cudaMemcpyDeviceToHost);
//
//    std::cout<<"k @ 0: "<<k[0]<<std::endl;

    input.process_png_file();
//    color_out(input,output);
//    output.write_png_file(argv[2]);

    return 0;
}
