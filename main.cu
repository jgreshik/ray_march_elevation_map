#include "png_proc.h"
#include <iostream>
#include <cmath>
#include <stdio.h>

float elev_function(float x, float z){
    return sin(x)*sin(z);
}

void fill_elev(int width, int height, float* elev){
    for(float y = 0; y < height; ++y) {
        for(float x = 0; x < width; ++x){
            elev[(int)y*width+(int)x]=elev_function(x/width,y/height);
        }
    }
}

struct vec{
    int x=0;
    int y=0;
    int z=0;
    vec operator +(vec b)
    {
        vec ret;
        ret.x=x+b.x;
        ret.y=y+b.y;
        ret.z=z+b.z;
        return ret;
    }
    vec operator -(vec b)
    {
        vec ret;
        ret.x=x-b.x;
        ret.y=y-b.y;
        ret.z=z-b.z;
        return ret;
    }
};

vec cross(vec a, vec b){
    vec ret;
    ret.x=a.y*b.z-a.z*b.y;
    ret.y=a.x*b.z-a.z*b.x;
    ret.z=a.x*b.y-a.y*b.x;
    return ret;
}

int dot(vec a, vec b){
    return a.x*b.x+a.y*b.y+a.z*b.z;
}

//__global__ void dummy(int x, int y, int*k){
//    vec test;
//    test.x=x*2;
//    test.y=y*4;
//    k[0]+=1;
//    printf("test: x=%d y=%d\n",test.x,test.y);
//}

int main(int argc, char *argv[]) {
    if(argc != 3) abort();

    PNGProc png_proc;

    png_proc.read_png_file(argv[1]);

    int size=png_proc.width*png_proc.height;
    float*elev_data=new float[size];
    fill_elev(png_proc.width,png_proc.height,elev_data);

//    int num_blocks=16;
//    int threads_per_block=16;
//    std::cout<<threads_per_block<<std::endl;
//    
//    int*k=new int[1];
//    int*k_m;
//
//    cudaMalloc(&k_m,sizeof(int));
//    cudaMemcpy(k_m,k,sizeof(int),cudaMemcpyHostToDevice);
//
//    dummy<<<num_blocks,threads_per_block>>>(3,4,k_m);
//
//    cudaMemcpy(k, k_m, sizeof(int), cudaMemcpyDeviceToHost);
//
//    std::cout<<"k @ 0: "<<k[0]<<std::endl;

    png_proc.process_png_file(elev_data);
    png_proc.write_png_file(argv[2]);

    delete[]elev_data;

    return 0;
}
