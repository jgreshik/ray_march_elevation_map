#include <png.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <stdio.h>

#include "png_proc.h"
#include "vec.h"

#define MAXSTEPS = 255 // change with dt
float elev_function(float x, float z){
    return sin(x)*sin(z);
}
double spheresdf(vec pos,double r){
    return length(pos)-r;
}
double scenesdf(vec pos){
    double m=spheresdf(pos,1);
    
    return m;
}
bool raymarch(vec pix_pos,vec pix_dir){
    vec pos;
    pos.x=pix_pos.x;
    pos.y=pix_pos.y;
    pos.z=pix_pos.z;
    for(int i=0;i<256;++i){
        double v=scenesdf(pos);
        if(v<.001){
            return true;
        }
        pos=pos+pix_dir*v;
    }
    return false;
}
vec raymarch2(vec pix_pos,vec pix_dir){
    double mint=.001;
    double maxt=15;
    double dt=.01;
    for(double t=mint;t<maxt;t+=dt){
        vec pos=pix_pos+pix_dir*t;
        double v=elev_function(pos.x,pos.z);
        if(pos.y<v)return pos;
    }
    return vec(1000,1000,1000);
}

vec color_pixel(int i,int j, PNGProc in_png, PNGProc out_png){
    double x=double(i)/out_png.width;
    double y=double(j)/out_png.height;
    x-=.5;
    y-=double(out_png.height)/double(out_png.width)/2;
    x*=2;
    y*=-2;

    vec eye(8,2,0);
    vec lookat(0,0,0);

    vec eye_dir=normalize(lookat-eye);
    double focal_length=1;
    double pi = 3.14159265358979323846;
    double fov=pi/2;
    vec pix_pos=eye+eye_dir*(focal_length);
    vec up(0,1,0);
    vec right=cross(eye_dir,up);
    pix_pos=pix_pos+right*tan(fov/2)*x*abs(focal_length);
    pix_pos=pix_pos+up*tan(fov/2)*y*abs(focal_length);
    vec pix_dir=normalize(pix_pos-eye);

    vec pos=raymarch2(pix_pos,pix_dir);
    int img_world_i=pos.x/5.*in_png.width;
    int img_world_j=pos.z/5.*in_png.height;
    img_world_i=abs(img_world_i)%(in_png.width);
    img_world_j=abs(img_world_j)%(in_png.height);

    vec image_color=in_png.get_pixel(img_world_i,img_world_j);
//    return length(pos)<100?vec(abs(pos.x)/5,0,abs(pos.z)/5):vec(0,0,0);
    return length(pos)<100?image_color:vec(0,0,0);
}

void fill_elev(int width, int height, float* elev){
    for(float y = 0; y < height; ++y) {
        for(float x = 0; x < width; ++x){
            elev[(int)y*width+(int)x]=elev_function(x/width,y/height);
        }
    }
}

void color(PNGProc in_png,PNGProc out_png){
    for(int y = 0; y < out_png.height; y++) {
        for(int x = 0; x < out_png.width; x++) {
            vec v=color_pixel(x,y,in_png,out_png);
            double red=255*v.x;
            double green=255*v.y;
            double blue=255*v.z;
            out_png.set_pixel(red,green,blue,x,y);
        }
    }
}

//__global__ void dummy(int x, int y, int*k){
//    vec test;
//    test.x=x*2;
//    test.y=y*4;
//    k[0]+=1;
//    printf("test: x=%d y=%d\n",test.x,test.y);
//}

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
    
    int size=input.width*input.height;
    float*elev_data=new float[size];
    fill_elev(input.width,input.height,elev_data);

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

//    input.process_png_file(elev_data);
    color(input,output);
    //png_proc.write_png_file(argv[2]);
    output.write_png_file(argv[2]);

    delete[]elev_data;

    return 0;
}
