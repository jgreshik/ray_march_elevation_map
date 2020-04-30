#include <cuda_runtime_api.h>
#include <cmath>

#include "march_kernel.h"

// -use_fast_math compiler flag 

__device__ double elev_function(double x, double z){
    return sin(x)*sin(z);
}

__device__ double3 raymarch(double3 pix_pos, double3 pix_dir){
    double mint=.001;
    double dt=.01;
    double max_dist=50;
    for(double t=mint;t<max_dist;t+=dt){

        double pos_x=pix_pos.x+pix_dir.x*t;
        double pos_y=pix_pos.y+pix_dir.y*t;
        double pos_z=pix_pos.z+pix_dir.z*t;

        double elev_val=elev_function(pos_x,pos_z);
        
        double3 return_pos;
        return_pos.x=pos_x;
        return_pos.y=pos_y;
        return_pos.z=pos_z;

        if(pos_y<elev_val)return return_pos;
    }
    double3 no_val;
    no_val.x=10000;
    no_val.y=10000;
    no_val.z=10000;
    return no_val;
}

__global__ void color_pixels(uint1*in_png, int in_width, int in_height, 
        uint1*out_png, int out_width, int out_height){

    int NUM_THREADS=32;

    int i=blockIdx.x*NUM_THREADS+threadIdx.x;
    int j=blockIdx.y*NUM_THREADS+threadIdx.y;

    double x=double(i)/out_width;
    double y=double(j)/out_height;
    x-=.5;
    y-=double(out_height)/double(out_width)/2;
    x*=2;
    y*=-2;

    double3 eye;
    eye.x=3.;
    eye.y=4.;
    eye.z=3.;
    double3 lookat;
    lookat.x=0.;
    lookat.y=0.;
    lookat.z=0.;

    //double3 eye_dir=normalize(lookat-eye);
    double3 eye_dir;
    eye_dir.x=lookat.x-eye.x;
    eye_dir.y=lookat.y-eye.y;
    eye_dir.z=lookat.z-eye.z;

    double eye_dir_mag = sqrt(eye_dir.x*eye_dir.x+eye_dir.y*eye_dir.y+eye_dir.z*eye_dir.z);
    eye_dir.x=eye_dir.x/eye_dir_mag;
    eye_dir.y=eye_dir.y/eye_dir_mag;
    eye_dir.z=eye_dir.z/eye_dir_mag;

    double focal_length=1;
    double pi = 3.14159265358979323846;
    double fov=pi/2;

    //vec pix_pos=eye+eye_dir*(focal_length);
    double3 pix_pos;
    pix_pos.x=eye.x+eye_dir.x*focal_length;
    pix_pos.y=eye.y+eye_dir.y*focal_length;
    pix_pos.z=eye.z+eye_dir.z*focal_length;

    //vec up=vec(-eye_dir.x*eye_dir.y,1-eye_dir.y*eye_dir.y,-eye_dir.z*eye_dir.y)/sqrt(1-eye_dir.y*eye_dir.y);
    double3 up;
    up.x=(-eye_dir.x*eye_dir.y)/sqrt(1-eye_dir.y*eye_dir.y);
    up.y=(1-eye_dir.y*eye_dir.y)/sqrt(1-eye_dir.y*eye_dir.y);
    up.z=(-eye_dir.z*eye_dir.y)/sqrt(1-eye_dir.y*eye_dir.y);

    //vec right=cross(eye_dir,up);
    double3 right;
    right.x=eye_dir.y*up.z-eye_dir.z*up.y;
    right.y=-eye_dir.x*up.z+eye_dir.z*up.x,
    right.z=eye_dir.x*up.y-eye_dir.y*up.x;
    
    //pix_pos=pix_pos+right*tan(fov/2)*x*abs(focal_length);
    pix_pos.x=pix_pos.x+right.x*tan(fov/2)*x*abs(focal_length);
    pix_pos.y=pix_pos.y+right.y*tan(fov/2)*x*abs(focal_length);
    pix_pos.z=pix_pos.z+right.z*tan(fov/2)*x*abs(focal_length);

    //pix_pos=pix_pos+up*tan(fov/2)*y*abs(focal_length);
    pix_pos.x=pix_pos.x+up.x*tan(fov/2)*y*abs(focal_length);
    pix_pos.y=pix_pos.y+up.y*tan(fov/2)*y*abs(focal_length);
    pix_pos.z=pix_pos.z+up.z*tan(fov/2)*y*abs(focal_length);

    //vec pix_dir=normalize(pix_pos-eye);
    double3 pix_dir;
    pix_dir.x=pix_pos.x-eye.x;
    pix_dir.y=pix_pos.y-eye.y;
    pix_dir.z=pix_pos.z-eye.z;
    double pix_dir_mag = sqrt(pix_dir.x*pix_dir.x+pix_dir.y*pix_dir.y+pix_dir.z*pix_dir.z);
    pix_dir.x=pix_dir.x/pix_dir_mag;
    pix_dir.y=pix_dir.y/pix_dir_mag;
    pix_dir.z=pix_dir.z/pix_dir_mag;

    double3 pos=raymarch(pix_pos,pix_dir);

    unsigned int img_world_i=pos.x/5.*in_width;
    unsigned int img_world_j=pos.z/5.*in_height;
    img_world_i=img_world_i%(in_width);
    img_world_j=img_world_j%(in_height);

    //vec image_color=in_png.get_pixel(img_world_i,img_world_j);
//    return length(pos)<100?vec(abs(pos.x)/5,0,abs(pos.z)/5):vec(0,0,0);
    //return length(pos)<100?image_color:vec(0,0,0);

    //int buffer=5;
    //int line_thresh=3;
    //double3 grid_color = vec(0,0,0);
    //double grid = double(width) / double(buffer);
    //if (fmod(i,grid)<line_thresh || fmod(j,grid)<line_thresh) return grid_color;
    //png_bytep row = row_pointers[j];
    //png_bytep px = &(row[i * 4]);
    //vec ret;
    //ret.x=double(px[0])/255;
    //ret.y=double(px[1])/255;
    //ret.z=double(px[2])/255;
    unsigned int alpha_zeros=0xFF000000;
    if (pos.x>9999) memcpy(&out_png[j*out_width+i],&alpha_zeros,sizeof(unsigned int));
    else memcpy(&out_png[j*out_width+i],&in_png[img_world_j*in_width+img_world_i],sizeof(uint1));
}

//__global__ void color_out(PNGProc in_png,PNGProc out_png){
//    for(int y = 0; y < out_png.height; y++) {
//        for(int x = 0; x < out_width; x++) {
//            vec v=color_pixel(x,y,in_png,out_png);
//            out_png.set_pixel(v*255,x,y);
//        }
//    }
//}

//double spheresdf(vec pos,double r){
//    return length(pos)-r;
//}
//
//double scenesdf(vec pos){
//    double m=spheresdf(pos,1);
//    return m;
//}
//
//bool raymarch_test(vec pix_pos,vec pix_dir){
//    vec pos;
//    pos.x=pix_pos.x;
//    pos.y=pix_pos.y;
//    pos.z=pix_pos.z;
//    for(int i=0;i<256;++i){
//        double v=scenesdf(pos);
//        if(v<.001){
//            return true;
//        }
//        pos=pos+pix_dir*v;
//    }
//    return false;
//}
//
//__global__ void dummy(int*k){
//    int j = blockIdx.x*blockDim.x+threadIdx.x;
//    int i = blockIdx.y*blockDim.y+threadIdx.y;
//    k[0]+=1;
//    printf("test: x=%d y=%d\n",i,j);
//}
//
