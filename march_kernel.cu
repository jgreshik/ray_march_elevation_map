#include <cmath>

#include "march_kernel.h"
#include "util.h"

__device__ double elev_function(double x, double z){
//    return 0;
    return sin(x)*sin(z);
//    return -1/sqrt((x/2)*(x/2)+(z/2)*(z/2));
}

__device__ double3 raymarch(double3 pix_pos, double3 pix_dir){
    double mint=MIN_T;
    double dt=DT;
    double max_dist=MAX_DIST;
    for(double t=mint;t<max_dist;t+=dt){

        double pos_x=pix_pos.x+pix_dir.x*t;
        double pos_y=pix_pos.y+pix_dir.y*t;
        double pos_z=pix_pos.z+pix_dir.z*t;

        double elev_val=elev_function(pos_x,pos_z);
        
        double3 return_pos;
        return_pos.x=pos_x;
        return_pos.y=pos_y;
        return_pos.z=pos_z;

        dt*=CONSTANT_FACTOR;

        if(pos_y<elev_val)return return_pos;
    }
    double3 no_val;
    no_val.x=10000;
    no_val.y=10000;
    no_val.z=10000;
    return no_val;
}

__global__ void color_pixels(unsigned int*in_png, int in_width, int in_height, 
        unsigned int*out_png, int out_width, int out_height){

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

    int img_world_i=pos.x/5.*in_width;
    int img_world_j=pos.z/5.*in_height;
    img_world_i=abs(img_world_i)%(in_width);
    img_world_j=abs(img_world_j)%(in_height);

    int buffer=5;
    int line_thresh=3;
    double grid = double(in_width) / double(buffer);

    unsigned int alpha_zeros=0xFF000000;
    if (fmod(img_world_i,grid)<line_thresh || fmod(img_world_j,grid)<line_thresh) memcpy(&out_png[j*out_width+i],&alpha_zeros,sizeof(unsigned int));
    else if (pos.x>9999) memcpy(&out_png[j*out_width+i],&alpha_zeros,sizeof(unsigned int));
    else memcpy(&out_png[j*out_width+i],&in_png[img_world_j*in_width+img_world_i],sizeof(unsigned int));
}
