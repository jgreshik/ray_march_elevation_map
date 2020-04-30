#include <png.h>
#include <cmath>

#include "vec.h"
#include "png_proc.h"
#include "util.h"

float elev_function(float x, float z){
//    return 0;
    return sin(x)*sin(z);
}

vec raymarch(vec pix_pos,vec pix_dir){
    double mint=MIN_T;
    double dt=DT;
    double max_dist=MAX_DIST;
    for(double t=mint;t<max_dist;t+=dt){
        vec pos=pix_pos+pix_dir*t;
        double v=elev_function(pos.x,pos.z);
        dt*=CONSTANT_FACTOR;
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

    vec eye(3,4,3);
    vec lookat(0,0,0);

    vec eye_dir=normalize(lookat-eye);
    double focal_length=1;
    double pi = 3.14159265358979323846;
    double fov=pi/2;
    vec pix_pos=eye+eye_dir*(focal_length);
    vec up=vec(-eye_dir.x*eye_dir.y,1-eye_dir.y*eye_dir.y,-eye_dir.z*eye_dir.y)/sqrt(1-eye_dir.y*eye_dir.y);
    vec right=cross(eye_dir,up);
    pix_pos=pix_pos+right*tan(fov/2)*x*abs(focal_length);
    pix_pos=pix_pos+up*tan(fov/2)*y*abs(focal_length);
    vec pix_dir=normalize(pix_pos-eye);

    vec pos=raymarch(pix_pos,pix_dir);
    int img_world_i=pos.x/5.*in_png.width;
    int img_world_j=pos.z/5.*in_png.height;
    img_world_i=abs(img_world_i)%(in_png.width);
    img_world_j=abs(img_world_j)%(in_png.height);

    vec image_color=in_png.get_pixel(img_world_i,img_world_j);
//    return length(pos)<100?vec(abs(pos.x)/5,0,abs(pos.z)/5):vec(0,0,0);
    return length(pos)<100?image_color:vec(0,0,0);
}

void color_out(PNGProc in_png,PNGProc out_png){
    for(int y = 0; y < out_png.height; y++) {
        for(int x = 0; x < out_png.width; x++) {
            vec v=color_pixel(x,y,in_png,out_png);
            out_png.set_pixel(v*255,x,y);
        }
    }
}

double spheresdf(vec pos,double r){
    return length(pos)-r;
}

double scenesdf(vec pos){
    double m=spheresdf(pos,1);
    return m;
}

bool raymarch_test(vec pix_pos,vec pix_dir){
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
