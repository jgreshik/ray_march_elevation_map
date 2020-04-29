#include <cmath>

#include "vec.h"

vec cross(vec a, vec b){
    vec ret;
    ret.x=a.y*b.z-a.z*b.y;
    ret.y=-a.x*b.z+a.z*b.x;
    ret.z=a.x*b.y-a.y*b.x;
    return ret;
}
double dot(vec a, vec b){
    return a.x*b.x+a.y*b.y+a.z*b.z;
}
double length(vec a){
    return sqrt(a.x*a.x+a.y*a.y+a.z*a.z);
}
vec normalize(vec a){
    return a*(1/length(a));
}
