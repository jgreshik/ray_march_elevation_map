#pragma once

struct vec{
    double x;
    double y;
    double z;

    vec(){
        this->x=0;
        this->y=0;
        this->z=0;
    }
    vec(double x,double y,double z){
        this->x=x;
        this->y=y;
        this->z=z;
    }

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
    vec operator *(double b)
    {
        vec ret;
        ret.x=b*x;
        ret.y=b*y;
        ret.z=b*z;
        return ret;
    }
};
vec cross(vec a, vec b);
double dot(vec a, vec b);
double length(vec a);
vec normalize(vec a);
