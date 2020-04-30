#pragma once

#include "vec.h"
#include "png_proc.h"

float elev_function(float x, float z);

double spheresdf(vec pos,double r);

double scenesdf(vec pos);

bool raymarch_test(vec pix_pos,vec pix_dir);

vec raymarch(vec pix_pos,vec pix_dir);

vec color_pixel(int i,int j, PNGProc in_png, PNGProc out_png);

void color_out(PNGProc in_png,PNGProc out_png);
