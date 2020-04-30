// http://zarb.org/~gc/html/libpng.html
#include <stdlib.h>
#include <cmath>
#include <stdio.h>
#include <png.h>
#include <cstring>

#include "png_proc.h"

void PNGProc::read_png_file(char *filename) {
    FILE *fp = fopen(filename, "rb");

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if(!png) abort();

    png_infop info = png_create_info_struct(png);
    if(!info) abort();

    if(setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, fp);

    png_read_info(png, info);

    width      = png_get_image_width(png, info);
    height     = png_get_image_height(png, info);
    color_type = png_get_color_type(png, info);
    bit_depth  = png_get_bit_depth(png, info);

    // Read any color_type into 8bit depth, RGBA format.
    // See http://www.libpng.org/pub/png/libpng-manual.txt

    if(bit_depth == 16)
        png_set_strip_16(png);

    if(color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
    if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if(png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    // These color_type don't have an alpha channel then fill it with 0xff.
    if(color_type == PNG_COLOR_TYPE_RGB ||
            color_type == PNG_COLOR_TYPE_GRAY ||
            color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    if(color_type == PNG_COLOR_TYPE_GRAY ||
            color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    if (row_pointers) abort();

    row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for(int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
    }

    png_read_image(png, row_pointers);

    fclose(fp);

    png_destroy_read_struct(&png, &info, NULL);
}

void PNGProc::write_png_file(char *filename) {
    FILE *fp = fopen(filename, "wb");
    if(!fp) abort();

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) abort();

    png_infop info = png_create_info_struct(png);
    if (!info) abort();

    if (setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, fp);

    // Output is 8bit depth, RGBA format.
    png_set_IHDR(
            png,
            info,
            width, height,
            8,
            PNG_COLOR_TYPE_RGBA,
            PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_DEFAULT,
            PNG_FILTER_TYPE_DEFAULT
            );
    png_write_info(png, info);

    // To remove the alpha channel for PNG_COLOR_TYPE_RGB format,
    // Use png_set_filler().
    //png_set_filler(png, 0, PNG_FILLER_AFTER);

    if (!row_pointers) abort();

    png_write_image(png, row_pointers);
    png_write_end(png, NULL);

    for(int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);

    fclose(fp);

    png_destroy_write_struct(&png, &info);
}

void PNGProc::set_pixel(vec color, int x, int y) {
    png_bytep row = row_pointers[y];
    png_bytep px = &(row[x * 4]);
    px[0]=double(color.x);
    px[1]=double(color.y);
    px[2]=double(color.z);
    px[3]=255;
}

vec PNGProc::get_pixel(int i, int j) {
    int buffer=5;
    int line_thresh=3;
    vec grid_color = vec(0,0,0);
    double grid = double(width) / double(buffer);
    if (fmod(i,grid)<line_thresh || fmod(j,grid)<line_thresh) return grid_color;
    png_bytep row = row_pointers[j];
    png_bytep px = &(row[i * 4]);
    vec ret;
    ret.x=double(px[0])/255;
    ret.y=double(px[1])/255;
    ret.z=double(px[2])/255;
    return ret;
}

void PNGProc::allo_mem() {
    row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for(int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)malloc(sizeof(png_bytep) * width);
    }
}

void PNGProc::png_to_uint_array(unsigned int*data_array){
    for(int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for(int x = 0; x < width; x++) {
            png_bytep px = &(row[x * 4]);
            memcpy(&data_array[y*width+x],px,sizeof(unsigned int));
        }
    }
}

void PNGProc::uint_array_to_png(unsigned int*data_array){
    for(int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for(int x = 0; x < width; x++) {
            png_bytep px = &(row[x * 4]);
            memcpy(px,&data_array[y*width+x],sizeof(unsigned int));
        }
    }
}

//            png_bytep px = &(row[x * 4]);
//            unsigned int b=0xFF0000FF;
//            row[x*4]=(data_array[y*width+x]&0xFF000000)>>24;// 255;
//            row[x*4+1]=(data_array[y*width+x]&0x00FF0000)>>16;//0;
//            row[x*4+2]=(data_array[y*width+x]&0x0000FF00)>>8;//255;
//            row[x*4+3]=255;
void PNGProc::process_png_file() {
    for(int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for(int x = 0; x < width; x++) {
            if (y%(height-1)==0 && x%(width-1)==0){
                //png_bytep px = &(row[x * 4]);
                unsigned int b;
                memcpy(&b,&row[y*width+x],4);
                printf("%4d, %4d = %u\n", x, y, b);
                //printf("%4d, %4d = RGBA(%3d, %3d, %3d, %3d)\n", x, y, px[0], px[1], px[2], px[3]);
            }
//            png_bytep px = &(row[x * 4]);
//            int temp=px[0];
//            row[x*4]=px[2];
//            row[x*4+2]=temp;
//            row[x*4+3]=(int)(row[x*4+3]*elev[width*y+x]);
        }
    }
}
