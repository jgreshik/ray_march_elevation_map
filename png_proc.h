#pragma once
// http://zarb.org/~gc/html/libpng.html
#include <png.h>

#include "vec.h"

class PNGProc{
    public:
        int width, height;
        png_byte color_type;
        png_byte bit_depth;
        png_bytep *row_pointers = NULL;
        
        void allo_mem();
        void set_pixel(vec color, int i, int j);
        vec get_pixel(int i, int j);
        void read_png_file(char *filename);
        void write_png_file(char *filename);
        void process_png_file();
        void png_to_uint_array(unsigned int*data_array);
        void uint_array_to_png(unsigned int*data_array);
};

