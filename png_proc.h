#ifndef PNG_PROC_H
#define PNG_PROC_H

#include <png.h>

class PNGProcessor{
    public:
        int width, height;
        png_byte color_type;
        png_byte bit_depth;
        png_bytep *row_pointers = NULL;

        void read_png_file(char *filename);
        void write_png_file(char *filename);
        void process_png_file();
};

#endif
