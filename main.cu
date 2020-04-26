#include "png_proc.h"

int main(int argc, char *argv[]) {
    if(argc != 3) abort();

    PNGProcessor png_proc;

    png_proc.read_png_file(argv[1]);
    png_proc.process_png_file();
    png_proc.write_png_file(argv[2]);

    return 0;
}
