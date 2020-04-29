CXX := nvcc
OPT_FLAGS := -O3 -G -g 
INC_FLAGS := /usr/lib/
# GENCODE for gpu server
#GENCODE := -gencode arch=compute_70,code=compute_70 -gencode arch=compute_75,code=compute_75
# GENCODE for isengard server
GENCODE := -gencode arch=compute_35,code=compute_35 

.phony: clean all release

all: main 

clean:
	rm -f main *.o

main: main.o png_proc.o vec.o
	$(CXX) $(OPT_FLAGS) -o $@ $^ $(GENCODE) -L$(INC_FLAGS) -lpng

main.o: main.cu png_proc.o vec.o
	$(CXX) -c $@ $^ $(GENCODE) 

png_proc.o: png_proc.cpp png_proc.h
	$(CXX) -I$(INC_FLAGS) -c $@ $<

vec.o: vec.cpp vec.h
	$(CXX) -c $@ $<

.phony: clean
