CXX := nvcc
OPT_FLAGS := -O3
INC_FLAGS := /usr/lib/
# GENCODE for holmes gpu server
#GENCODE := --expt-relaxed-constexpr -gencode arch=compute_70,code=compute_70 -gencode arch=compute_75,code=compute_75
# GENCODE for isengard server
GENCODE := --expt-relaxed-constexpr -gencode arch=compute_35,code=compute_35 

.phony: clean all release

all: main 

clean:
	rm -f main *.o

main: main.o march_kernel.o march_host.o png_proc.o vec.o 
	$(CXX) $(OPT_FLAGS) -o $@ $^ $(GENCODE) -L$(INC_FLAGS) -lpng

main.o: main.cu march_kernel.o march_host.o png_proc.o vec.o
	$(CXX) $(OPT_FLAGS) -c $@ $^ $(GENCODE) 

march_kernel.o: march_kernel.cu march_kernel.h util.h
	$(CXX) $(OPT_FLAGS) -I$(INC_FLAGS) -c $@ $< $(GENCODE) 

march_host.o: march_host.cpp march_host.h util.h
	$(CXX) $(OPT_FLAGS) -I$(INC_FLAGS) -c $@ $<

png_proc.o: png_proc.cpp png_proc.h
	$(CXX) $(OPT_FLAGS) -I$(INC_FLAGS) -c $@ $<

vec.o: vec.cpp vec.h
	$(CXX) $(OPT_FLAGS) -c $@ $<

.phony: clean
