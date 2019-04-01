# ALMA Image Reconstruction MEM
# by Miguel Cárcamo

CUFFTFLAG = -lcufft
CFLAGS = -c -w -O3 -Xptxas -O3
INC_DIRS = -Iinclude -I/usr/local/include/casacore/
CFFLAG = -lcfitsio -lm -lcasa_casa -lcasa_tables -lcasa_ms -lcasa_measures
LDFLAGS = -lcuda -lcudart
FOPENFLAG = -Xcompiler -fopenmp -lgomp
CCFLAG = -lstdc++
# Gencode arguments
SMS ?= 30 35 37 50 52

ifeq ($(NEWCASA),1)
CFLAGS += -DNEWCASA
endif

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified <<<)
endif

ifeq ($(ARCHFLAG),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval ARCHFLAG += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
ARCHFLAG += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

main:	build/main.o build/MSFITSIO.o build/functions.o build/directioncosines.o build/rngs.o build/rvgs.o
	@ echo "Linking GPUVSIM"
	@ mkdir -p bin
	@ nvcc build/*.o -o bin/gpuvsim $(LDFLAGS) $(CFFLAG) $(FOPENFLAG) $(CUFFTFLAG) $(ARCHFLAG) $(CCFLAG)
	@ echo "The compilation has been completed successfully"

build/main.o: src/main.cu
	@ echo "Building Main"
	@ mkdir -p build
	@ nvcc $(CFLAGS) $(INC_DIRS) src/main.cu -o build/main.o $(LDFLAGS) $(CFFLAG) $(FOPENFLAG) $(CUFFTFLAG) $(ARCHFLAG)

build/MSFITSIO.o: src/MSFITSIO.cu
	@ echo "Building MSFITSIO"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/MSFITSIO.cu -o build/MSFITSIO.o $(LDFLAGS) $(CFFLAG) $(FOPENFLAG) $(CUFFTFLAG) $(ARCHFLAG)

build/functions.o: src/functions.cu
	@ echo "Building Functions"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/functions.cu -o build/functions.o $(LDFLAGS) $(CFFLAG) $(FOPENFLAG) $(CUFFTFLAG) $(ARCHFLAG)

build/directioncosines.o: src/directioncosines.cu
	@ echo "Building directioncosines"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/directioncosines.cu -o build/directioncosines.o  $(LDFLAGS) $(CFFLAG) $(ARCHFLAG)

build/rngs.o: src/rngs.cu
	@ echo "Building Random number generator"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/rngs.cu -o build/rngs.o $(LDFLAGS) $(CFFLAG) $(ARCHFLAG)

build/rvgs.o: src/rvgs.cu
	@ echo "Building Random number generator 2"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/rvgs.cu -o build/rvgs.o $(LDFLAGS) $(CFFLAG) $(ARCHFLAG)

clean:
	@ echo "Cleaning folders.."
	@ rm -rf build/*
	@ rm -rf bin/*
