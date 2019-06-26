#include <math.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include <float.h>
#include <unistd.h>
#include <getopt.h>
#include <fcntl.h>
#include <omp.h>
#include <string.h>
#include <sys/stat.h>
#include "MSFITSIO.cuh"


#define FLOAT_IMG   -32
#define DOUBLE_IMG  -64

#define TSTRING      16
#define TLONG        41
#define TINT         31
#define TFLOAT       42
#define TDOUBLE      82
#define TCOMPLEX     83
#define TDBLCOMPLEX 163

const float PI = CUDART_PI_F;
const double PI_D = CUDART_PI;
const float RPDEG = (PI/180.0);
const double RPDEG_D = (PI_D/180.0);
const float RPARCM = (PI/(180.0*60.0));
const float LIGHTSPEED = 2.99792458E8;
const float RZ = 1.2196698912665045;

typedef struct variablesPerFreq{
  cufftHandle plan;
  cufftComplex *device_image;
  cufftComplex *device_V;
}VPF;

typedef struct variablesPerField{
  float *atten_image;
  VPF *device_vars;
}VariablesPerField;

typedef struct variables {
	char *input;
  char *output;
  char *inputdat;
  char *modin;
  char *alpha;
  int select;
  int blockSizeX;
  int blockSizeY;
  int blockSizeV;
  float randoms;
  float nu_0;
} Vars;

__host__ void goToError();
__host__ long NearestPowerOf2(long x);

__host__ void readInputDat(char *file);
__host__ void init_beam(int telescope);
__host__ void print_help();
__host__ char *strip(const char *string, const char *chars);
__host__ Vars getOptions(int argc, char **argv);
__host__ void uvsim(cufftComplex *I);

__global__ void hermitianSymmetry(double3 *UVW, cufftComplex *Vo, float freq, int numVisibilities);
__global__ void apply_beam(float antenna_diameter, float pb_factor, float pb_cutoff, cufftComplex *image, long N, float xobs, float yobs, float freq, double DELTAX, double DELTAY);
__global__ void phase_rotate(cufftComplex *data, long M, long N, float xphs, float yphs);
__global__ void vis_mod(cufftComplex *Vm, cufftComplex *Vo, cufftComplex *V, double3 *Vx, double deltau, double deltav, long numVisibilities, long N);
