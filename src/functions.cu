/* -------------------------------------------------------------------------
  Copyright (C) 2016-2017  Miguel Carcamo, Pablo Roman, Simon Casassus,
  Victor Moral, Fernando Rannou - miguel.carcamo@usach.cl

  This program includes Numerical Recipes (NR) based routines whose
  copyright is held by the NR authors. If NR routines are included,
  you are required to comply with the licensing set forth there.

	Part of the program also relies on an an ANSI C library for multi-stream
	random number generation from the related Prentice-Hall textbook
	Discrete-Event Simulation: A First Course by Steve Park and Larry Leemis,
  for more information please contact leemis@math.wm.edu

  For the original parts of this code, the following license applies:

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
* -------------------------------------------------------------------------
*/
#include "functions.cuh"


extern long M, N;
extern int numVisibilities, iterations, iterthreadsVectorNN, blocksVectorNN, crpix1, crpix2, \
status_mod_in, verbose_flag, t_telescope, multigpu, firstgpu, reg_term, apply_noise;

extern cufftHandle plan1GPU;
extern cufftComplex *device_I, *device_V, *device_image;

extern float *device_noise_image;
extern float noise_jypix, fg_scale, DELTAX, DELTAY, deltau, deltav, random_probability;

extern dim3 threadsPerBlockNN, numBlocksNN;

extern float beam_noise, beam_bmaj, beam_bmin, b_noise_aux, beam_fwhm, beam_freq, beam_cutoff;
extern double ra, dec;

extern freqData data;

extern char* mempath, *out_image;

extern fitsfile *mod_in;

extern Field *fields;

extern VariablesPerField *vars_per_field;

__host__ void goToError()
{
  printf("An error has ocurred, exiting\n");
  exit(0);

}

__host__ void init_beam(int telescope)
{
  switch(telescope) {
  case 1:
    beam_fwhm = 33.0*RPARCM;   /* radians CBI2 */
    beam_freq = 30.0;          /* GHz */
    beam_cutoff = 90.0*RPARCM; /* radians */
    break;
  case 2:
    beam_fwhm = (8.4220/60)*RPARCM;   /* radians ALMA */
    beam_freq = 691.4;          /* GHz */
    beam_cutoff = 1.0*RPARCM; /* radians */
    break;
  case 3: //test
    beam_fwhm = 5*RPARCM;   /* radians CBI2 */
    beam_freq = 1000;          /* GHz */
    beam_cutoff = 10*RPARCM; /* radians */
    break;
  case 4:
    beam_fwhm = (9.0/60)*RPARCM*12/22;   /* radians ATCA */
    beam_freq = 691.4;          /* GHz */
    beam_cutoff = 1.0*RPARCM; /* radians */
    break;
  case 5:
    beam_fwhm = (9.0/60)*RPARCM*12/25;   /* radians VLA */
    beam_freq = 691.4;          /* GHz */
    beam_cutoff = 1.0*RPARCM; /* radians */
    break;
  case 6:
    beam_fwhm = 10.5*RPARCM;   /* radians SZA */
    beam_freq = 30.9380;          /* GHz */
    beam_cutoff = 20.0*RPARCM; /* radians */
    break;

  default:
    printf("Telescope type not defined\n");
    goToError();
    break;
  }
}


__host__ long NearestPowerOf2(long x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}


bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}


__host__ void readInputDat(char *file)
{
  FILE *fp;
  char item[50];
  float status;
  if((fp = fopen(file, "r")) == NULL){
    printf("ERROR. The input file wasn't provided by the user.\n");
    goToError();
  }else{
    while(true){
      int ret = fscanf(fp, "%s %e", item, &status);

      if(ret==EOF){
        break;
      }else{
        if (strcmp(item,"t_telescope")==0){
          t_telescope = status;
        }else if(strcmp(item,"random_probability")==0){
          if(random_probability == -1){
            random_probability = status;
          }
        }else{
          printf("Keyword not defined in input\n");
          goToError();
        }
      }
    }
  }
}



__host__ void print_help() {
	printf("Example: ./bin/gpuvmem options [ arguments ...]\n");
	printf("    -h  --help             Shows this\n");
  printf(	"   -X  --blockSizeX       Block X Size for Image (Needs to be pow of 2)\n");
  printf(	"   -Y  --blockSizeY       Block Y Size for Image (Needs to be pow of 2)\n");
  printf(	"   -V  --blockSizeV       Block Size for Visibilities (Needs to be pow of 2)\n");
  printf(	"   -i  --input            The name of the input file of visibilities(MS)\n");
  printf(	"   -o  --output           The name of the output file of residual visibilities(MS)\n");
  printf("    -I  --inputdat         The name of the input file of parameters\n");
  printf("    -m  --modin            mod_in_0 FITS file\n");
  printf("    -r  --randoms          Percentage of data used when random sampling (Default = 1.0, optional)\n");
  printf("    -s  --select           If multigpu option is OFF, then select the GPU ID of the GPU you will work on. (Default = 0)\n");
  printf("    -c  --copyright        Shows copyright conditions\n");
  printf("    -w  --warranty         Shows no warranty details\n");
  printf("        --apply-noise      Apply random gaussian noise to visibilities\n");
  printf("        --verbose          Shows information through all the execution\n");
}

__host__ char *strip(const char *string, const char *chars)
{
  char * newstr = (char*)malloc(strlen(string) + 1);
  int counter = 0;

  for ( ; *string; string++) {
    if (!strchr(chars, *string)) {
      newstr[ counter ] = *string;
      ++ counter;
    }
  }

  newstr[counter] = 0;
  return newstr;
}

__host__ Vars getOptions(int argc, char **argv) {
	Vars variables;

  variables.select = 0;
  variables.blockSizeX = -1;
  variables.blockSizeY = -1;
  variables.blockSizeV = -1;
  variables.randoms = 1.0;

	long next_op;
	const char* const short_op = "hi:o:O:I:m:s:X:Y:V:r:";

	const struct option long_op[] = { //Flag for help, copyright and warranty
                                    {"help", 0, NULL, 'h' },
                                    /* These options set a flag. */
                                    {"verbose", 0, &verbose_flag, 1},
                                    {"apply-noise", 0, &apply_noise, 1},
                                    /* These options don’t set a flag. */
                                    {"input", 1, NULL, 'i' }, {"output", 1, NULL, 'o'}, {"inputdat", 1, NULL, 'I'},
                                    {"modin", 1, NULL, 'm' }, {"select", 1, NULL, 's'}, {"blockSizeX", 1, NULL, 'X'},
                                    {"blockSizeY", 1, NULL, 'Y'}, {"blockSizeV", 1, NULL, 'V'}, {"random", 0, NULL, 'r'},
                                    { NULL, 0, NULL, 0 }};

	if (argc == 1) {
		printf(
				"ERROR. THE PROGRAM HAS BEEN EXECUTED WITHOUT THE NEEDED PARAMETERS OR OPTIONS\n");
		print_help();
		exit(EXIT_SUCCESS);
	}
  int option_index = 0;
	while (1) {
		next_op = getopt_long(argc, argv, short_op, long_op, &option_index);
		if (next_op == -1) {
			break;
		}

		switch (next_op) {
    case 0:
      /* If this option set a flag, do nothing else now. */
      if (long_op[option_index].flag != 0)
        break;
        printf ("option %s", long_op[option_index].name);
      if (optarg)
        printf (" with arg %s", optarg);
        printf ("\n");
        break;
		case 'h':
			print_help();
			exit(EXIT_SUCCESS);
		case 'i':
      variables.input = (char*) malloc((strlen(optarg)+1)*sizeof(char));
			strcpy(variables.input, optarg);
			break;
    case 'o':
      variables.output = (char*) malloc((strlen(optarg)+1)*sizeof(char));
  		strcpy(variables.output, optarg);
  		break;
    case 'I':
      variables.inputdat = (char*) malloc((strlen(optarg)+1)*sizeof(char));
      strcpy(variables.inputdat, optarg);
      break;
    case 'm':
      variables.modin = (char*) malloc((strlen(optarg)+1)*sizeof(char));
    	strcpy(variables.modin, optarg);
    	break;
    case 's':
      variables.select = atoi(optarg);
      break;
    case 'r':
      variables.randoms = atof(optarg);
      break;
    case 'X':
      variables.blockSizeX = atoi(optarg);
      break;
    case 'Y':
      variables.blockSizeY = atoi(optarg);
      break;
    case 'V':
      variables.blockSizeV = atoi(optarg);
      break;
		case '?':
			print_help();
			exit(EXIT_FAILURE);
		case -1:
			break;
		default:
      print_help();
			exit(EXIT_FAILURE);
		}
	}

  if(variables.blockSizeX == -1 && variables.blockSizeY == -1 && variables.blockSizeV == -1 ||
     strcmp(strip(variables.input, " "),"") == 0 && strcmp(strip(variables.output, " "),"") == 0 && strcmp(strip(variables.inputdat, " "),"") == 0 ||
     strcmp(strip(variables.modin, " "),"") == 0 ) {
        print_help();
        exit(EXIT_FAILURE);
  }

  if(variables.randoms < 0.0 || variables.randoms > 1.0){
    print_help();
    exit(EXIT_FAILURE);
  }

  if(!isPow2(variables.blockSizeX) && !isPow2(variables.blockSizeY) && !isPow2(variables.blockSizeV)){
    print_help();
    exit(EXIT_FAILURE);
  }

	return variables;
}

__global__ void hermitianSymmetry(float *Ux, float *Vx, cufftComplex *Vo, float freq, int numVisibilities)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < numVisibilities){
      if(Ux[i] < 0.0){
        Ux[i] *= -1.0;
        Vx[i] *= -1.0;
        Vo[i].y *= -1.0;
      }
      Ux[i] = (Ux[i] * freq) / LIGHTSPEED;
      Vx[i] = (Vx[i] * freq) / LIGHTSPEED;
  }
}



__global__ void apply_beam(float beam_fwhm, float beam_freq, float beam_cutoff, cufftComplex *image, cufftComplex *fg_image, long N, float xobs, float yobs, float freq, float DELTAX, float DELTAY)
{
    int j = threadIdx.x + blockDim.x * blockIdx.x;
    int i = threadIdx.y + blockDim.y * blockIdx.y;


    float dx = DELTAX * 60.0;
    float dy = DELTAY * 60.0;
    float x = (j - xobs) * dx;
    float y = (i - yobs) * dy;
    float arc = RPARCM*sqrtf(x*x+y*y);
    float c = 4.0*logf(2.0);
    float a = (beam_fwhm*beam_freq/(freq*1e-9));
    float r = arc/a;
    float atten = expf(-c*r*r);

    if(arc <= beam_cutoff){
      image[N*i+j].x = fg_image[N*i+j].x * atten;
      image[N*i+j].y = 0.0;
    }else{
      image[N*i+j].x = 0.0;
      image[N*i+j].y = 0.0;
    }


}


/*--------------------------------------------------------------------
 * Phase rotate the visibility data in "image" to refer phase to point
 * (x,y) instead of (0,0).
 * Multiply pixel V(i,j) by exp(-2 pi i (x/ni + y/nj))
 *--------------------------------------------------------------------*/
__global__ void phase_rotate(cufftComplex *data, long M, long N, float xphs, float yphs)
{

		int j = threadIdx.x + blockDim.x * blockIdx.x;
		int i = threadIdx.y + blockDim.y * blockIdx.y;

    float u,v, phase, c, s, re, im;
    float du = xphs/(float)M;
    float dv = yphs/(float)N;

    if(j < M/2){
      u = du * j;
    }else{
      u = du * (j-M);
    }

    if(i < N/2){
      v = dv * i;
    }else{
      v = dv * (i-N);
    }

    phase = 2.0*(u+v);
    #if (__CUDA_ARCH__ >= 300 )
      sincospif(phase, &s, &c);
    #else
      c = cospif(phase);
      s = sinpif(phase);
    #endif
    re = data[N*i+j].x;
    im = data[N*i+j].y;
    data[N*i+j].x = re * c - im * s;
    data[N*i+j].y = re * s + im * c;
}


/*
 * Interpolate in the visibility array to find the visibility at (u,v);
 */
__global__ void vis_mod(cufftComplex *Vm, cufftComplex *Vo, cufftComplex *V, float *Ux, float *Vx, float deltau, float deltav, long numVisibilities, long N)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  long i1, i2, j1, j2;
  float du, dv, u, v;
  float v11, v12, v21, v22;
  float Zreal;
  float Zimag;

  if (i < numVisibilities){

    u = Ux[i]/deltau;
    v = Vx[i]/deltav;

    if (fabsf(u) > (N/2)+0.5 || fabsf(v) > (N/2)+0.5) {
      printf("Error in residual: u,v = %f,%f\n", u, v);
      asm("trap;");
    }

    if(u < 0.0){
      u = N + u;
    }

    if(v < 0.0){
      v = N + v;
    }

    i1 = u;
    i2 = (i1+1)%N;
    du = u - i1;
    j1 = v;
    j2 = (j1+1)%N;
    dv = v - j1;

    if (i1 < 0 || i1 > N || j1 < 0 || j2 > N) {
      printf("Error in residual: u,v = %f,%f, %ld,%ld, %ld,%ld\n", u, v, i1, i2, j1, j2);
      asm("trap;");
    }

    /* Bilinear interpolation: real part */
    v11 = V[N*j1 + i1].x; /* [i1, j1] */
    v12 = V[N*j2 + i1].x; /* [i1, j2] */
    v21 = V[N*j1 + i2].x; /* [i2, j1] */
    v22 = V[N*j2 + i2].x; /* [i2, j2] */
    Zreal = (1-du)*(1-dv)*v11 + (1-du)*dv*v12 + du*(1-dv)*v21 + du*dv*v22;
    /* Bilinear interpolation: imaginary part */
    v11 = V[N*j1 + i1].y; /* [i1, j1] */
    v12 = V[N*j2 + i1].y; /* [i1, j2] */
    v21 = V[N*j1 + i2].y; /* [i2, j1] */
    v22 = V[N*j2 + i2].y; /* [i2, j2] */
    Zimag = (1-du)*(1-dv)*v11 + (1-du)*dv*v12 + du*(1-dv)*v21 + du*dv*v22;

    Vm[i].x = Zreal;
    Vm[i].y = Zimag;

  }

}

__host__ void uvsim(cufftComplex *I)
{

    for(int f=0; f<data.nfields;f++){
      for(int i=0; i<data.total_frequencies;i++){
        if(fields[f].numVisibilitiesPerFreq[i] != 0){

        	apply_beam<<<numBlocksNN, threadsPerBlockNN>>>(beam_fwhm, beam_freq, beam_cutoff, device_image, device_I, N, fields[f].global_xobs, fields[f].global_yobs, fields[f].visibilities[i].freq, DELTAX, DELTAY);
        	gpuErrchk(cudaDeviceSynchronize());

        	//FFT 2D
        	if ((cufftExecC2C(plan1GPU, (cufftComplex*)device_image, (cufftComplex*)device_V, CUFFT_FORWARD)) != CUFFT_SUCCESS) {
        		printf("CUFFT exec error\n");
        		goToError();
        	}
        	gpuErrchk(cudaDeviceSynchronize());

          //PHASE_ROTATE
          phase_rotate<<<numBlocksNN, threadsPerBlockNN>>>(device_V, M, N, fields[f].global_xobs, fields[f].global_yobs);
        	gpuErrchk(cudaDeviceSynchronize());

          //RESIDUAL CALCULATION
          vis_mod<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].Vm, fields[f].device_visibilities[i].Vo, device_V, fields[f].device_visibilities[i].u, fields[f].device_visibilities[i].v, deltau, deltav, fields[f].numVisibilitiesPerFreq[i], N);
        	gpuErrchk(cudaDeviceSynchronize());

        }
      }
    }
}
