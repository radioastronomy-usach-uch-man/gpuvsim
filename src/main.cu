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
#include "directioncosines.cuh"
#include <time.h>

long M, N, numVisibilities;
int iter=0;

cufftHandle plan1GPU;

cufftComplex *device_I, *device_V, *device_image;

float DELTAX, DELTAY, deltau, deltav, beam_noise, beam_bmaj;
float beam_bmin, b_noise_aux, random_probability = 1.0;
float noise_jypix, fg_scale, antenna_diameter, pb_factor, pb_cutoff;

dim3 threadsPerBlockNN;
dim3 numBlocksNN;

int threadsVectorReduceNN, blocksVectorReduceNN, crpix1, crpix2, verbose_flag = 0, apply_noise = 0, it_maximum, status_mod_in;
int selected, t_telescope, reg_term;
char *output;

double ra, dec;

freqData data;
fitsfile *mod_in;

Field *fields;

VariablesPerField *vars_per_field;

inline bool IsGPUCapableP2P(cudaDeviceProp *pProp)
{
  #ifdef _WIN32
      return (bool)(pProp->tccDriver ? true : false);
  #else
      return (bool)(pProp->major >= 2);
  #endif
}

inline bool IsAppBuiltAs64()
{
  #if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
      return 1;
  #else
      return 0;
  #endif
}

__host__ int main(int argc, char **argv) {
  clock_t t;
  double start, end;
	////CHECK FOR AVAILABLE GPUs
  printf("gpuvsim Copyright (C) 2016-2017  Miguel Carcamo, Pablo Roman, Simon Casassus, Victor Moral, Fernando Rannou - miguel.carcamo@usach.cl\n");
  printf("This program comes with ABSOLUTELY NO WARRANTY; for details use option -w\n");
  printf("This is free software, and you are welcome to redistribute it under certain conditions; use option -c for details.\n\n\n");

	if (!IsAppBuiltAs64()){
        printf("%s is only supported with on 64-bit OSs and the application must be built as a 64-bit target. Test is being waived.\n", argv[0]);
        exit(EXIT_SUCCESS);
  }


	float noise_min = 1E32;

	Vars variables = getOptions(argc, argv);
	char *msinput = variables.input;
	char *msoutput = variables.output;
  char *inputdat = variables.inputdat;
	char *modinput = variables.modin;
  selected = variables.select;
  int total_visibilities = 0;
  random_probability = variables.randoms;

  readInputDat(inputdat);
  init_beam(t_telescope);
  if(verbose_flag){
	   printf("Counting data for memory allocation\n");
  }

  canvasVariables canvas_vars = readCanvas(modinput, mod_in, b_noise_aux, status_mod_in, verbose_flag);

  M = canvas_vars.M;
  N = canvas_vars.N;
  DELTAX = canvas_vars.DELTAX;
  DELTAY = canvas_vars.DELTAY;
  ra = canvas_vars.ra;
  dec = canvas_vars.dec;
  crpix1 = canvas_vars.crpix1;
  crpix2 = canvas_vars.crpix2;
  beam_bmaj = canvas_vars.beam_bmaj;
  beam_bmin = canvas_vars.beam_bmin;
  beam_noise = canvas_vars.beam_noise;

  data = countVisibilities(msinput, fields);

  vars_per_field = (VariablesPerField*)malloc(data.nfields*sizeof(VariablesPerField));

  if(verbose_flag){
     printf("Number of fields = %d\n", data.nfields);
	   printf("Number of frequencies = %d\n", data.total_frequencies);
   }

  for(int f=0; f<data.nfields; f++){
  	fields[f].visibilities = (Vis*)malloc(data.total_frequencies*sizeof(Vis));
  	fields[f].device_visibilities = (Vis*)malloc(data.total_frequencies*sizeof(Vis));
  	vars_per_field[f].device_vars = (VPF*)malloc(data.total_frequencies*sizeof(VPF));
  }

  //ALLOCATE MEMORY AND GET TOTAL NUMBER OF VISIBILITIES
  for(int f=0; f<data.nfields; f++){
  	for(int i=0; i < data.total_frequencies; i++){
  		fields[f].visibilities[i].stokes = (int*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(int));
  		fields[f].visibilities[i].u = (float*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(float));
  		fields[f].visibilities[i].v = (float*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(float));
  		fields[f].visibilities[i].weight = (float*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(float));
  		fields[f].visibilities[i].Vo = (cufftComplex*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(cufftComplex));
      fields[f].visibilities[i].Vm = (cufftComplex*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(cufftComplex));
      total_visibilities += fields[f].numVisibilitiesPerFreq[i];
  	}
  }

  if(verbose_flag){
	   printf("Reading visibilities and FITS input files...\n");
  }


  readMS(msinput, fields, data);

  if(verbose_flag){
    printf("MS File Successfully Read\n");
    if(beam_noise == -1){
      printf("Beam noise wasn't provided by the user... Calculating...\n");
    }
  }

  //Declaring block size and number of blocks for visibilities
  for(int f=0; f<data.nfields; f++){
  	for(int i=0; i< data.total_frequencies; i++){
  		fields[f].visibilities[i].numVisibilities = fields[f].numVisibilitiesPerFreq[i];
  		long UVpow2 = NearestPowerOf2(fields[f].visibilities[i].numVisibilities);
      fields[f].visibilities[i].threadsPerBlockUV = variables.blockSizeV;
  		fields[f].visibilities[i].numBlocksUV = UVpow2/fields[f].visibilities[i].threadsPerBlockUV;
    }
  }

  cudaSetDevice(selected);
  for(int f=0; f<data.nfields; f++){
    for(int i=0; i<data.total_frequencies; i++){
  	   gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].u, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
  		 gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].v, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
  		 gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].Vo, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
  		 gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].weight, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
       gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].Vm, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
  		 gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].Vr, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
  	}
  }




  cudaSetDevice(selected);
  for(int f=0; f<data.nfields; f++){
    gpuErrchk(cudaMalloc((void**)&vars_per_field[f].atten_image, sizeof(float)*M*N));
    gpuErrchk(cudaMemset(vars_per_field[f].atten_image, 0, sizeof(float)*M*N));
  	for(int i=0; i < data.total_frequencies; i++){

  		gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].u, fields[f].visibilities[i].u, sizeof(float)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

  		gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].v, fields[f].visibilities[i].v, sizeof(float)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

  		gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].weight, fields[f].visibilities[i].weight, sizeof(float)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

  		gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].Vo, fields[f].visibilities[i].Vo, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

      gpuErrchk(cudaMemset(fields[f].device_visibilities[i].Vm, 0, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));

  	}
  }


  //Declaring block size and number of blocks for Image
  dim3 threads(variables.blockSizeX, variables.blockSizeY);
	dim3 blocks(M/threads.x, N/threads.y);
	threadsPerBlockNN = threads;
	numBlocksNN = blocks;

	noise_jypix = beam_noise / (PI * beam_bmaj * beam_bmin / (4 * log(2) ));

	float deltax = RPDEG*DELTAX; //radians
	float deltay = RPDEG*DELTAY; //radians
	deltau = 1.0 / (M * deltax);
	deltav = 1.0 / (N * deltay);



	float2 *host_I = (float2*)malloc(M*N*sizeof(float2));
  /////////////////////////////////////////////////////CALCULATE DIRECTION COSINES/////////////////////////////////////////////////
  double raimage = ra * RPDEG_D;
  double decimage = dec * RPDEG_D;
  if(verbose_flag){
    printf("FITS: Ra: %lf, dec: %lf\n", raimage, decimage);
  }
  for(int f=0; f<data.nfields; f++){
  	double lobs, mobs;

  	direccos(fields[f].obsra, fields[f].obsdec, raimage, decimage, &lobs,  &mobs);

  	if(crpix1 != crpix2){
		fields[f].global_xobs = (crpix1 - 1.0) - (lobs/deltax) + 1.0;
		fields[f].global_yobs = (crpix2 - 1.0) - (mobs/deltay) - 1.0;
	}else{
		fields[f].global_xobs = (crpix1 - 1.0) - (lobs/deltax) - 1.0;
		fields[f].global_yobs = (crpix2 - 1.0) - (mobs/deltay) - 1.0;
	}
    if(verbose_flag){
  	   printf("Field %d - Ra: %f, dec: %f , x0: %f, y0: %f\n", f, fields[f].obsra, fields[f].obsdec, fields[f].global_xobs, fields[f].global_yobs);
    }

    if(fields[f].global_xobs < 0 || fields[f].global_xobs > M || fields[f].global_xobs < 0 || fields[f].global_yobs > N) {
      printf("Pointing center (%f,%f) is outside the range of the image\n", fields[f].global_xobs, fields[f].global_xobs);
      goToError();
    }
  }
	////////////////////////////////////////////////////////MAKE STARTING IMAGE////////////////////////////////////////////////////////
	float *input_sim;
  float *input_sim_alpha;

  readFITSImageValues(modinput, mod_in, input_sim, status_mod_in, M, N);

  int x = M-1;
  int y = N-1;
	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++){
			host_I[N*i+j].x = input_sim[N*y+x];
			host_I[N*i+j].y = 0;
      x--;
		}
    x=M-1;
    y--;
	}

  free(input_sim);
  free(input_sim_alpha);
	////////////////////////////////////////////////CUDA MEMORY ALLOCATION FOR DEVICE///////////////////////////////////////////////////


  cudaSetDevice(selected);
	gpuErrchk(cudaMalloc((void**)&device_V, sizeof(cufftComplex)*M*N));
	gpuErrchk(cudaMalloc((void**)&device_image, sizeof(cufftComplex)*M*N));

  cudaSetDevice(selected);

	gpuErrchk(cudaMalloc((void**)&device_I, sizeof(cufftComplex)*M*N));
  gpuErrchk(cudaMemset(device_I, 0, sizeof(cufftComplex)*M*N));

  gpuErrchk(cudaMemcpy2D(device_I, sizeof(cufftComplex), host_I, sizeof(cufftComplex), sizeof(cufftComplex), M*N, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemset(device_V, 0, sizeof(cufftComplex)*M*N));
	gpuErrchk(cudaMemset(device_image, 0, sizeof(cufftComplex)*M*N));

  cudaSetDevice(selected);
	if ((cufftPlan2d(&plan1GPU, N, M, CUFFT_C2C))!= CUFFT_SUCCESS) {
		printf("cufft plan error\n");
		return -1;
	}

  //Time is taken from first kernel
  t = clock();
  start = omp_get_wtime();
  cudaSetDevice(selected);
  for(int f=0; f < data.nfields; f++){
  	for(int i=0; i<data.total_frequencies; i++){
  		hermitianSymmetry<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].u, fields[f].device_visibilities[i].v, fields[f].device_visibilities[i].Vo, fields[f].visibilities[i].freq, fields[f].numVisibilitiesPerFreq[i]);
  		gpuErrchk(cudaDeviceSynchronize());
  	}
  }

  uvsim(device_I);

	//Saving visibilities to disk
  residualsToHost(fields, data, 1, 0);
  printf("Saving residuals to MS...\n");


  if(apply_noise && random_probability < 1.0){
    writeMSSIMSubsampledMC(msinput, msoutput, fields, data, random_probability, verbose_flag);
  }else if(random_probability < 1.0){
    writeMSSIMSubsampled(msinput, msoutput, fields, data, random_probability, verbose_flag);
  }else if(apply_noise){
    writeMSSIMMC(msinput, msoutput, fields, data, verbose_flag);
  }else{
     writeMSSIM(msinput, msoutput, fields, data, verbose_flag);
  }

	printf("Visibilities saved.\n");

	//Free device and host memory
	printf("Free device and host memory\n");
	cufftDestroy(plan1GPU);
  for(int f=0; f<data.nfields; f++){
  	for(int i=0; i<data.total_frequencies; i++){
  		cudaFree(fields[f].device_visibilities[i].u);
  		cudaFree(fields[f].device_visibilities[i].v);
  		cudaFree(fields[f].device_visibilities[i].weight);

  		cudaFree(fields[f].device_visibilities[i].Vo);

  		cufftDestroy(vars_per_field[f].device_vars[i].plan);
  	}
  }

  for(int f=0; f<data.nfields; f++){
  	for(int i=0; i<data.total_frequencies; i++){
      if(fields[f].numVisibilitiesPerFreq[i] != 0){
    		free(fields[f].visibilities[i].u);
    		free(fields[f].visibilities[i].v);
    		free(fields[f].visibilities[i].weight);
    		free(fields[f].visibilities[i].Vo);
        free(fields[f].visibilities[i].Vm);
      }
  	}
  }

	cudaFree(device_I);

	cudaFree(device_V);
	cudaFree(device_image);

	free(host_I);
	free(msinput);
	free(msoutput);
	free(modinput);

  fits_close_file(mod_in, &status_mod_in);
  if (status_mod_in) {
    fits_report_error(stderr, status_mod_in);
    goToError();
  }

	return 0;
}
