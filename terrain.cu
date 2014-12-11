#include <stdio.h>
#include <float.h>
#include <climits>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuPrintf.cu"
#include "cuPrintf.cuh"
#include "sys/time.h"

//region ERROR CHECKING
#define CUDA_ERROR_CHECK
#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError()    __cudaCheckError(__FILE__, __LINE__)
 
inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
  if (cudaSuccess != err)
  {
    fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
    exit( -1 );
  }
#endif
 
  return;
}
 
inline void __cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err)
  {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  err = cudaDeviceSynchronize();
  if (cudaSuccess != err)
  {
      fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
      exit(-1);
  }
#endif
 
  return;
}
//endregion ERROR CHECKING


//region CONSTANTS
const dim3 maxBlockDim(16, 16);
const dim3 maxGridDim(65535, 65535);
//endregion CONSTANTS


//region RANDOM NUMBER GENERATOR
unsigned int h_m_w = 521288629;
unsigned int h_m_z = 362436069;

// magic, DON'T TOUCH!
float GetUniform() {
  h_m_w = 36969 * (h_m_z & 65535) + (h_m_z >> 16);
  h_m_z = 18000 * (h_m_w & 65535) + (h_m_w >> 16);
  return (float)((((h_m_z << 16) + h_m_w) + 1.0) * (float)2.328306435454494e-10);
}

__device__ void d_GetUniform(unsigned int *m_w, unsigned int *m_z, float* ret) {
	*m_z = 36969 * (*m_z & 65535) + (*m_z >> 16);
  *m_w = 18000 * (*m_w & 65535) + (*m_w >> 16);
  *ret = (float)((((*m_z << 16) + *m_w) + 1.0f) * (float)2.328306435454494e-10);
}

void RandomizeGeneratorSeed(unsigned int *m_w, unsigned int *m_z) {
	do {
		timeval t1, t2;
	  gettimeofday(&t1, NULL);
	  gettimeofday(&t2, NULL);
	  *m_w = (t1.tv_usec * t1.tv_sec) & UINT_MAX;
	  *m_z = (t2.tv_usec * t2.tv_sec) & INT_MAX;
	} while (*m_w == *m_z);
}
//endregion RANDOM NUMBER GENERATOR


//region UTILITY
// Flat[x + WIDTH * (y + DEPTH * z)] = Original[x, y, z]
// https://stackoverflow.com/questions/7367770/how-to-flatten-or-index-3d-array-in-1d-array
__device__ int d_transform3Dto1D(const int x,
															const int y,
															const int z,
															const int width,
															const int height,
															const int depth) {
	return x + width * y + width * height * z;
}

int h_transform3Dto1D(const int x,
											const int y,
											const int z,
											const int width,
											const int height,
											const int depth) {
	return x + width * y + width * height * z;
}
//endregion UTILITY


//region GENERATE RANDOM ARRAY
float* generate_random_array(int count) {
  float *randoms = (float*) malloc(count * sizeof(float));
  for (int i = 0; i < count; ++i) {
    randoms[i] = GetUniform();
    assert(randoms[i] <= 1);
  }
  return randoms;
}

void test_generate_random_array() {
	const int count = 10000000;

	float* h_arr = generate_random_array(count);

	for (int i = 0; i < count; i++) {
		assert(h_arr[i] >= 0);
		assert(h_arr[i] <= 1);
	}

	free(h_arr);
}
//endregion GENERATE RANDOM ARRAY


//region GENERATE PLANE
__global__ void generate_plane(float* d_arr, const int rows, const int cols, const float init_val) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= rows || col >= cols) return;
	d_arr[row * cols + col] = init_val;
}

void h_generate_plane(float* h_arr, const int rows, const int cols, const float init_val) {
	// calculate block and grid size
	dim3 blockDim = maxBlockDim;
	int blocksX = ceil((float)cols / blockDim.x);
	int blocksY = ceil((float)rows / blockDim.y);
	dim3 gridDim(blocksX, blocksY);
	printf("\nGenerate Plane Kernel\n");
	printf("Rows: %d Cols: %d InitVal: %f\n", rows, cols, init_val);
	printf("BlockDim(%d,%d) GridDim(%d,%d)\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);

	size_t plane_data_size = rows * cols * sizeof(float);

	// allocate
	float* d_arr; CudaSafeCall(cudaMalloc((void**)&d_arr, plane_data_size));

	// initialize
	CudaSafeCall(cudaMemset(d_arr, 0, plane_data_size));

	// invoke kernel and print output
	generate_plane<<<gridDim, blockDim>>>(d_arr, rows, cols, init_val);
	CudaCheckError();
	cudaPrintfDisplay(stdout, true);

	// copy data from device to host
	CudaSafeCall(cudaMemcpy((void*)h_arr, (void*)d_arr, plane_data_size, cudaMemcpyDeviceToHost));

	// free memory
	cudaFree(d_arr);
}

void test_generate_plane() {
	const int rows = 1000;
	const int cols = 1000;
	const float init_val = 5.5f;

	float* h_arr = (float*) malloc(rows * cols * sizeof(float));

	h_generate_plane(h_arr, rows, cols, init_val);

	for (int i = 0; i < rows * cols; i++) {
		assert(h_arr[i] == init_val);
	}

	free(h_arr);
}
//endregion GENERATE PLANE


//region SCALE MAP
__global__ void scale_map(float *d_arr, const float scalar, const int count) {
  int blockId = gridDim.x * blockIdx.y + blockIdx.x;
  int tid = (blockDim.x * threadIdx.y) + threadIdx.x; // id inside block
  int global_tid = blockId * (blockDim.x * blockDim.y) + tid; // global id

  if (global_tid >= count) return;

  d_arr[global_tid] = d_arr[global_tid] * scalar;
}

void h_scale(float* h_arr, const float scalar, const int count) {
	dim3 blockDim = maxBlockDim;
  dim3 gridDim(ceil(sqrt(count)), ceil(sqrt(count)));
  printf("\nNormalize Kernel\n");
  printf("Count: %d\n", count);
  printf("BlockDim(%d, %d) GridDim(%d, %d)\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);
  printf("Scaling by: %f\n", scalar);

  // allocate
  float *d_arr; CudaSafeCall(cudaMalloc((void**)&d_arr, count * sizeof(float)));

  // initialize
  CudaSafeCall(cudaMemcpy((void*)d_arr, (void*)h_arr, count * sizeof(float), cudaMemcpyHostToDevice));

  // invoke kernel and print output
  scale_map<<<gridDim, blockDim>>>(d_arr, scalar, count);
  cudaPrintfDisplay(stdout, true);
  cudaDeviceSynchronize();

  // copy data from device to host
  CudaSafeCall(cudaMemcpy((void*)h_arr, (void*)d_arr, count * sizeof(float), cudaMemcpyDeviceToHost));

  // free memory
  cudaFree(d_arr);
}

void test_scale() {
	const int rows = 1000;
	const int cols = 1000;
	const float init_val = 5.5f;
	const float scalar = 3.5f;

	float* h_arr = (float*) malloc(rows * cols * sizeof(float));

	h_generate_plane(h_arr, rows, cols, init_val);
	h_scale(h_arr, scalar, rows * cols);

	for(int i = 0; i < rows * cols; i++) {
		assert(h_arr[i] == init_val * scalar);
	}

	free(h_arr);
}
//endregion SCALE MAP


//region FIND MAX
__global__ void max_reduce(const float *d_arr, float *d_maxes,  const int count, const int responsible) {
	int blockId = gridDim.x * blockIdx.y + blockIdx.x;
  int tid = (blockDim.x * threadIdx.y) + threadIdx.x; // id inside block
  int global_tid = blockId * (blockDim.x * blockDim.y) + tid; // global id

  extern __shared__ float threadMaxes[];

  threadMaxes[tid] = -FLT_MAX; // initialize shared array
  __syncthreads();

  int startIdx = global_tid * responsible;
  int endIdx = startIdx + responsible - 1;
  if (endIdx >= count) endIdx = count - 1;
  //cuPrintf("StartIdx: %d EndIdx: %d\n", startIdx, endIdx);
  if (startIdx >= count) { return; }
  float localMax = d_arr[startIdx];
  for (int q = startIdx; q < endIdx; q++) {
  	if (d_arr[q] > localMax) localMax = d_arr[q];
  	//if (q == 3347) cuPrintf("FUUUUUUCCCKKK! %d %f %d %f\n", q, d_arr[q], 3347, d_arr[3347]); // prints out different values on Tesla machine for some reason
  }

  threadMaxes[tid] = localMax;
  __syncthreads();

  if (tid == 0) {
  	float blockMax = -FLT_MAX;
  	for (int i = 0; i < blockDim.x * blockDim.y; ++i) {
  		if (threadMaxes[i] > blockMax) {
  			blockMax = threadMaxes[i];
  		}
  	}

  	cuPrintf("Block(%d) Max: %f\n", blockId, blockMax);
  	d_maxes[blockId] = blockMax;
  }
}

float h_findMax(const float* h_arr, const int count, const int responsible) {
	// calculate block and grid size
	dim3 blockDim = maxBlockDim;
	int singleThreadDim = ceil(sqrt(count / responsible)); // threads in single dimension
	int singleGridDim = ceil((float)singleThreadDim / blockDim.x);
	dim3 gridDim(singleGridDim, singleGridDim);
	int totalThreads = singleThreadDim * singleThreadDim;
	int totalBlocks = singleGridDim * singleGridDim;
	printf("\nFindMax Kernel\n");
	printf("Count: %d Responsible: %d\n", count, responsible);
	printf("Total Threads: %d Total Blocks: %d\n", totalThreads, totalBlocks);
	printf("BlockDim(%d,%d) GridDim(%d,%d)\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);

	size_t arr_data_size = count * sizeof(float);
	size_t max_data_size = totalBlocks * sizeof(float);

	// allocate
	float *d_arr; CudaSafeCall(cudaMalloc((void**)&d_arr, arr_data_size));
	float *h_maxes = (float*) malloc(max_data_size);
	float *d_maxes; CudaSafeCall(cudaMalloc((void**)&d_maxes, max_data_size));

	// initialize
	for (int i = 0; i < totalBlocks; ++i) h_maxes[i] = 0;
	CudaSafeCall(cudaMemset(d_arr, 0, arr_data_size));
	CudaSafeCall(cudaMemset(d_maxes, 0, max_data_size));

	// copy data from host to device
	CudaSafeCall(cudaMemcpy((void*)d_arr, (void*)h_arr, arr_data_size, cudaMemcpyHostToDevice));

	// invoke kernel and print output
	cudaDeviceSynchronize();
	max_reduce<<<gridDim, blockDim, blockDim.x * blockDim.y * sizeof(float)>>>(d_arr, d_maxes, count, responsible);
	CudaCheckError();
	cudaPrintfDisplay(stdout, true);

	// copy data from device to host
	cudaMemcpy(h_maxes, d_maxes, max_data_size, cudaMemcpyDeviceToHost);

	// free memory
	cudaFree(d_arr);
	cudaFree(d_maxes);

	// get global max from block maxes
	float global_max = -FLT_MAX;
	for (int i = 0; i < totalBlocks; i++) {
		if (h_maxes[i] > global_max) {
			global_max = h_maxes[i];
		}
	}

	free(h_maxes);

	return global_max;
}

void test_find_max() {
	const int count = 10000000;
	const float expected_max = 1.05f;

	float* h_arr = generate_random_array(count);

	// it is possible that the double, when casted to a float, will equal 1.
	// So make sure that doesn't happen with a while loop
	int randomIndex = -1;
	do {
		randomIndex = floor(count * GetUniform());
	} while(randomIndex < 0 || randomIndex >= count);

	h_arr[randomIndex] = expected_max; // set random index to something greater than the rest of the values

	for (int responsible = 125; responsible > 0; responsible-=25) {
		float actual_max = h_findMax(h_arr, count, responsible);
		assert(actual_max == expected_max);
	}

	free(h_arr);
}
//endregion FIND MAX


//region GENERATE GRADIENT FOR PERLIN NOISE
__global__ void generate_grid_gradient(float* gradient, float* randoms, const int rows, const int cols) {
	int blockId = gridDim.x * blockIdx.y + blockIdx.x;
  int tid = (blockDim.x * threadIdx.y) + threadIdx.x; // id inside block
  int global_tid = blockId * (blockDim.x * blockDim.y) + tid; // global id
  int row = (int)((float)global_tid / rows);
  int col = global_tid % rows;

  if (global_tid >= rows * cols) return;

  // random point on unit circle
  float xValue = randoms[global_tid];
  float yValue = sqrt(1-xValue*xValue);

	int indexX = d_transform3Dto1D(col, row, 0, cols, rows, 2);
	int indexY = d_transform3Dto1D(col, row, 1, cols, rows, 2);

	cuPrintf("1D(%d,%d) = 3D(%d,%d) = (%f,%f)\n", indexX, indexY, col, row, xValue, yValue);

  gradient[indexX] = xValue;
  gradient[indexY] = yValue;
}

float* h_generate_grid_gradient(const int rows, const int cols) {
	// calculate block and grid size
	dim3 blockDim = maxBlockDim;
	int blocksX = ceil((float)cols / blockDim.x);
	int blocksY = ceil((float)rows / blockDim.y);
	dim3 gridDim(blocksX, blocksY);
	printf("\nGenerate Grid Gradient Kernel\n");
	printf("Rows: %d Cols: %d GridSize: %d\n", rows, cols, rows * cols * 2);
	printf("BlockDim(%d,%d) GridDim(%d,%d)\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);

	size_t grid_data_size = rows * cols * 2 * sizeof(float);
	size_t random_data_size = rows * cols * sizeof(float);

	float* h_arr = (float*)malloc(grid_data_size);
	float* h_randoms = generate_random_array(rows * cols);

	// allocate
	float* d_arr; CudaSafeCall(cudaMalloc((void**)&d_arr, grid_data_size));
	float* d_randoms; CudaSafeCall(cudaMalloc((void**)&d_randoms, random_data_size));

	// initialize
	CudaSafeCall(cudaMemset(d_arr, 0, grid_data_size));
	CudaSafeCall(cudaMemcpy((void*)d_randoms, (void*)h_randoms, random_data_size, cudaMemcpyHostToDevice));

	// invoke kernel and print output
	generate_grid_gradient<<<gridDim, blockDim>>>(d_arr, d_randoms, rows, cols);
	CudaCheckError();
	cudaPrintfDisplay(stdout, true);

	// copy data from device to host
	CudaSafeCall(cudaMemcpy((void*)h_arr, (void*)d_arr, grid_data_size, cudaMemcpyDeviceToHost));

	// free memory
	cudaFree(d_arr);
	cudaFree(d_randoms);
	//free(h_randoms); // don't free or it seg faults
	
	return h_arr;
}

void test_generate_grid_gradient() {
	const int rows = 5;
	const int cols = 5;
	const float epsilon = 0.001f;

	float* h_arr = h_generate_grid_gradient(rows, cols);

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			int ix = h_transform3Dto1D(c, r, 0, cols, rows, 2);
			int iy = h_transform3Dto1D(c, r, 1, cols, rows, 2);
			float xValue = h_arr[ix];
			float yValue = h_arr[iy];
			assert((1 - fabs(xValue * xValue + yValue * yValue)) <= epsilon);
		}
	}

	//free(h_arr); // don't free or it seg faults
}
//endregion GENERATE GRADIENT


//region WRITE FILE
void h_output_file(const char *filename, int* colors, int rows, int cols) {
	FILE *fp = fopen(filename, "wb"); // binary mode
	(void) fprintf(fp, "P6\n%d %d\n255\n", cols, rows);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int rgb = colors[i * cols + j];
			unsigned char color[3];
			color[0] = (rgb >> 16) & 0xFF; /* red */
			color[1] = (rgb >> 8) & 0xFF; /* green */
			color[2] = rgb & 0xFF; /* blue */
			(void) fwrite(color, 1, 3, fp);
		}
	}
	(void) fclose(fp);
}

void test_output_file() {
	const int rows = 800, cols = 400;
	int *colors = new int[rows * cols];
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int rgb = ((255&0x0ff)<<16)|((128&0x0ff)<<8)|(55&0x0ff); // #ff8037
			colors[i * cols + j] = rgb;
		}
	}
	
	h_output_file("output.ppm", colors, rows, cols);
}
//endregion WRITE FILE


//region SMOOTH NOISE
//http://lodev.org/cgtutor/randomnoise.html
__device__ float smoothNoise(float *noise, float x, float y, const int rows, const int cols)
{
   //get fractional part of x and y
   float fractX = x - int(x);
   float fractY = y - int(y);
   
   //wrap around
   int x1 = (int(x) + cols) % cols;
   int y1 = (int(y) + rows) % rows;
   
   //neighbor values
   int x2 = (x1 + cols - 1) % cols;
   int y2 = (y1 + rows - 1) % rows;

   //smooth the noise with bilinear interpolation
   float value = 0.0;
   value += fractX * fractY * noise[x1 + y1 * cols];
   value += fractX * (1 - fractY) * noise[x1 + y2 * cols];
   value += (1 - fractX) * fractY * noise[x2 + y1 * cols];
   value += (1 - fractX) * (1 - fractY) * noise[x2 + y2 * cols];

   return value;
}

__global__ void add_turbulence(float* input_noise, float* output_noise, const int size, const int rows, const int cols) {
	int blockId = gridDim.x * blockIdx.y + blockIdx.x;
  int tid = (blockDim.x * threadIdx.y) + threadIdx.x; // id inside block
  int global_tid = blockId * (blockDim.x * blockDim.y) + tid; // global id
  int row = (int)((float)global_tid / rows);
  int col = global_tid % rows;

  if (global_tid >= rows * cols) return;

  float f = smoothNoise(input_noise, (float)col / size, (float)row / size, rows, cols) * size;
  output_noise[row * cols + col] += f;
}

float* h_generate_smooth_noise(const int rows, const int cols) {
	dim3 blockDim = maxBlockDim;
	int blocksX = ceil((float)cols / blockDim.x);
	int blocksY = ceil((float)rows / blockDim.y);
	dim3 gridDim(blocksX, blocksY);
	printf("\nGenerate Smooth Noise Kernel\n");
	printf("Rows: %d Cols: %d\n", rows, cols);
	printf("BlockDim(%d,%d) GridDim(%d,%d)\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);

	size_t noise_data_size = rows * cols * sizeof(float);

	float* h_noise = generate_random_array(rows * cols);
	float* d_input_noise; CudaSafeCall(cudaMalloc((void**)&d_input_noise, noise_data_size));
	float* d_output_noise; CudaSafeCall(cudaMalloc((void**)&d_output_noise, noise_data_size));

	CudaSafeCall(cudaMemcpy((void*)d_input_noise, (void*)h_noise, noise_data_size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemset(d_output_noise, 0, noise_data_size));

	int initial_size = sqrt(rows * cols) / 4; // can be anything really
	int size = initial_size;
	do { // layer resized noise maps on top of each other
		add_turbulence<<<gridDim, blockDim>>>(d_input_noise, d_output_noise, size, rows, cols);
		CudaCheckError();
		cudaPrintfDisplay(stdout, true);

		size >>= 1;
	} while(size >= 1);

	scale_map<<<gridDim, blockDim>>>(d_output_noise, 128.0f / initial_size, rows * cols);
	CudaCheckError();

	CudaSafeCall(cudaMemcpy((void*)h_noise, (void*)d_output_noise, noise_data_size, cudaMemcpyDeviceToHost));

	cudaFree(d_input_noise);
	cudaFree(d_output_noise);

	return h_noise;
}

void test_generate_smooth_noise() {
	const int rows = 1000;
	const int cols = 1000;

	float* h_noise = h_generate_smooth_noise(rows, cols);
	int* colors = (int*)malloc(rows * cols * sizeof(int));

	for (int x = 0; x < cols; x++)
	for (int y = 0; y < rows; y++) {
		int r = h_noise[y * cols + x];
		int g = r;
		int b = r;
		int rgb = ((r&0x0ff)<<16)|((g&0x0ff)<<8)|(b&0x0ff);
		colors[y * cols + x] = rgb;
	}

	h_output_file("output.ppm", colors, rows, cols);
}
//endregion SMOOTH NOISE


//region COLORIZE TERRAIN
// terrain is assumed to be normalized and is colorized in grayscale
int* h_colorize_terrain(const float* h_terrain, const float water_plane, const int rows, const int cols) {
	int* colors = (int*)malloc(rows * cols * sizeof(int));
	float scaled_water_plane = water_plane * 255;

	for (int x = 0; x < cols; x++)
	for (int y = 0; y < rows; y++) {
		float scaled_value = h_terrain[y * cols + x] * 255;
		int r = 0;
		int g = scaled_value;
		int b = 0;
		if (scaled_value < scaled_water_plane) {
			r = 0;
			g = 0;
			b = scaled_value;
		}
		int rgb = ((r&0x0ff)<<16)|((g&0x0ff)<<8)|(b&0x0ff);
		colors[y * cols + x] = rgb;
	}

	return colors;
}
//endregion COLORIZE TERRAIN

//region TERRAIN TESTING
void test_generate_terrain() {
	const int rows = 1000;
	const int cols = 1000;

	// STEP 1: Define plane (not explicitly needed)
	// but the code is there if it was needed

	// STEP 2: Generate Smooth Noise
	float* h_terrain = h_generate_smooth_noise(rows, cols);

	// STEP 3: Generate unique terrain structures (not done yet)
	// Not sure how to do this yet
	
	// STEP 4: Add low frequency noise (not needed until step 4 is done)
	// I will use the existing noise algorithm with very few layers (more course noise grid)
	// and layer that on top of the existing terrain to create small bumps
	
	// STEP 5: Find max height of terrain
	float max = h_findMax(h_terrain, rows * cols, 125);
	
	// STEP 6: Normalize terrain
	h_scale(h_terrain, 1.0f / max, rows * cols);

	// STEP 7: Define water plane
	float water_plane = 1.0f;
	do {
	  water_plane = GetUniform();
	} while (water_plane >= 0.65f || water_plane <= 0.35f); // position water plane reasonably
	printf("Water Plane: %f\n", water_plane);

	// STEP 8: Color terrain (working, but not fully implemented)
	int* colors = h_colorize_terrain(h_terrain, water_plane, rows, cols);

	// STEP 9: Save colorized terrain
	h_output_file("output.ppm", colors, rows, cols);
}
//endregion TERRAIN TESTING

int main()
{
	printf("Random Generator Seed: m_w(%u) m_z(%u)\n", h_m_w, h_m_z);

	cudaPrintfInit();

	// test_generate_plane();
	// test_generate_random_array();
	// test_scale();
	// test_find_max();
	// test_generate_grid_gradient();
	// test_generate_smooth_noise();
	 test_generate_terrain();

	cudaPrintfEnd();

	return EXIT_SUCCESS;
}