#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "..\common\cpu_bitmap.h"
#include "..\common\book.h"
#include "..\Math\Vec3f.h"

#include <stdio.h>
#include <time.h>
#include <math.h>

#define DIM 1024
#define TriDIM 128
#define RayDIM 32
#define ToothNum 14

#define PI 3.1415926535897932f

struct cuComplex
{
	float r;
	float i;

	__device__ cuComplex(float a, float b) : r(a), i(b) {}

	__device__ float magnitude2(){ return r * r + i * i; }

	__device__ cuComplex operator*(const cuComplex& a)
	{
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}

	__device__ cuComplex operator+(const cuComplex& a)
	{
		return cuComplex(r+a.r, i+a.i);
	}
};

__device__ bool IntersectLineAndTriangle(Vec3f& R, const Vec3f& RayDir, const Vec3f& RayOrign,
	                                               const Vec3f& A, const Vec3f& B, const Vec3f& C)
{
	R = Vec3f(0.f, 0.f, 0.f);

	Vec3f e1 = B - A;
	Vec3f e2 = C - A;

	Vec3f p = RayDir^e2;
	float tmp = p*e1;
	if (fabs(tmp) < 0.00000001)
		return false;

	tmp = 1.f / tmp;
	Vec3f s = RayOrign - A;
	float u = tmp*(s*p);
	if (u < 0.0 || u > 1.0)
		return false;

	Vec3f q = s^e1;
	float v = tmp * (RayDir*q);
	if (v < 0.0 || v > 1.0)
		return false;

	float w = 1.f - u - v;
	if (w < 0.f)
		return false;

	R = A*w + B*u + C*v;
	return true;
}

__device__ int julia(int x, int y)
{
	const float scale = 1.5;
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);

	int i = 0;
	for (i = 0; i < 200; ++i)
	{
		a = a*a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}

	return 1;
}

__global__ void kernel(unsigned char* ptr)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = y*gridDim.x*blockDim.x + x;

	int juliaValue = julia(x, y);
	ptr[offset * 4 + 0] = 255 * juliaValue;
	ptr[offset * 4 + 1] = 255 * juliaValue;
	ptr[offset * 4 + 2] = 255 * juliaValue;
	ptr[offset * 4 + 3] = 255;
}

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void IntersectKernel(float* R0, float* R1, float* R2,
	                            float RayDir0, float RayDir1, float RayDir2,
								float RayOrign0, float RayOrign1, float RayOrign2,
	                            const float *A0, const float *A1, const float *A2,
	                            const float *B0, const float *B1, const float *B2, 
	                            const float *C0, const float *C1, const float *C2)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = y*gridDim.x*blockDim.x + x;
	if (offset < TriDIM*TriDIM)
	{
		Vec3f R;
		Vec3f RayDir(RayDir0, RayDir1, RayDir2);
		Vec3f RayOrign(RayOrign0, RayOrign1, RayOrign2);
		Vec3f A(A0[offset], A1[offset], A2[offset]);
		Vec3f B(B0[offset], B1[offset], B2[offset]);
		Vec3f C(C0[offset], C1[offset], C2[offset]);
		IntersectLineAndTriangle(R, RayDir, RayOrign, A, B, C);
		R0[offset] = R[0];
		R1[offset] = R[1];
		R2[offset] = R[2];
	}
}

__global__ void RayKernel(float* R0, float* R1, float* R2,
	                      const float *RayDir0, const float *RayDir1, const float *RayDir2,
						  const float *RayOrign0, const float *RayOrign1, const float *RayOrign2,
	                      const float *A0, const float *A1, const float *A2,
	                      const float *B0, const float *B1, const float *B2,
	                      const float *C0, const float *C1, const float *C2)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = y*gridDim.x*blockDim.x + x;
	if (offset < RayDIM*RayDIM)
	{
		int nDIM = RayDIM % 16 == 0 ? RayDIM / 16 : RayDIM / 16 + 1;
		dim3 grid(nDIM, nDIM);
		dim3 block(16, 16);
		IntersectKernel << <grid, block >> >(R0, R1, R2, 
			RayDir0[offset], RayDir1[offset], RayDir1[offset],
			RayOrign0[offset], RayOrign1[offset], RayOrign2[offset],
			A0, A1, A2, B0, B1, B2, C0, C1, C2);
	}
}

__global__ void ToothKernel(float* R0, float* R1, float* R2,
	const float *RayDir0, const float *RayDir1, const float *RayDir2,
	const float *RayOrign0, const float *RayOrign1, const float *RayOrign2,
	const float *A0, const float *A1, const float *A2,
	const float *B0, const float *B1, const float *B2,
	const float *C0, const float *C1, const float *C2)
{
	int tid = blockIdx.x;
	if (tid < ToothNum)
	{
		int nDIM = RayDIM % 16 == 0 ? RayDIM / 16 : RayDIM / 16 + 1;
		dim3 grid(nDIM, nDIM);
		dim3 block(16, 16);
		RayKernel << <grid, block >> >(R0, R1, R2,
			RayDir0, RayDir1, RayDir2, RayOrign0, RayOrign1, RayOrign2,
			A0, A1, A2, B0, B1, B2, C0, C1, C2);
	}
}

int main()
{
	//  get device properties
	cudaDeviceProp maxProp;
	int nNumDevices;
	cudaGetDeviceCount(&nNumDevices);
	if (nNumDevices >= 1)
	{
		int maxMultiProcessors = 0, maxDevice = 0;
		for (int device = 0; device < nNumDevices; ++device)
		{
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, device);
			if (maxMultiProcessors < prop.multiProcessorCount)
			{
				maxMultiProcessors = prop.multiProcessorCount;
				maxDevice = device;
				maxProp = prop;
			}
		}

		cudaSetDevice(maxDevice);
	}
	

	int driverVersion, runtimeVersion;
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);


	// Init Ray
	int nRayNum = RayDIM*RayDIM;
	float *RayDir0 = (float*)malloc(sizeof(float)*nRayNum);
	float *RayDir1 = (float*)malloc(sizeof(float)*nRayNum);
	float *RayDir2 = (float*)malloc(sizeof(float)*nRayNum);

	float *RayOrign0 = (float*)malloc(sizeof(float)*nRayNum);
	float *RayOrign1 = (float*)malloc(sizeof(float)*nRayNum);
	float *RayOrign2 = (float*)malloc(sizeof(float)*nRayNum);

	
	for (int i = 0; i < nRayNum; ++i)
	{
		RayDir0[i] = sqrtf(3.f) / 3.f;
		RayDir1[i] = sqrtf(3.f) / 3.f;
		RayDir2[i] = sqrtf(3.f) / 3.f;

		RayOrign0[i] = 0.f;
		RayOrign1[i] = 0.f;
		RayOrign2[i] = 0.f;
	}

	// Init Triangle
	int nTriNum = TriDIM*TriDIM;
	float *A0 = (float*)malloc(sizeof(float)*nTriNum);
	float *A1 = (float*)malloc(sizeof(float)*nTriNum);
	float *A2 = (float*)malloc(sizeof(float)*nTriNum);

	float *B0 = (float*)malloc(sizeof(float)*nTriNum);
	float *B1 = (float*)malloc(sizeof(float)*nTriNum);
	float *B2 = (float*)malloc(sizeof(float)*nTriNum);

	float *C0 = (float*)malloc(sizeof(float)*nTriNum);
	float *C1 = (float*)malloc(sizeof(float)*nTriNum);
	float *C2 = (float*)malloc(sizeof(float)*nTriNum);

	for (int i = 0; i < nTriNum; ++i)
	{
		A0[i] = 1.f;
		A1[i] = 0.f;
		A2[i] = 0.f;

		B0[i] = 0.f;
		B1[i] = 1.f;
		B2[i] = 0.f;

		C0[i] = 0.f;
		C1[i] = 0.f;
		C2[i] = 1.f;
	}

	// Init Result
	float *R0 = (float*)malloc(sizeof(float)*nRayNum*nTriNum);
	float *R1 = (float*)malloc(sizeof(float)*nRayNum*nTriNum);
	float *R2 = (float*)malloc(sizeof(float)*nRayNum*nTriNum);

	// Init Device Memeory
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	// Ray
	float *dRayDir0, *dRayDir1, *dRayDir2, *dRayOrign0, *dRayOrign1, *dRayOrign2;
	HANDLE_ERROR(cudaMalloc((void**)&dRayDir0, sizeof(float)*nRayNum));
	HANDLE_ERROR(cudaMalloc((void**)&dRayDir1, sizeof(float)*nRayNum));
	HANDLE_ERROR(cudaMalloc((void**)&dRayDir2, sizeof(float)*nRayNum));
	HANDLE_ERROR(cudaMemcpy(dRayDir0, RayDir0, sizeof(float)*nRayNum, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dRayDir1, RayDir1, sizeof(float)*nRayNum, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dRayDir2, RayDir2, sizeof(float)*nRayNum, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void**)&dRayOrign0, sizeof(float)*nRayNum));
	HANDLE_ERROR(cudaMalloc((void**)&dRayOrign1, sizeof(float)*nRayNum));
	HANDLE_ERROR(cudaMalloc((void**)&dRayOrign2, sizeof(float)*nRayNum));
	HANDLE_ERROR(cudaMemcpy(dRayOrign0, RayOrign0, sizeof(float)*nRayNum, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dRayOrign1, RayOrign1, sizeof(float)*nRayNum, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dRayOrign2, RayOrign2, sizeof(float)*nRayNum, cudaMemcpyHostToDevice));

	// Triangle
	float *dA0, *dA1, *dA2, *dB0, *dB1, *dB2, *dC0, *dC1, *dC2;
	HANDLE_ERROR(cudaMalloc((void**)&dA0, sizeof(float)*nTriNum));
	HANDLE_ERROR(cudaMalloc((void**)&dA1, sizeof(float)*nTriNum));
	HANDLE_ERROR(cudaMalloc((void**)&dA2, sizeof(float)*nTriNum));
	HANDLE_ERROR(cudaMemcpy(dA0, A0, sizeof(float)*nTriNum, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dA1, A1, sizeof(float)*nTriNum, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dA2, A2, sizeof(float)*nTriNum, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void**)&dB0, sizeof(float)*nTriNum));
	HANDLE_ERROR(cudaMalloc((void**)&dB1, sizeof(float)*nTriNum));
	HANDLE_ERROR(cudaMalloc((void**)&dB2, sizeof(float)*nTriNum));
	HANDLE_ERROR(cudaMemcpy(dB0, B0, sizeof(float)*nTriNum, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dB1, B1, sizeof(float)*nTriNum, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dB2, B2, sizeof(float)*nTriNum, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void**)&dC0, sizeof(float)*nTriNum));
	HANDLE_ERROR(cudaMalloc((void**)&dC1, sizeof(float)*nTriNum));
	HANDLE_ERROR(cudaMalloc((void**)&dC2, sizeof(float)*nTriNum));
	HANDLE_ERROR(cudaMemcpy(dC0, C0, sizeof(float)*nTriNum, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dC1, C1, sizeof(float)*nTriNum, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dC2, C2, sizeof(float)*nTriNum, cudaMemcpyHostToDevice));

	// Result
	float *dR0, *dR1, *dR2;												
	HANDLE_ERROR(cudaMalloc((void**)&dR0, sizeof(float)*nRayNum*nTriNum));
	HANDLE_ERROR(cudaMalloc((void**)&dR1, sizeof(float)*nRayNum*nTriNum));
	HANDLE_ERROR(cudaMalloc((void**)&dR2, sizeof(float)*nRayNum*nTriNum));

	ToothKernel << <ToothNum, 1 >> >(dR0, dR1, dR2,
		dRayDir0, dRayDir1, dRayDir2, dRayOrign0, dRayOrign1, dRayOrign2,
		dA0, dA1, dA2, dB0, dB1, dB2, dC0, dC1, dC2);

	//dim3 grid(RayDIM / 16, RayDIM / 16);
	//dim3 block(16, 16);
	//RayKernel << <grid, block >> >(dR0, dR1, dR2, 
	//	                           dRayDir0, dRayDir1, dRayDir2, dRayOrign0, dRayOrign1, dRayOrign2, 
	//	                           dA0, dA1, dA2, dB0, dB1, dB2, dC0, dC1, dC2);
	//IntersectKernel << <grid, block >> >(dR0, dR1, dR2, dA0, dA1, dA2, dB0, dB1, dB2, dC0, dC1, dC2);
	cudaThreadSynchronize();
	HANDLE_ERROR(cudaGetLastError());
	HANDLE_ERROR(cudaMemcpy(R0, dR0, sizeof(float)*nRayNum*nTriNum, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(R1, dR1, sizeof(float)*nRayNum*nTriNum, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(R2, dR2, sizeof(float)*nRayNum*nTriNum, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Time to hash: %3.1f ms\n", elapsedTime);
	system("pause");
	HANDLE_ERROR(cudaFree(dR0));
	HANDLE_ERROR(cudaFree(dR1));
	HANDLE_ERROR(cudaFree(dR2));
	HANDLE_ERROR(cudaFree(dA0));
	HANDLE_ERROR(cudaFree(dA1));
	HANDLE_ERROR(cudaFree(dA2));
	HANDLE_ERROR(cudaFree(dB0));
	HANDLE_ERROR(cudaFree(dB1));
	HANDLE_ERROR(cudaFree(dB2));
	HANDLE_ERROR(cudaFree(dC0));
	HANDLE_ERROR(cudaFree(dC1));
	HANDLE_ERROR(cudaFree(dC2));
	//
	/*CPUBitmap bitmap(DIM, DIM);
	unsigned char* dev_bitmap;
	HANDLE_ERROR(cudaMalloc(&dev_bitmap, bitmap.image_size()));
	dim3 grid(DIM / 16, DIM / 16);
	dim3 block(16, 16);
	kernel << <grid, block >> >(dev_bitmap);
	cudaThreadSynchronize();
	cudaError_t temp = cudaGetLastError();
	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap,
		         bitmap.image_size(), cudaMemcpyDeviceToHost));
	bitmap.display_and_exit();
	HANDLE_ERROR(cudaFree(dev_bitmap));



    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t HANDLE_ERROR(addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    HANDLE_ERROR(cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }*/

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
