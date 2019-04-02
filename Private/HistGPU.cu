#include "../Public/HistGPU.h"

HistGPU::HistGPU(int* inputArray_in, int inputArraySize_in, int* HistogramGPU_in) :
	inputArray(inputArray_in), inputArraySize(inputArraySize_in), HistogramGPU(HistogramGPU_in)
{
	if( !inputArraySize_in || 0 == inputArraySize_in || !HistogramGPU_in )
		throw std::invalid_argument("HistCPU class: Received invalid argument in constructor.");
}
HistGPU::~HistGPU()
{

}

void HistGPU::RunSingleTest_GPU()
{
	int* dev_inputArray = nullptr;
	int* dev_Histogram = nullptr;
	cudaError_t cudaStatus;

	cudaEventRecord(beforeAlloc);

	//Allocate space on GPU.
	cudaStatus = cudaMalloc((void**)&dev_inputArray, inputArraySize * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaMalloc() fail! Can not allocate memory on GPU.\n");
		throw(cudaStatus);
	}

	cudaStatus = cudaMalloc((void**)&dev_Histogram, 256 * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaMalloc() fail! Can not allocate memory on GPU.\n");
		throw(cudaStatus);
	}

	// Initialize device Histogram with 0
	cudaStatus = cudaMemset(dev_Histogram, 0, 256 * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaMemset() fail! Can not set memory on GPU.\n");
		throw(cudaStatus);
	}

	// Copy input to previously allocated memory on GPU.
	cudaStatus = cudaMemcpy(dev_inputArray, inputArray, inputArraySize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy() fail! Can not copy data to GPU device.\n");
		throw(cudaStatus);
	}

	//Check available number of multiprocessors on GPU device- it will be used in kernel function.
	cudaDeviceProp properties;
	cudaStatus = cudaGetDeviceProperties(&properties, 0);
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaGetDeviceProperties() fail.");
		throw(cudaStatus);
	}

	cudaEventRecord(beforeCompute);

	//Launch kernel. ==============================================================================
	int blocks = properties.multiProcessorCount;
	GPU_Histogram_Kernel << <blocks * 2, 256 >> > (dev_inputArray, inputArraySize, dev_Histogram);

	cudaEventRecord(afterCompute);
	cudaEventSynchronize(afterCompute);

	// Check for kernel errors.
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("GPU_Histogram() kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// Wait for kernel to finish work, and check for any errors during kernel work.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize() returned error code %d after launching!\n", cudaStatus);
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(HistogramGPU, dev_Histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy() device to host failed!");
	}

	cudaEventRecord(afterAlloc);
	cudaEventSynchronize(afterAlloc);

	float withAllocation = 0;
	float woAllocation = 0;
	cudaEventElapsedTime(&withAllocation, beforeAlloc, afterAlloc);
	cudaEventElapsedTime(&woAllocation, beforeCompute, afterCompute);
	totalMiliseconds_withAllocation += withAllocation;
	totalMiliseconds_woAllocation += woAllocation;

	cudaFree(dev_inputArray);
	cudaFree(dev_Histogram);
}



void HistGPU::Test_GPU(unsigned int NumberOfExec)
{
	cudaError_t cudaStatus;

	//Assume, we will use first GPU device.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaSetDevice() fail! Do you have CUDA available device?\n");
		throw(cudaStatus);
	}

	// Cuda events used to measure execution time.
	CreateTimeEvents();
	
	for (int TryNumber = 0; TryNumber < NumberOfExec; TryNumber++)
	{
		RunSingleTest_GPU();
	}

	cudaEventDestroy(beforeAlloc);
	cudaEventDestroy(afterAlloc);
	cudaEventDestroy(beforeCompute);
	cudaEventDestroy(afterCompute);
	
	ComputeMeanTimes(NumberOfExec);
}

void HistGPU::CreateTimeEvents()
{
	cudaError_t cudaStatus;
	cudaEvent_t* Event[4] = { &beforeAlloc, &beforeCompute, &afterAlloc, &afterCompute };

	for (int i = 0; i < 4; i++)
	{
		cudaStatus = cudaEventCreate(Event[i]);
		if (cudaStatus != cudaSuccess) {
			printf("cudaEventCreate() fail! Can not create beforeAlloc event to measure execution time.\n");
			throw(cudaStatus);
		}
	}

	/*cudaStatus = cudaEventCreate(&beforeAlloc);
	if (cudaStatus != cudaSuccess) {
		printf("cudaEventCreate() fail! Can not create beforeAlloc event to measure execution time.\n");
		throw(cudaStatus);
	}

	cudaStatus = cudaEventCreate(&beforeCompute);
	if (cudaStatus != cudaSuccess) {
		printf("cudaEventCreate() fail! Can not create beforeCompute event to measure execution time.\n");
		throw(cudaStatus);
	}

	cudaStatus = cudaEventCreate(&afterAlloc);
	if (cudaStatus != cudaSuccess) {
		printf("cudaEventCreate() fail! Can not create afterAlloc event to measure execution time.\n");
		throw(cudaStatus);
	}

	cudaStatus = cudaEventCreate(&afterCompute);
	if (cudaStatus != cudaSuccess) {
		printf("cudaEventCreate() fail! Can not create afterCompute event to measure execution time.\n");
		throw(cudaStatus);
	}*/
}

void HistGPU::ComputeMeanTimes(unsigned int NumberOfExec)
{
	msWithAlloc = totalMiliseconds_withAllocation / NumberOfExec;
	msWithoutAlloc = totalMiliseconds_woAllocation / NumberOfExec;
}

__global__ void GPU_Histogram_Kernel(int* inputArray, int inputArraySize, int* HistogramGPU)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (i < inputArraySize)
	{
		atomicAdd(&HistogramGPU[inputArray[i]], 1);
		i += stride;
	}
}