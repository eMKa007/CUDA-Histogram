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

float HistGPU::RunSingleTest_GPU()
{
	int* dev_inputArray = nullptr;
	int* dev_Histogram = nullptr;
	cudaError_t cudaStatus;

	cudaEventRecord(start);

	//Allocate space on GPU.
	cudaStatus = cudaMalloc((void**)&dev_inputArray, inputArraySize * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaMalloc() fail! Can not allocate memory on GPU.\n");
		exit(-1);
	}

	cudaStatus = cudaMalloc((void**)&dev_Histogram, 256 * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaMalloc() fail! Can not allocate memory on GPU.\n");
		exit(-1);
	}

	// Initialize device Histogram with 0
	cudaStatus = cudaMemset(dev_Histogram, 0, 256 * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaMemset() fail! Can not set memory on GPU.\n");
		exit(-1);
	}

	// Copy input to previously allocated memory on GPU.
	cudaStatus = cudaMemcpy(dev_inputArray, inputArray, inputArraySize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy() fail! Can not copy data to GPU device.\n");
		exit(-1);
	}

	//Check available number of multiprocessors on GPU device- it will be used in kernel function.
	cudaDeviceProp properties;
	cudaStatus = cudaGetDeviceProperties(&properties, 0);
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaGetDeviceProperties() fail.");
		exit(-1);
	}

	//Launch kernel. ==============================================================================
	int blocks = properties.multiProcessorCount;
	GPU_Histogram_Kernel << <blocks * 2, 256 >> > (dev_inputArray, inputArraySize, dev_Histogram);

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

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float miliseconds = 0;
	cudaEventElapsedTime(&miliseconds, start, stop);

	cudaFree(dev_inputArray);
	cudaFree(dev_Histogram);
	
	return miliseconds;
}



float HistGPU::Test_GPU(unsigned int NumberOfExec)
{
	cudaError_t cudaStatus;

	//Assume, we will use first GPU device.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaSetDevice() fail! Do you have CUDA available device?\n");
		exit(-1);
	}

	// Cuda events used to measure execution time.
	CreateTimeEvents();
	
	float TotalTime = 0;
	for (int TryNumber = 0; TryNumber < NumberOfExec; TryNumber++)
	{
		TotalTime += RunSingleTest_GPU();
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	return TotalTime/NumberOfExec;
}

void HistGPU::CreateTimeEvents()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaEventCreate(&start);
	if (cudaStatus != cudaSuccess) {
		printf("cudaEventCreate() fail! Can not create start event to measure execution time.\n");
		exit(-1);
	}

	cudaStatus = cudaEventCreate(&stop);
	if (cudaStatus != cudaSuccess) {
		printf("cudaEventCreate() fail! Can not create start event to measure execution time.\n");
		exit(-1);
	}
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