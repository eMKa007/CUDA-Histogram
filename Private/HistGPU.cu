#include "../Public/HistGPU.h"

HistGPU::HistGPU(int* inputArray_in, int inputArraySize_in, int* HistogramGPU_in) :
	inputArray(inputArray_in), inputArraySize(inputArraySize_in), HistogramGPU(HistogramGPU_in)
{
	if( !inputArraySize_in || 0 == inputArraySize_in || !HistogramGPU_in )
		throw std::invalid_argument("HistCPU class: Received invalid argument in constructor.");
}
HistGPU::~HistGPU() { }

/*	----------------------------------------------------------
*	Function name:	RunSingleTest_GPU
*	Parameters:		None
*	Used to:		Compute histogram with GPU strictly by adding every pixel value occurrence of input image to 256's histogram array.
*	Return:			None. Updating values of TotalComputeTime.
*/
void HistGPU::RunSingleTest_GPU(int blocks)
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

	cudaEventRecord(beforeCompute);

	//Launch kernel. ==============================================================================
	GPU_Histogram_Kernel << <blocks*16, 256 >> > (dev_inputArray, inputArraySize, dev_Histogram);

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

/*	----------------------------------------------------------
*	Function name:	Test_GPU
*	Parameters:		unsigned int NumberOfExec - How many times did GPU will be tested.
*	Used to:		Run GPU test exactly number of times and compute mean execution time.
*	Return:			None. Update vaues of mean cumpute time. 
*/
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

	//Check available number of multiprocessors on GPU device- it will be used in kernel function.
	cudaDeviceProp properties;
	cudaStatus = cudaGetDeviceProperties(&properties, 0);
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaGetDeviceProperties() fail.");
		throw(cudaStatus);
	}
	
	for (int TryNumber = 0; TryNumber < NumberOfExec; TryNumber++)
	{
		RunSingleTest_GPU( properties.multiProcessorCount );
	}

	cudaEventDestroy(beforeAlloc);
	cudaEventDestroy(afterAlloc);
	cudaEventDestroy(beforeCompute);
	cudaEventDestroy(afterCompute);
	
	ComputeMeanTimes(NumberOfExec);
}

/*	----------------------------------------------------------
*	Function name:	CreateTimeEvents
*	Parameters:		None.
*	Used to:		Create events used to measure compute time.
*	Return:			None. Events are created.
*/
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
}

/*	----------------------------------------------------------
*	Function name:	ComputeMeanTimes
*	Parameters:		unsigned int NumberOfExec - number of cycles, GPU was tested.
*	Used to:		Determine mean value of computing time.
*	Return:			None. Public values are updated.
*/
void HistGPU::ComputeMeanTimes(unsigned int NumberOfExec)
{
	msWithAlloc = totalMiliseconds_withAllocation / NumberOfExec;
	msWithoutAlloc = totalMiliseconds_woAllocation / NumberOfExec;
}

/*	----------------------------------------------------------
*	Function name:	PrintGPUInfo
*	Parameters:		None.
*	Used to:		Printout to stdout information about GPU device.
*	Return:			None.
*/
void HistGPU::PrintGPUInfo()
{
	cudaDeviceProp inf;
	cudaError_t cudaStatus = cudaGetDeviceProperties(&inf, 0);
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaGetDeviceProperties() fail.");
		throw(cudaStatus);
	}

	printf("*************************** CPU Info  *****************************\n");
	printf("GPU Device Name: \t\t%s\n", inf.name);
	printf("Number of Muliprocessors:\t%d\n", inf.multiProcessorCount);
	printf("Clock rate:\t\t\t%f [GHz]\n", inf.clockRate/1000000.f);
	printf("Major compute capability:\t\t%d\n", inf.major);
	printf("Max size of each dimension block:\t%d\n", inf.maxThreadsDim[0], inf.maxThreadsDim[1], inf.maxThreadsDim[2]);
	printf("Max number of threads per block:\t%d\n", inf.maxThreadsPerBlock);
	printf("*******************************************************************\n");
}

/*	----------------------------------------------------------
*	Function name:	PrintMeanComputeTime
*	Parameters:		None.
*	Used to:		Print out computed values.
*	Return:			None.
*/
void HistGPU::PrintMeanComputeTime()
{
	if (msWithAlloc == 0 || msWithoutAlloc == 0)
	{
		printf("GPU mean compute time is 0. Something happen wrong. Did you choose valid image?\n");
		cudaError_t exception = cudaError_t::cudaErrorInvalidValue;
		throw exception;
	}
		
	printf("Mean histogram computing time on GPU:\n");
	printf("  - with memory allocation: %f[ms], which is about %f[s]\n", msWithAlloc, (msWithAlloc / 1000.f));
	printf("  - without memory allocation: %f [ms], which is about %f [s]\n\n", msWithoutAlloc, (msWithoutAlloc / 1000.f));
}

/*	----------------------------------------------------------
*	Function name:	GPU_Histogram_Kernel
*	Parameters:		int* inputArray - Pointer to input array of pixel values. 
					int inputArraySize - Size of input array.
					int* HistogramGPU - Pointer to array storing computed values.
*	Used to:		Compute histogram with GPU. Main GPU function. Multithread function. 
*	Return:			None. Histogram on GPU is computed. 
*/
__global__ void GPU_Histogram_Kernel(int* inputArray, int inputArraySize, int* HistogramGPU)
{
	//Create and set to 0 local memory for single block.
	__shared__ unsigned int temp[256];
	temp[threadIdx.x] = 0;
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;

	while (i < inputArraySize)
	{
		atomicAdd(&temp[inputArray[i]], 1);
		i += offset;
	}
	__syncthreads();

	atomicAdd(&(HistogramGPU[threadIdx.x]), temp[threadIdx.x] );
}