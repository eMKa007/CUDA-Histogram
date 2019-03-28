
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
	
#include "./Public/HistCPU.h"
#include "./Public/Image.h"

/* Functions */
int checkArguments( int argc, char* argv[] );
void PrintUsage();
float GPU_Histogram( int* inputArray, int* HistogramGPU, unsigned int size);
__global__ void GPU_Histogram_Kernel( int* inputArray, int inputArraySize, int* HistogramGPU );

int main( int argc, char* argv[])
{
	/*	----------------------------------------------------------
	*	Parse arguments from command line.
	*/
	int NumberOfExecutions = 0;
	if( (NumberOfExecutions = checkArguments( argc, argv ) ) == 0 )
	{
		PrintUsage();
		exit(-1);
	}

	/*	----------------------------------------------------------
	*	Load in image and get its size.
	*/
	Image* ImagePtr = nullptr;
	unsigned int imgArraySize = 0;
	try 
	{
		ImagePtr = new Image(argv[1]);
		imgArraySize = ImagePtr->GetArraySize();
	}
	catch (Exception ex)
	{
		printf("Image class throw an exception: %s.\n", ex.what());
		exit(-1);
	}
	
	/*	----------------------------------------------------------
	*	Alloc memory for 1d image pixel table, and two histograms.
	*/
	int* imageArray = (int*)calloc(imgArraySize, sizeof(int));
	int* histogramCPU = (int*)calloc(256, sizeof(int));
	int* histogramGPU = (int*)calloc(256, sizeof(int));

	if( !imageArray || !histogramCPU || !histogramGPU )
	{
		printf("Memory allocation error.");
		exit(-1);
	}

	/*	----------------------------------------------------------
	*	Fill in Image array. Show image and procceed array size. 
	*/
	ImagePtr->img2array(imageArray);
	ImagePtr->ShowInputImage("After any key press- computing will start. Wait till end :)");
	
	/*	----------------------------------------------------------
	*	CPU computing time test code.
	*/
	try
	{
		HistCPU CPU_Test(imageArray, imgArraySize, histogramCPU, NumberOfExecutions);
		CPU_Test.Test_CPU_Execution();
		CPU_Test.PrintHistogramAndExecTime();
		CPU_Test.~HistCPU();
	}
	catch (std::exception ex)
	{
		printf("CPU_Test throw an exception: %s.\n", ex.what() );
		exit(-1);
	}

	/*	----------------------------------------------------------
	*	GPU computing time test code.
	*/
	float DurationGPU = GPU_Histogram( imageArray, histogramGPU, imgArraySize );
	printf("Duration: %f [ms], which is about %f [s]\n", DurationGPU, (DurationGPU/1000.f));
    
	/*	----------------------------------------------------------
	*	Cleaning resources.
	*/
	free( imageArray );
	free( histogramCPU );
	free( histogramGPU );

    return 0;
}

/*	----------------------------------------------------------
*	Function name:	checkArguments
*	Parameters:		argc <int>, argv <char**>
*	Used to:		Check input arguments and parse NumberOfExecutions number to valid one.
*	Return:			Number of executions in integer.
*/
int checkArguments(int argc, char* argv[])
{
	if( argc < 3 )
		return false;

	char* EndPtr;
	int NumberOfExecutions = 0;

	NumberOfExecutions = strtod(argv[2], &EndPtr);
	if( *EndPtr != '\0' )
		return false;

	return NumberOfExecutions;
}

/*	----------------------------------------------------------
*	Function name:	PrintUsage
*	Parameters:		None.
*	Used to:		Print out message how to load an image, what is optimal number of executions etc.
*	Return:			None.
*/
void PrintUsage()
{
	printf("Usage: \n\tprogramname.exe <imageName.jpg> <NumberOfExecutions>\n");
	printf("\n\tTips: Locate image in the same folder as this *.exe file.\n");
	printf("\tNumberOfExecutions [integer] above 10000 can cause problems. Optimal: 1000 - 5000.\n");
}

float GPU_Histogram( int* inputArray, int* HistogramGPU, unsigned int inputArraySize)
{
	int* dev_inputArray = 0;
	int* dev_Histogram = 0;
	cudaError_t cudaStatus;

	//Assume, we will use first GPU device.
	cudaStatus = cudaSetDevice(0);
	if( cudaStatus != cudaSuccess )
	{
		printf("cudaSetDevice() fail! Do you have CUDA available device?\n");
		exit(-1);
	}

	// Cuda events used to measure execution time.
	cudaEvent_t start, stop;

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

	cudaEventRecord(start);

	//Allocate space on GPU.
	cudaStatus = cudaMalloc( (void**)&dev_inputArray, inputArraySize * sizeof(int) );
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaMalloc() fail! Can not allocate memory on GPU.\n");
		exit(-1);
	}

	cudaStatus = cudaMalloc( (void**)&dev_Histogram, 256 * sizeof(int) );
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaMalloc() fail! Can not allocate memory on GPU.\n");
		exit(-1);
	}

	// Initialise device Histogram with 0
	cudaStatus = cudaMemset( dev_Histogram, 0, 256 * sizeof(int) );
	if( cudaStatus != cudaSuccess )
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

	//Check available number of multiprocesors on GPU device- it will be used in kernel function.
	cudaDeviceProp properties;
	cudaStatus = cudaGetDeviceProperties( &properties, 0 );
	if( cudaStatus != cudaSuccess )
	{
		printf("cudaGetDeviceProperties() fail.");
		exit(-1);
	}

	int blocks = properties.multiProcessorCount;

	//Launch kernel. ==============================================================================
	GPU_Histogram_Kernel<<<blocks*2, 256>>>( dev_inputArray, inputArraySize, dev_Histogram);

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

	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	cudaFree(dev_inputArray);
	cudaFree(dev_Histogram);

	return miliseconds;
}

__global__ void GPU_Histogram_Kernel(int* inputArray, int inputArraySize, int* HistogramGPU)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while( i < inputArraySize )
	{
		atomicAdd( &HistogramGPU[ inputArray[i] ], 1);
		i += stride;
	}
}

