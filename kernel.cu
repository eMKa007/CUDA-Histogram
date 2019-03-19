
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>

#include "CPU_Hist.h"

#include <opencv2/opencv.hpp>
using namespace cv;



/* Functions */
void PrintHistogramAndExecTime(int * histogram, double durationCPU);
void ShowInputImage(cv::Mat &img);
int img2array(cv::Mat &img, int * &histogramCPU);
bool checkArguments( int argc, char* argv[], int* NumberOfExeutions );
void ShowInputImage(cv::Mat &img);
void PrintUsage();

float GPU_Histogram( int* inputArray, int* HistogramGPU, unsigned int size);
__global__ void GPU_Histogram_Kernel( int* inputArray, int inputArraySize, int* HistogramGPU );

int main( int argc, char* argv[])
{
	int NumberOfExecutions = 0;
	if( !checkArguments( argc, argv, &NumberOfExecutions ) )
	{
		PrintUsage();
		exit(-1);
	}

	//OpenCV input image. 
	cv::Mat img = imread(argv[1], IMREAD_GRAYSCALE);
	if( img.cols == 0 | img.rows == 0 )
	{
		PrintUsage();
		exit(-1);
	}

	ShowInputImage(img);

	//Alloc memory for 1d image pixel table, and two histograms.
	int* imageArray = (int*)calloc( img.rows*img.cols, sizeof(int));
	int* histogramCPU = (int*)calloc(256, sizeof(int));
	int* histogramGPU = (int*)calloc(256, sizeof(int));

	if( !imageArray || !histogramCPU || !histogramGPU )
	{
		printf("Memory allocation error.");
		return 0;
	}

	int imgArraySize = img2array(img, imageArray);

	//double meanDurationCPU = Test_CPU_Execution(imageArray, imgArraySize, histogramCPU, NumberOfExecutions);
	//PrintHistogramAndExecTime( histogramCPU, meanDurationCPU );
	  
	float DurationGPU = GPU_Histogram( imageArray, histogramGPU, imgArraySize );
	printf(" Duration: %ld [ms], which is about %f [s]\n", DurationGPU, DurationGPU/1000);
    //// Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}
	////GPU device have to be reset before exit. 
	//cudaStatus = cudaDeviceReset();
	//if (cudaStatus != cudaSuccess) {
	//    fprintf(stderr, "cudaDeviceReset failed!");
	//    return 1;

	free( imageArray );
	free( histogramCPU );
	free( histogramGPU );

    return 0;
}

void PrintHistogramAndExecTime(int* histogram, double durationCPU)
{
	long sum = 0;
	for (int i = 0; i < 256; i++)
	{
		printf("%d. %d\n", i, histogram[i]);
		sum += histogram[i];
	}

	printf("\nTotal pixel number is: %d\n", sum);
	printf("Mean histogram computing time: %f [ms]\n", durationCPU);
}

int img2array(cv::Mat &img, int * &imageArray)
{
	int Row = 0;
	int Col = 0;
	int idx = 0;
	while (Row < img.rows)
	{
		//Acces every pixel and count them by color.
		imageArray[idx++] = *(img.data + img.step[0] * Row + img.step[1] * Col++);

		if (Col == img.cols)
		{
			Col = 0;
			Row++;
		}
	}

	return img.rows*img.cols;
}

bool checkArguments(int argc, char* argv[], int* NumberOfExecutions)
{
	if( argc < 3 )
		return false;

	char* EndPtr;
	*NumberOfExecutions = strtod(argv[2], &EndPtr);
	if( *EndPtr != '\0' )
		return false;

	return true;
}

void ShowInputImage(cv::Mat &img)
{
	namedWindow("After enter press- computing will start. Wait till end :)", WINDOW_NORMAL);
	imshow("After enter press- computing will start. Wait till end :)", img);
	waitKey(0);
	destroyWindow("After enter press- computing will start. Wait till end :)");
}

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

