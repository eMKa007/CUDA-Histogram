
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>

#include <opencv2/opencv.hpp>
using namespace cv;

#include "CPU_Hist.h"

/* Functions */
void PrintHistogramAndExecTime(int * histogram, double durationCPU);
int img2array(cv::Mat &img, int * &histogramCPU);
bool checkArguments( int argc, char* argv[], int* NumberOfExeutions );
void PrintUsage();


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main( int argc, char* argv[])
{
	int NumberOfExecutions = 0;
	if( !checkArguments( argc, argv, &NumberOfExecutions ) )
	{
		PrintUsage();
		exit(-1);
	}

	//OpenCV test case. 
	cv::Mat img = imread(argv[1], IMREAD_GRAYSCALE);
	if( img.cols == 0 | img.rows == 0 )
	{
		PrintUsage();
		exit(-1);
	}

	namedWindow("After enter press- computing will start. Wait till end :)", WINDOW_NORMAL);
	imshow("After enter press- computing will start. Wait till end :)", img);
	waitKey(0);
	destroyWindow("After enter press- computing will start. Wait till end :)");

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

	double meanDurationCPU = Test_CPU_Execution(imageArray, imgArraySize, histogramCPU, NumberOfExecutions);
	PrintHistogramAndExecTime( histogramCPU, meanDurationCPU );


    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };

    //// Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);

    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}

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

void PrintUsage()
{
	printf("Usage: \n\tprogramname.exe <imageName.jpg> <NumberOfExecutions>\n\n\tTips: Locate image in the same folder as this *.exe file.\n\tNumberOfExecutions [integer] above 10000 can cause problems. Optimal: 1000 - 5000.\n");
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
