#pragma once

#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>


class HistGPU
{
public:
	HistGPU(int* inputArray, int inputArraySize, int* HistogramGPU);
	~HistGPU();

	//Measured time with resource allocation. Run Test_GPU() first! 
	float			msWithAlloc = 0;

	//Measured time without resource allocation. Only computing time. Run Test_GPU() first!
	float			msWithoutAlloc = 0;

	int*			HistogramGPU = nullptr;

	void			Test_GPU(unsigned int NumberOfExec);
	void			PrintMeanComputeTime();
	void			PrintGPUInfo();

private:
	int*			inputArray = nullptr;
	unsigned int	inputArraySize = 0;
	float			totalMiliseconds_withAllocation = 0;
	float			totalMiliseconds_woAllocation = 0;

	cudaEvent_t		beforeAlloc, afterAlloc;
	cudaEvent_t		beforeCompute, afterCompute;

	void			RunSingleTest_GPU(int blocks);
	void			CreateTimeEvents();
	void			ComputeMeanTimes(unsigned int NumberOfExec);

};

__global__ void		GPU_Histogram_Kernel(int* inputArray, int inputArraySize, int* HistogramGPU);







