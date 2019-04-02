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

	// Run Test_GPU() first! Measured time with resource allocation.
	float			msWithAlloc = 0;

	// Run Test_GPU() first! Measured time without resource allocation. Only computing time.
	float			msWithoutAlloc = 0;

	void			Test_GPU(unsigned int NumberOfExec);

private:
	int*			inputArray = nullptr;
	int*			HistogramGPU = nullptr;
	unsigned int	inputArraySize = 0;
	float			totalMiliseconds_withAllocation = 0;
	float			totalMiliseconds_woAllocation = 0;

	cudaEvent_t		beforeAlloc, afterAlloc;
	cudaEvent_t		beforeCompute, afterCompute;

	void			RunSingleTest_GPU();
	void			CreateTimeEvents();
	
	void			ComputeMeanTimes(unsigned int NumberOfExec);
};

__global__ void		GPU_Histogram_Kernel(int* inputArray, int inputArraySize, int* HistogramGPU);







