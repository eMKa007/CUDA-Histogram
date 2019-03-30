#pragma once

#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>

class HistGPU
{
private:
	int* inputArray = nullptr;
	int* HistogramGPU = nullptr;
	unsigned int inputArraySize;

	cudaEvent_t start, stop;

public:
	HistGPU(int* inputArray, int inputArraySize, int* HistogramGPU);
	~HistGPU();

	float Test_GPU(unsigned int NumberOfExec);

private:
	float RunSingleTest_GPU();
	void CreateTimeEvents();
	
};

__global__ void GPU_Histogram_Kernel(int* inputArray, int inputArraySize, int* HistogramGPU);







