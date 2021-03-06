#pragma once

#include <chrono>
#include <stdexcept>
#include <iostream>
#include <Windows.h>

class HistCPU
{
public:
	HistCPU(int* imageArray, int imageArraySize, int NumberOfExec);
	~HistCPU();

	int*		histogramCPU = nullptr;

	void		Test_CPU_Execution();
	void		PrintComputeTime();
	void		PrintCPUInfo();

private:
	int*		imageArray		= nullptr;
	int			imageArraySize	= 0;
	int			NumberOfExec	= 0;
	double		MeanComputeTime = 0;

	double		RunSingleTest_CPU();	
};

