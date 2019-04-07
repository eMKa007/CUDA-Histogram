#pragma once

#include <chrono>
#include <stdexcept>
#include <iostream>
#include <Windows.h>

class HistCPU
{
public:
	HistCPU(int* imageArray, int imageArraySize, int* histogramCPU, int NumberOfExec);
	~HistCPU();

	void		Test_CPU_Execution();
	void		PrintComputeTime();
	void		PrintCPUInfo();

	private:
		int*		imageArray		= nullptr;
		int			imageArraySize	= 0;
		int*		histogramCPU	= nullptr;
		int			NumberOfExec	= 0;
		double		MeanComputeTime = 0;

		double		RunSingleTest_CPU();
		void		PrintHistogram();
	
};

