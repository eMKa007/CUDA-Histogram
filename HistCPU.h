#pragma once

#include <chrono>
#include <stdexcept>

class HistCPU
{
	private:	/* Variables */
		int*	imageArray		= nullptr;
		int		imageArraySize	= 0;
		int*	histogramCPU	= nullptr;
		int		NumberOfExec	= 0;
		double	MeanComputeTime = 0;

	public:
		HistCPU( int* imageArray, int imageArraySize, int* histogramCPU, int NumberOfExec );
		~HistCPU();

		void	Test_CPU_Execution();
		void	PrintHistogramAndExecTime();

	private:
		double	CPU_Histogram();
		void	PrintComputeTime();
		void	PrintHistogram();
	
};

