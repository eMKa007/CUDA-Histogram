#include "CPU_Hist.h"
#include <chrono>

long long CPU_Histogram(int * imageArray, int imageArraySize, int * histogramCPU)
{
	int idx = 0;
	std::chrono::high_resolution_clock::time_point clockBefore = std::chrono::high_resolution_clock::now();

	while ( idx < imageArraySize )
	{
		histogramCPU[ imageArray[idx] ]++;
		idx++;
	}

	std::chrono::high_resolution_clock::time_point clockAfter = std::chrono::high_resolution_clock::now();
	auto durationCPU = std::chrono::duration_cast<std::chrono::milliseconds>(clockAfter - clockBefore).count();

	return durationCPU;
}

double Test_CPU_Execution(int* imageArray, int imageArraySize, int* histogramCPU, int NumberOfExec )
{
	int trialNo = 0;
	double totalExecTime = 0;

	while( 1 /*trialNo < NumberOfExec*/ )
	{
		totalExecTime += CPU_Histogram( imageArray, imageArraySize, histogramCPU);
		trialNo++;

		if( trialNo > NumberOfExec )
			break;
		
		memset( histogramCPU, 0, 256*sizeof(int) );
	}

	return totalExecTime/NumberOfExec;
}
