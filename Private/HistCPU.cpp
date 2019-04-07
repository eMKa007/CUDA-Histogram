#include "../Public/HistCPU.h"

/*	----------------------------------------------------------
*	Function name:	HistCPU class constructor.
*	Parameters:		int* imageArray_in		-	Pointer to input array of pixel values.
					int imageArraySize_in	-	Size of input image array. Number of pixels in image.
					int* histogramCPU_in	-	Pointer to array storing computed values.
					int NumberOfExec_in		-	Number of computing process to estimate mean compute time.
*	Used to:		Compute histogram with CPU strictly by adding every pixel value occurrence of input image to 256's histogram array.
*	Return:			Computing time in ms <double>.
*/
HistCPU::HistCPU(int* imageArray_in, int imageArraySize_in, int* histogramCPU_in, int NumberOfExec_in) : 
	imageArray(imageArray_in), imageArraySize(imageArraySize_in), histogramCPU(histogramCPU_in), NumberOfExec(NumberOfExec_in)
{
	if (!imageArray_in || 0 == imageArraySize_in || !histogramCPU_in || 0 == NumberOfExec_in)
		throw std::invalid_argument("HistCPU class: Received invalid argument in constructor.");
}
HistCPU::~HistCPU()
{
}

/*	----------------------------------------------------------
*	Function name:	CPU_Histogram
*	Parameters:		None
*	Used to:		Compute histogram with CPU strictly by adding every pixel value occurrence of input image to 256's histogram array.
*	Return:			Computing time in ms <double>.
*/	
double HistCPU::RunSingleTest_CPU()
{
	int idx = 0;
	
	std::chrono::high_resolution_clock::time_point clockBefore = std::chrono::high_resolution_clock::now();

	while (idx < imageArraySize)
	{
		histogramCPU[imageArray[idx]]++;
		idx++;
	}

	std::chrono::high_resolution_clock::time_point clockAfter = std::chrono::high_resolution_clock::now();
	auto durationCPU = std::chrono::duration_cast<std::chrono::milliseconds>(clockAfter - clockBefore).count();

	return (double)durationCPU;
}

/*	----------------------------------------------------------
*	Function name:	Test_CPU_Execution
*	Parameters:		None
*	Used to:		Test Histogram Computing Time for NumberOfExec times. Calculate mean value of computing time.
*	Return:			None. MeanComputeTime variable is updated.
*/
void HistCPU::Test_CPU_Execution()
{
	int trialNo = 0;
	double totalExecTime = 0;

	while( 1 )
	{
		totalExecTime += RunSingleTest_CPU();
		trialNo++;

		if (trialNo > NumberOfExec)
			break;

		memset(histogramCPU, 0, 256 * sizeof(int));			//Clear computed Histogram.
	}

	MeanComputeTime = totalExecTime/NumberOfExec;
}

/*	----------------------------------------------------------
*	Function name:	PrintComputeTime
*	Parameters:		None
*	Used to:		Printout to stdout information about histogram's mean computing time.
*	Return:			None
*/
void HistCPU::PrintComputeTime()
{
	printf("Mean histogram computing time on CPU:\n");
	printf("  - without memory allocation: %f [ms], which is about %f[s]\n", MeanComputeTime, MeanComputeTime / 1000.f);
}

/*	----------------------------------------------------------
*	Function name:	PrintCPUInfo
*	Parameters:		None
*	Used to:		Printout to stdout information about CPU device.
					Code partly from: https://stackoverflow.com/questions/850774/how-to-determine-the-hardware-cpu-and-ram-on-a-machine
*	Return:			None
*/
void HistCPU::PrintCPUInfo()
{
	LPSYSTEM_INFO inf = (LPSYSTEM_INFO)calloc(1, sizeof(LPSYSTEM_INFO));
	GetSystemInfo(inf);

	int CPUInfo[4] = { -1 };
	unsigned   nExIds, i = 0;
	char CPUBrandString[0x40];
	// Get the information associated with each extended ID.
	__cpuid(CPUInfo, 0x80000000);
	nExIds = CPUInfo[0];
	for (i = 0x80000000; i <= nExIds; ++i)
	{
		__cpuid(CPUInfo, i);
		// Interpret CPU brand string
		if (i == 0x80000002)
			memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
		else if (i == 0x80000003)
			memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
		else if (i == 0x80000004)
			memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
	}
	
	printf("*************************** CPU Info  *****************************\n");
	printf("CPU brand:\t\t%s\n", CPUBrandString);
	printf("Number of processors:\t%d\n", inf->dwNumberOfProcessors);
	printf("*******************************************************************\n");
}


/*	----------------------------------------------------------
*	Function name:	PrintComputeTime
*	Parameters:		None
*	Used to:		Printout to stdout computed histogram.
*	Return:			None
*/
void HistCPU::PrintHistogram()
{
	long sum = 0;
	for (int i = 0; i < 256; i++)
	{
		printf("%d. %d\n", i, histogramCPU[i]);
		sum += histogramCPU[i];
	}
	printf("\nTotal pixel number is: %d\n", sum);
}

