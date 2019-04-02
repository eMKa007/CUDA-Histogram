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
double HistCPU::CPU_Histogram()
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
		totalExecTime += CPU_Histogram();
		trialNo++;

		if (trialNo > NumberOfExec)
			break;

		memset(histogramCPU, 0, 256 * sizeof(int));			//Clear computed Histogram.
	}

	MeanComputeTime = totalExecTime/NumberOfExec;
}

/*	----------------------------------------------------------
*	Function name:	PrintHistogramAndExecTime
*	Parameters:		None
*	Used to:		Printout to stdout computed histogram and information about it's mean computing time.
*	Return:			None
*/
void HistCPU::PrintHistogramAndExecTime()
{
	PrintHistogram();
	PrintComputeTime();	
}

/*	----------------------------------------------------------
*	Function name:	PrintHistogram
*	Parameters:		None
*	Used to:		Printout to stdout information about histogram's mean computing time.
*	Return:			None
*/
void HistCPU::PrintComputeTime()
{
	printf("Mean histogram computing time: %f [ms]\n", MeanComputeTime);
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

