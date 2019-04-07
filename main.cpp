
#include <stdio.h>
#include <string>

#include "Public/HistCPU.h"
#include "Public/Image.h"
#include "Public/HistGPU.h"

/* Functions */
void MainTestFunction(Image* ImagePtr,  unsigned int imgArraySize, int NumberOfExecutions);
int checkArguments(int argc, char* argv[]);
void PrintUsage();

int main(int argc, char* argv[])
{
	/*	----------------------------------------------------------
	*	Parse arguments from command line.
	*/
	int NumberOfExecutions = 0;
	if ((NumberOfExecutions = checkArguments(argc, argv)) == 0)
	{
		PrintUsage();
		exit(-1);
	}

	/*	----------------------------------------------------------
	*	Load in image and get its size.
	*/
	Image* ImagePtr = nullptr;
	unsigned int imgArraySize = 0;
	try
	{
		ImagePtr = new Image(argv[1]);
		imgArraySize = ImagePtr->GetArraySize();
	}
	catch (Exception ex)
	{
		printf("Image class throw an exception: %s.\n", ex.what());
		exit(-1);
	}

	MainTestFunction(ImagePtr, imgArraySize, NumberOfExecutions);

	return 0;
}

/*	----------------------------------------------------------
*	Function name:	MainTestFunction
*	Parameters:		Image* ImagePtr, unsigned int imgArraySize
*	Used to:		
*	Return:			
*/
void MainTestFunction(Image* ImagePtr, unsigned int imgArraySize, int NumberOfExecutions)
{
	/*	----------------------------------------------------------
	*	Alloc memory for 1d image pixel table, and two histograms.
	*/
	int* imageArray = new int[imgArraySize]();
	int* histogramCPU = new int[256]();
	int* histogramGPU = new int[256]();

	if (!imageArray || !histogramCPU || !histogramGPU)
	{
		printf("Memory allocation error.");
		exit(-1);
	}

	/*	----------------------------------------------------------
	*	Fill in Image array. Show image and proceed array size.
	*/
	ImagePtr->img2array(imageArray);
	ImagePtr->ShowInputImage("After any key press- computing will start. Wait till end :)");

	/*	----------------------------------------------------------
	*	GPU computing time test case.
	*/
	try
	{
		HistGPU GPU_Test(imageArray, imgArraySize, histogramGPU);
		GPU_Test.Test_GPU(NumberOfExecutions);
		GPU_Test.PrintMeanComputeTime();
		GPU_Test.~HistGPU();
	}
	catch (cudaError_t ex)
	{
		printf(" Cuda error: %s\n", cudaGetErrorString(ex));
		exit(-1);
	}

	/*	----------------------------------------------------------
	*	CPU computing time test case.
	*/
	try
	{
		HistCPU CPU_Test(imageArray, imgArraySize, histogramCPU, NumberOfExecutions);
		CPU_Test.Test_CPU_Execution();
		CPU_Test.PrintHistogramAndExecTime();
		CPU_Test.~HistCPU();
	}
	catch (std::exception ex)
	{
		printf("CPU_Test throw an exception: %s.\n", ex.what());
		exit(-1);
	}


	/*	----------------------------------------------------------
	*	Cleaning resources.
	*/
	delete[] imageArray;
	delete[] histogramCPU;
	delete[] histogramGPU;
}

/*	----------------------------------------------------------
*	Function name:	checkArguments
*	Parameters:		argc <int>, argv <char**>
*	Used to:		Check input arguments and parse NumberOfExecutions number to valid one.
*	Return:			Number of executions as integer.
*/
int checkArguments(int argc, char* argv[])
{
	if (argc < 3)
		return false;

	char* EndPtr;
	int NumberOfExecutions = 0;

	NumberOfExecutions = strtod(argv[2], &EndPtr);
	if (*EndPtr != '\0')
		return 0;

	return NumberOfExecutions;
}

/*	----------------------------------------------------------
*	Function name:	PrintUsage
*	Parameters:		None.
*	Used to:		Print out message how to load an image, what is optimal number of executions etc.
*	Return:			None.
*/
void PrintUsage()
{
	printf("Usage: \n\tprogramname.exe <imageName.jpg> <NumberOfExecutions>\n");
	printf("\n\tTips: Locate image in the same folder as this *.exe file.\n");
	printf("\tNumberOfExecutions [integer] above 10000 can cause problems. Optimal: 1000 - 5000.\n");
}



