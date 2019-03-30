
#include <stdio.h>
#include <string>

#include "Public/HistCPU.h"
#include "Public/Image.h"
#include "Public/HistGPU.h"

/* Functions */
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

	/*	----------------------------------------------------------
	*	Alloc memory for 1d image pixel table, and two histograms.
	*/
	int* imageArray = (int*)calloc(imgArraySize, sizeof(int));
	int* histogramCPU = (int*)calloc(256, sizeof(int));
	int* histogramGPU = (int*)calloc(256, sizeof(int));

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
	*	CPU computing time test code.
	*/
	HistCPU CPU_Test(imageArray, imgArraySize, histogramCPU, NumberOfExecutions);
	try
	{
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
	*	GPU computing time test code.
	*/
	HistGPU GPU_Test(imageArray, imgArraySize, histogramGPU);
	float DurationGPU = GPU_Test.Test_GPU(NumberOfExecutions);
	printf("Duration: %f [ms], which is about %f [s]\n", DurationGPU, (DurationGPU / 1000.f));

	/*	----------------------------------------------------------
	*	Cleaning resources.
	*/
	free(imageArray);
	free(histogramCPU);
	free(histogramGPU);

	return 0;
}

/*	----------------------------------------------------------
*	Function name:	checkArguments
*	Parameters:		argc <int>, argv <char**>
*	Used to:		Check input arguments and parse NumberOfExecutions number to valid one.
*	Return:			Number of executions in integer.
*/
int checkArguments(int argc, char* argv[])
{
	if (argc < 3)
		return false;

	char* EndPtr;
	int NumberOfExecutions = 0;

	NumberOfExecutions = strtod(argv[2], &EndPtr);
	if (*EndPtr != '\0')
		return false;

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



