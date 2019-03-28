#include "../Public/Image.h"

/*
*	Constructor
*	Destructor
*/
Image::Image( char* ImagePath )
{
	Img = imread(ImagePath, IMREAD_GRAYSCALE);
	
	if (Img.cols == 0 || Img.rows == 0)
	{
		throw std::invalid_argument("Invalid input image. Please select valid one.");
	}
}

Image::~Image()
{
}

/*	----------------------------------------------------------
*	Function name:	img2array
*	Parameters:		imageArray <int*>
*	Used to:		Transform OpenCV data type to typical 1D array of ints representing all of pixels.
*	Return:			None.
*/
void Image::img2array( int* imageArray )
{
	int Row = 0;
	int Col = 0;
	int idx = 0;
	while (Row < Img.rows)
	{
		// From: https://docs.opencv.org/3.1.0/d3/d63/classcv_1_1Mat.html
		imageArray[idx++] = *(Img.data + Img.step[0] * Row + Img.step[1] * Col++);

		if (Col == Img.cols)
		{
			Col = 0;
			Row++;
		}
	}
}

/*	----------------------------------------------------------
*	Function name:	ShowInputImage
*	Parameters:		WindowName <char*>
*	Used to:		Show loaded image inside window. Press any key to close it.
*	Return:			None.
*/
void Image::ShowInputImage( char* WindowName )
{
	namedWindow(WindowName, WINDOW_NORMAL);
	imshow(WindowName, Img);
	waitKey(0);
	destroyWindow( WindowName );
}

/*	----------------------------------------------------------
*	Function name:	GetArraySize
*	Parameters:		None.
*	Used to:		Return total number of pixels in image.
*	Return:			Pixel total number in uint.
*/
unsigned int Image::GetArraySize()
{
	return Img.rows*Img.cols;
}