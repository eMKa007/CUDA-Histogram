#pragma once

#include <exception>
#include <opencv2/opencv.hpp>
#include <Windows.h>

using namespace cv;

class Image
{
private:
	cv::Mat		Img;

public:
	Image( char* ImagePath );
	~Image();

	void			ShowInputImage( char* WindowName );
	void			img2array( int* imageArray );
	unsigned int	GetArraySize();
	void			PrintImageInfo( const char* Name );
};

