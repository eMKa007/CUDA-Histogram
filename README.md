## __GPU vs CPU - histogram computing time.__

Command line program. Main idea was to compare computing time of histogram on GPU (CUDA) and CPU. 

CUDA uses shared memory to avoid bottlenecks in atomic operations.

CPU uses simple loop through all the pixels in the image.

NOTE: This program is intended to be use with nVidia GPU devices. Be sure you are able to run CUDA C code on your machine.

Inside this repository there is example image "atol.jpg". It is pretty big ( 5160x2696px ), but it is good candidate as a testing image.

__USAGE:__
	 

	Histogram_Compare.exe <imageName.jpg> <NumberOfExecutions>
    Where: 
    	<imageName.jpg> - path to input image.
        <NumberOfExecutions> - Number of tests computing time.
        
    Tips: 
    	It is good to locate input image at the same place as *.exe file. 
        Bigger NumberOfExecutions <int> cause more computing time. Optimal amount
        is about 250-500. Max value = 1000. Can not be < 0. 
    

#### __Example Output:__    


![alt text](https://raw.githubusercontent.com/eMKa007/CUDA-Histogram-/master/example_output.PNG?token=AkttxrI88vrQJoGGaT9hev_n6FveO-ytks5csywWwA%3D%3D)

