#include "headers.h"





int main() {
	//Test_Image.png
	//lena1.png -->salt and pepper
	//image1.jpg
	//image2.jpg
	/*1.*///histogram_gray_scale("Test_Image.png");
	/*1.1*///histogram_RGB("Test_Image.png");
	/*2.*///image_equalization_gray_scale("Test_Image.png");
	/*2.1*///image_equalization_RGB("Test_Image.png");
	/*3.*///global_function_gray_scale("Test_Image.png", 2, 15);
	/*3.1*///global_function_RGB("Test_Image.png", 2, 15);
	/*4.*///arithmetic_operations("image1.jpg", "image2.jpg");
	/*5.*///convolution_gray_scale("lena1.png");
	/*5.1*/convolution_RGB("Test_Image.png");
	/*6.*///bilinear_interpolation("imagep.jpg", 5);

	
	return 0;
}
