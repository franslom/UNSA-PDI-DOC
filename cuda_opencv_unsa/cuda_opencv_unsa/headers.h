#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

/*1. Histogram in Gray Scale*/
void histogram_gray_scale(string name);
int * histogram_calculation_gray_scale(string name);
void plot_histogram(int histogram[], const char* name, Scalar color);
void histogram_calculation_gray_scale_CUDA(unsigned char* Image, int Height, int Width, int Channels, int* Histogram);

/*1.1 Histogram in RGB Scale*/
void histogram_RGB(string name);
void histogram_calculation_RGB(string name, int * histogram_Blue, int* histogram_Green, int* histogram_Red);
void histogram_calculation_RGB_CUDA(unsigned char* Image, int Height, int Width, int Channels, int* Histogram_Blue, int* Histogram_Green, int* Histogram_Red);

/*2. Equalization of an image in Gray Scale*/
void compute_N_M(int * histogram, int &N, int &M);
void equalization_func(int* histogram, int NP, int* hist_function);
void image_equalization_gray_scale(string name);
void image_equalization_gray_scale_CUDA(unsigned char* Image, int Height, int Width, int Channels, unsigned char* image_equalized, int * hist_func);

/*2.2 Equalization of an image in RGB Scale*/
void image_equalization_RGB(string name);
void image_equalization_RGB_CUDA(unsigned char* Image, int Height, int Width, int Channels, unsigned char* image_equalized, int* f_Histogram_Blue, int* f_Histogram_Green, int* f_Histogram_Red);

/*3. Global function for gray scale image*/
void global_function_gray_scale(string name, int A, int B);
void global_function_gray_scale_CUDA(unsigned char* Image, int Height, int Width, int Channels, unsigned char* image_equalized, int A, int B);

/*3.1 Global function for RGB image*/
void global_function_RGB(string name, int A, int B);
void global_function_RGB_CUDA(unsigned char* Image, int Height, int Width, int Channels, unsigned char* image_equalized, int A, int B);

/*4. Arithmetic operations*/
void arithmetic_operations(string image1, string image2);
void arithmetic_operations_CUDA(unsigned char* Image1, unsigned char* Image2, int Height, int Width, int Channels, unsigned char* image_res, int type_operation);

/*5. Convolution for gray scale*/
void convolution_gray_scale(string name);
void convolution_gray_scale_CUDA(unsigned char* Image, int Height, int Width, int Channels, unsigned char* image_res, int type_convolution);

/*5.1 Convolution for RGB */
void convolution_RGB(string name);

/*6. Bilinear interpolation*/
void bilinear_interpolation(string name, int size);
void bilinear_interpolation_CUDA(unsigned char* Image, int Height, int Width, int Channels, unsigned char* image_res, int size);



