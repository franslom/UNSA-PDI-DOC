#include "headers.h"


void plot_histogram(int histogram[], const char* name, Scalar color)
{
	int hist[256];
	for (int i = 0; i < 256; i++)
	{
		hist[i] = histogram[i];
	}
	// draw the histograms
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / 256);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));

	// find the maximum intensity element from histogram
	int max = hist[0];
	for (int i = 1; i < 256; i++) {
		if (max < hist[i]) {
			max = hist[i];
		}
	}
	// normalize the histogram between 0 and histImage.rows
	for (int i = 0; i < 256; i++)
	{
		hist[i] = ((double)hist[i] / max)*histImage.rows;
	}
	// draw the intensity line for histogram
	for (int i = 0; i < 256; i++)
	{
		line(histImage, Point(bin_w*(i), hist_h), Point(bin_w*(i), hist_h - hist[i]), color, 1, 8, 0);
	}
	imshow(name, histImage);
	
}

int * histogram_calculation_gray_scale(string name)
{
	Mat Input_Image = imread(name, 0);
	cout << "Image Height: " << Input_Image.rows << ", Image Width: " << Input_Image.cols << ", Image Channels: " << Input_Image.channels() << endl;

	int static Histogram_GrayScale[256] = { 0 };
	histogram_calculation_gray_scale_CUDA(Input_Image.data, Input_Image.rows, Input_Image.cols, Input_Image.channels(), Histogram_GrayScale);
	for (int i = 0; i < 256; i++) {
		cout << "Histogram_GrayScale[" << i << "]: " << Histogram_GrayScale[i] << endl;
	}
	return Histogram_GrayScale;
}


void histogram_gray_scale(string name) {
	cout << "----Histogram Gray Scale----" << endl;
	int * histogram;
	histogram = histogram_calculation_gray_scale(name);
	Mat Input_Image = imread(name, 0);
	imshow("Input Image", Input_Image);
	plot_histogram(histogram, "Histogram Gray Scale", Scalar(0,0,0));
	waitKey(0);
}

void histogram_calculation_RGB(string name, int * histogram_Blue, int* histogram_Green, int* histogram_Red)
{
	Mat Input_Image = imread(name);
	cout << "Image Height: " << Input_Image.rows << ", Image Width: " << Input_Image.cols << ", Image Channels: " << Input_Image.channels() << endl;
	histogram_calculation_RGB_CUDA(Input_Image.data, Input_Image.rows, Input_Image.cols, Input_Image.channels(), histogram_Blue, histogram_Green, histogram_Red);
	
}

void histogram_RGB(string name)
{
	cout << "----Histogram RGB----" << endl;
	int histogram_Blue[256] = { 0 };
	int histogram_Green[256] = { 0 };
	int histogram_Red[256] = { 0 };
	histogram_calculation_RGB(name, histogram_Blue, histogram_Green, histogram_Red);
	for (int i = 0; i < 256; i++) {
		cout << "Histogram_Blue[" << i << "]: " << histogram_Blue[i] << endl;
		cout << "Histogram_Green[" << i << "]: " << histogram_Green[i] << endl;
		cout << "Histogram_Red[" << i << "]: " << histogram_Red[i] << endl;
	}
	Mat Input_Image = imread(name);
	imshow("Input Image", Input_Image);
	plot_histogram(histogram_Blue, "Histogram Blue", Scalar(255, 0, 0));
	plot_histogram(histogram_Green, "Histogram Green", Scalar(0, 255, 0));
	plot_histogram(histogram_Red, "Histogram Red", Scalar(0, 0, 255));
	waitKey(0);
}


void compute_N_M(int * histogram, int &N, int &M)
{
	N = -1;
	M = -1;
	for (size_t i = 0; i < 256; i++)
	{
		if (histogram[i] > 0)
		{
			N = i;
			break;
		}
	}
	for (size_t i = 255; i >= 0; i--)
	{
		if (histogram[i] > 0)
		{
			M = i;
			break;
		}
	}
}



void equalization_func(int* histogram, int NP, int* hist_function)
{
	int acum = 0;
	for (int i = 0; i < 256; i++)
	{
		acum += histogram[i];
		hist_function[i] = (acum * 1.0 / NP) * 255;
	}
}

void image_equalization_gray_scale(string name)
{
	Mat Input_Image = imread(name, 0);
	cout << "Image equalization gray scale" << endl;
	cout << "Image Height: " << Input_Image.rows << ", Image Width: " << Input_Image.cols << ", Image Channels: " << Input_Image.channels() << endl;
	Mat Input_Image_eq = Input_Image.clone();

	int * histogram;
	int * histogram_func;
	histogram = histogram_calculation_gray_scale(name);
	histogram_func = histogram;
	equalization_func(histogram, Input_Image.rows*Input_Image.cols, histogram_func);

	image_equalization_gray_scale_CUDA(Input_Image.data, Input_Image.rows, Input_Image.cols, Input_Image.channels(), Input_Image_eq.data, histogram_func);
	
	imshow("Input", Input_Image);
	imshow("Output", Input_Image_eq);
	waitKey(0);
}



void image_equalization_RGB(string name)
{
	Mat Input_Image = imread(name);
	cout << "Image equalization RGB" << endl;
	cout << "Image Height: " << Input_Image.rows << ", Image Width: " << Input_Image.cols << ", Image Channels: " << Input_Image.channels() << endl;
	Mat Input_Image_eq = Input_Image.clone();
	
	int histogram_Blue[256] = { 0 };
	int f_histogram_Blue[256] = { 0 };
	int histogram_Green[256] = { 0 };
	int f_histogram_Green[256] = { 0 };
	int histogram_Red[256] = { 0 };
	int f_histogram_Red[256] = { 0 };

	histogram_calculation_RGB(name, histogram_Blue, histogram_Green, histogram_Red);
	//Compute histogram function for Red
	equalization_func(histogram_Red, Input_Image.rows*Input_Image.cols, f_histogram_Red);
	//Compute histogram function for Green
	equalization_func(histogram_Green, Input_Image.rows*Input_Image.cols, f_histogram_Green);
	//Compute histogram function for Blue
	equalization_func(histogram_Blue, Input_Image.rows*Input_Image.cols, f_histogram_Blue);

	image_equalization_RGB_CUDA(Input_Image.data, Input_Image.rows, Input_Image.cols, Input_Image.channels(), Input_Image_eq.data, f_histogram_Blue, f_histogram_Green, f_histogram_Red);

	imshow("Input", Input_Image);
	imshow("Output", Input_Image_eq);
	waitKey(0);
	
	
}






void global_function_gray_scale(string name, int A, int B)
{
	cout << "Global function for gray scale image" << endl;
	cout << A << "--" << B << endl;

	Mat Input_Image = imread(name, 0);
	cout << "Image Height: " << Input_Image.rows << ", Image Width: " << Input_Image.cols << ", Image Channels: " << Input_Image.channels() << endl;
	Mat Input_Image_eq = Input_Image.clone();
	global_function_gray_scale_CUDA(Input_Image.data, Input_Image.rows, Input_Image.cols, Input_Image.channels(), Input_Image_eq.data, A, B);

	imshow("Input", Input_Image);
	imshow("Output", Input_Image_eq);
	waitKey(0);

}

void global_function_RGB(string name, int A, int B)
{
	cout << "Global function for RGB image" << endl;
	cout << A << "--" << B << endl;

	Mat Input_Image = imread(name);
	cout << "Image Height: " << Input_Image.rows << ", Image Width: " << Input_Image.cols << ", Image Channels: " << Input_Image.channels() << endl;
	Mat Input_Image_eq = Input_Image.clone();
	global_function_RGB_CUDA(Input_Image.data, Input_Image.rows, Input_Image.cols, Input_Image.channels(), Input_Image_eq.data, A, B);

	imshow("Input", Input_Image);
	imshow("Output", Input_Image_eq);
	waitKey(0);

}


void arithmetic_operations(string image1, string image2)
{
	int type_operation;
	cout << "Arithmetic operations for an image" << endl;
	
	cout << "1. Addition" << endl;
	cout << "2. Subtraction" << endl;
	cout << "3. Multiplication" << endl;
	cout << "4. Division" << endl;
	cout << "Select one operation: ";
	cin >> type_operation;
	Mat Input_Image1 = imread(image1);
	Mat Input_Image2 = imread(image2);
	Mat Input_Image_res = Input_Image1.clone();

	arithmetic_operations_CUDA(Input_Image1.data, Input_Image2.data, Input_Image1.rows, Input_Image1.cols, Input_Image1.channels(), Input_Image_res.data, type_operation);

	imshow("Input 1", Input_Image1);
	imshow("Input 2", Input_Image2);
	imshow("Output", Input_Image_res);
	waitKey(0);
}

void convolution_gray_scale(string name)
{
	int type_convolution;
	cout << "Convolution gray scale image" << endl;
	cout << "1. Median filter" << endl;
	cout << "2. Sobel filter" << endl;
	cout << "Select one filter: ";
	cin >> type_convolution;

	Mat Input_Image = imread(name,0);
	cout << "Image Height: " << Input_Image.rows << ", Image Width: " << Input_Image.cols << ", Image Channels: " << Input_Image.channels() << endl;
	Mat Input_Image_res = Input_Image.clone();
	convolution_gray_scale_CUDA(Input_Image.data, Input_Image.rows, Input_Image.cols, Input_Image.channels(), Input_Image_res.data, type_convolution);

	imshow("Input", Input_Image);
	imshow("Output", Input_Image_res);
	waitKey(0);
}


void convolution_RGB(string name)
{
	int type_convolution;
	cout << "Convolution RGB image" << endl;
	cout << "1. Median filter" << endl;
	cout << "2. Sobel filter" << endl;
	cout << "Select one filter: ";
	cin >> type_convolution;

	Mat Input_Image = imread(name);
	cout << "Image Height: " << Input_Image.rows << ", Image Width: " << Input_Image.cols << ", Image Channels: " << Input_Image.channels() << endl;
	Mat Input_Image_res;

	Mat image_three_channels[3];
	split(Input_Image, image_three_channels);
	vector<Mat> image_three_channels_res;
	image_three_channels_res.push_back(image_three_channels[0].clone());
	image_three_channels_res.push_back(image_three_channels[1].clone());
	image_three_channels_res.push_back(image_three_channels[2].clone());


	convolution_gray_scale_CUDA(image_three_channels[0].data, Input_Image.rows, Input_Image.cols, 1, image_three_channels_res[0].data, type_convolution);
	convolution_gray_scale_CUDA(image_three_channels[1].data, Input_Image.rows, Input_Image.cols, 1, image_three_channels_res[1].data, type_convolution);
	convolution_gray_scale_CUDA(image_three_channels[2].data, Input_Image.rows, Input_Image.cols, 1, image_three_channels_res[2].data, type_convolution);
	merge(image_three_channels_res, Input_Image_res);
	imshow("Input", Input_Image);
	imshow("Output", Input_Image_res);


	waitKey(0);
}




void bilinear_interpolation(string name, int size)
{
	cout << "Bilinear interpolation for gray scale image" << endl;

	Mat Input_Image = imread(name, 0);
	cout << "Image Height: " << Input_Image.rows << ", Image Width: " << Input_Image.cols << ", Image Channels: " << Input_Image.channels() << endl;
	Mat Input_Image_res =  Mat::zeros(Size(Input_Image.cols*size, Input_Image.rows*size), CV_8UC1);
	bilinear_interpolation_CUDA(Input_Image.data, Input_Image.rows, Input_Image.cols, Input_Image.channels(), Input_Image_res.data, size);
	imshow("Input", Input_Image);
	imshow("Output", Input_Image_res);
	waitKey(0);
}