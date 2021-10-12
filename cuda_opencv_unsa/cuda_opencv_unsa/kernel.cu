#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "headers.h"

__managed__ int Dev_A;
__managed__ int Dev_B;

__managed__ int Dev_size;

#define TILE_SIZE 4 
#define WINDOW_SIZE 3


__global__ void histogram_gray_sacale_CUDA(unsigned char* Image, int* Histogram) {
	int x = blockIdx.x;
	int y = blockIdx.y;

	int Image_Idx = x + y * gridDim.x;

	atomicAdd(&Histogram[Image[Image_Idx]], 1);
}


void histogram_calculation_gray_scale_CUDA(unsigned char* Image, int Height, int Width, int Channels, int* Histogram) {
	unsigned char* Dev_Image = NULL;
	int* Dev_Histogram = NULL;

	//allocate cuda variable memory
	cudaMalloc((void**)&Dev_Image, Height * Width * Channels);
	cudaMalloc((void**)&Dev_Histogram, 256 * sizeof(int));

	//copy CPU data to GPU
	cudaMemcpy(Dev_Image, Image, Height * Width * Channels, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Histogram, Histogram, 256 * sizeof(int), cudaMemcpyHostToDevice);

	dim3 Grid_Image(Width, Height);
	histogram_gray_sacale_CUDA << <Grid_Image, 1 >> > (Dev_Image, Dev_Histogram);

	//copy memory back to CPU from GPU
	cudaMemcpy(Histogram, Dev_Histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	//free up the memory of GPU
	cudaFree(Dev_Histogram);
	cudaFree(Dev_Image);
}



__global__ void histogram_RGB_CUDA(unsigned char* Image, int Channels, int* Histogram_Blue, int* Histogram_Green, int* Histogram_Red) {
	int x = blockIdx.x;
	int y = blockIdx.y;

	int Image_Idx = (x + y * gridDim.x) * Channels;

	atomicAdd(&Histogram_Blue[Image[Image_Idx]], 1);
	atomicAdd(&Histogram_Green[Image[Image_Idx + 1]], 1);
	atomicAdd(&Histogram_Red[Image[Image_Idx + 2]], 1);
}






void histogram_calculation_RGB_CUDA(unsigned char* Image, int Height, int Width, int Channels, int* Histogram_Blue, int* Histogram_Green, int* Histogram_Red)
{
	unsigned char* Dev_Image = NULL;
	int* Dev_Histogram_Blue = NULL;
	int* Dev_Histogram_Green = NULL;
	int* Dev_Histogram_Red = NULL;

	//allocate cuda variable memory
	cudaMalloc((void**)&Dev_Image, Height * Width * Channels);
	cudaMalloc((void**)&Dev_Histogram_Blue, 256 * sizeof(int));
	cudaMalloc((void**)&Dev_Histogram_Green, 256 * sizeof(int));
	cudaMalloc((void**)&Dev_Histogram_Red, 256 * sizeof(int));

	//copy CPU data to GPU
	cudaMemcpy(Dev_Image, Image, Height * Width * Channels, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Histogram_Blue, Histogram_Blue, 256 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Histogram_Green, Histogram_Green, 256 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Histogram_Red, Histogram_Red, 256 * sizeof(int), cudaMemcpyHostToDevice);

	dim3 Grid_Image(Width, Height);
	histogram_RGB_CUDA << <Grid_Image, 1 >> > (Dev_Image, Channels, Dev_Histogram_Blue, Dev_Histogram_Green, Dev_Histogram_Red);

	//copy memory back to CPU from GPU
	cudaMemcpy(Histogram_Blue, Dev_Histogram_Blue, 256 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(Histogram_Green, Dev_Histogram_Green, 256 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(Histogram_Red, Dev_Histogram_Red, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	//free up the memory of GPU
	cudaFree(Dev_Histogram_Blue);
	cudaFree(Dev_Histogram_Green);
	cudaFree(Dev_Histogram_Red);
	cudaFree(Dev_Image);

}



__global__ void equalization_CUDA(unsigned char* Image, unsigned char* Image_eq, int * hist_func) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int Image_Idx = x + y * gridDim.x;
	Image_eq[Image_Idx] = (unsigned char)hist_func[Image[Image_Idx]];
}

void image_equalization_gray_scale_CUDA(unsigned char* Image, int Height, int Width, int Channels, unsigned char* image_equalized, int * hist_function) {
	
	unsigned char* Dev_Image = NULL;
	unsigned char* Dev_Image_eq = NULL;
	int* Dev_Histogram = NULL;
	//allocate cuda variable memory
	cudaMalloc((void**)&Dev_Image, Height * Width * Channels);
	cudaMalloc((void**)&Dev_Image_eq, Height * Width * Channels);
	cudaMalloc((void**)&Dev_Histogram, 256 * sizeof(int));
	//copy CPU data to GPU
	cudaMemcpy(Dev_Image, Image, Height * Width * Channels, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Image_eq, Image, Height * Width * Channels, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Histogram, hist_function, 256 * sizeof(int), cudaMemcpyHostToDevice);
	dim3 Grid_Image(Width, Height);
	equalization_CUDA << <Grid_Image, 1 >> > (Dev_Image, Dev_Image_eq, Dev_Histogram);
	//copy memory back to CPU from GPU
	cudaMemcpy(image_equalized, Dev_Image_eq, Height * Width * Channels, cudaMemcpyDeviceToHost);
	//free up the memory of GPU
	cudaFree(Dev_Image_eq);
	cudaFree(Dev_Image);
}


__global__ void equalization_RGB_CUDA(unsigned char* Image, unsigned char* Image_eq, int Channels, int* f_Histogram_Blue, int* f_Histogram_Green, int* f_Histogram_Red) {

	int x = blockIdx.x;
	int y = blockIdx.y;

	int Image_Idx = (x + y * gridDim.x) * Channels;
	Image_eq[Image_Idx] = (unsigned char)f_Histogram_Blue[Image[Image_Idx]];
	Image_eq[Image_Idx+1] = (unsigned char)f_Histogram_Green[Image[Image_Idx+1]];
	Image_eq[Image_Idx+2] = (unsigned char)f_Histogram_Red[Image[Image_Idx+2]];
}

void image_equalization_RGB_CUDA(unsigned char* Image, int Height, int Width, int Channels, unsigned char* image_equalized, int* f_Histogram_Blue, int* f_Histogram_Green, int* f_Histogram_Red)
{
	unsigned char* Dev_Image = NULL;
	unsigned char* Dev_Image_eq = NULL;
	int* Dev_Histogram_Blue = NULL;
	int* Dev_Histogram_Green = NULL;
	int* Dev_Histogram_Red = NULL;
	//allocate cuda variable memory
	cudaMalloc((void**)&Dev_Image, Height * Width * Channels);
	cudaMalloc((void**)&Dev_Image_eq, Height * Width * Channels);
	cudaMalloc((void**)&Dev_Histogram_Blue, 256 * sizeof(int));
	cudaMalloc((void**)&Dev_Histogram_Green, 256 * sizeof(int));
	cudaMalloc((void**)&Dev_Histogram_Red, 256 * sizeof(int));
	//copy CPU data to GPU
	cudaMemcpy(Dev_Image, Image, Height * Width * Channels, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Image_eq, Image, Height * Width * Channels, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Histogram_Blue, f_Histogram_Blue, 256 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Histogram_Green, f_Histogram_Green, 256 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Histogram_Red, f_Histogram_Red, 256 * sizeof(int), cudaMemcpyHostToDevice);
	dim3 Grid_Image(Width, Height);
	equalization_RGB_CUDA << <Grid_Image, 1 >> > (Dev_Image, Dev_Image_eq, Channels, Dev_Histogram_Blue, Dev_Histogram_Green, Dev_Histogram_Red);
	//copy memory back to CPU from GPU
	cudaMemcpy(image_equalized, Dev_Image_eq, Height * Width * Channels, cudaMemcpyDeviceToHost);
	//free up the memory of GPU
	cudaFree(Dev_Image_eq);
	cudaFree(Dev_Image);
}





__global__ void global_function_CUDA(unsigned char* Image, unsigned char* Image_eq) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int Image_Idx = x + y * gridDim.x;
	int new_value = Dev_A * Image[Image_Idx] + Dev_B;
	if (new_value > 255)
		new_value = 255;
	Image_eq[Image_Idx] = new_value;
}

void global_function_gray_scale_CUDA(unsigned char* Image, int Height, int Width, int Channels, unsigned char* image_equalized, int A, int B)
{
	Dev_A = A;
	Dev_B = B;
	unsigned char* Dev_Image = NULL;
	unsigned char* Dev_Image_eq = NULL;
	//allocate cuda variable memory
	cudaMalloc((void**)&Dev_Image, Height * Width * Channels);
	cudaMalloc((void**)&Dev_Image_eq, Height * Width * Channels);
	//copy CPU data to GPU
	cudaMemcpy(Dev_Image, Image, Height * Width * Channels, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Image_eq, Image, Height * Width * Channels, cudaMemcpyHostToDevice);
	dim3 Grid_Image(Width, Height);
	global_function_CUDA << <Grid_Image, 1 >> > (Dev_Image, Dev_Image_eq);
	//copy memory back to CPU from GPU
	cudaMemcpy(image_equalized, Dev_Image_eq, Height * Width * Channels, cudaMemcpyDeviceToHost);
	//free up the memory of GPU
	cudaFree(Dev_Image_eq);
	cudaFree(Dev_Image);
}




__global__ void global_function_RGB_CUDA(unsigned char* Image, unsigned char* Image_eq, int Channels) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int Image_Idx = (x + y * gridDim.x)* Channels;
	int new_valueB = Dev_A * Image[Image_Idx] + Dev_B;
	int new_valueG = Dev_A * Image[Image_Idx+1] + Dev_B;
	int new_valueR = Dev_A * Image[Image_Idx+2] + Dev_B;

	if (new_valueB > 255)
		new_valueB = 255;
	if (new_valueG > 255)
		new_valueG = 255;
	if (new_valueR > 255)
		new_valueR = 255;
	Image_eq[Image_Idx] = new_valueB;
	Image_eq[Image_Idx+1] = new_valueG;
	Image_eq[Image_Idx+2] = new_valueR;
}

void global_function_RGB_CUDA(unsigned char* Image, int Height, int Width, int Channels, unsigned char* image_equalized, int A, int B)
{
	Dev_A = A;
	Dev_B = B;
	unsigned char* Dev_Image = NULL;
	unsigned char* Dev_Image_eq = NULL;
	//allocate cuda variable memory
	cudaMalloc((void**)&Dev_Image, Height * Width * Channels);
	cudaMalloc((void**)&Dev_Image_eq, Height * Width * Channels);
	//copy CPU data to GPU
	cudaMemcpy(Dev_Image, Image, Height * Width * Channels, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Image_eq, Image, Height * Width * Channels, cudaMemcpyHostToDevice);
	dim3 Grid_Image(Width, Height);
	global_function_RGB_CUDA << <Grid_Image, 1 >> > (Dev_Image, Dev_Image_eq, Channels);
	//copy memory back to CPU from GPU
	cudaMemcpy(image_equalized, Dev_Image_eq, Height * Width * Channels, cudaMemcpyDeviceToHost);
	//free up the memory of GPU
	cudaFree(Dev_Image_eq);
	cudaFree(Dev_Image);

}



__global__ void addition_arithmetic_operations_CUDA(unsigned char* Image1, unsigned char* Image2, unsigned char* Image_res, int Channels) {

	int x = blockIdx.x;
	int y = blockIdx.y;
	int Image_Idx = (x + y * gridDim.x) * Channels;
	int new_valueB = (Image1[Image_Idx] + Image2[Image_Idx])/2;
	int new_valueG = (Image1[Image_Idx+1] + Image2[Image_Idx+1])/2;
	int new_valueR = (Image1[Image_Idx+2] + Image2[Image_Idx+2])/2;

	Image_res[Image_Idx] = new_valueB;
	Image_res[Image_Idx+1] = new_valueG;
	Image_res[Image_Idx+2] = new_valueR;
}

__global__ void substraction_arithmetic_operations_CUDA(unsigned char* Image1, unsigned char* Image2, unsigned char* Image_res, int Channels) {

	int x = blockIdx.x;
	int y = blockIdx.y;
	int Image_Idx = (x + y * gridDim.x) * Channels;
	int new_valueB = ((Image1[Image_Idx] - Image2[Image_Idx]) / 2 ) + 128;
	int new_valueG = ((Image1[Image_Idx + 1] - Image2[Image_Idx + 1]) / 2) + 128;
	int new_valueR = ((Image1[Image_Idx + 2] - Image2[Image_Idx + 2]) / 2) + 128;

	Image_res[Image_Idx] = new_valueB;
	Image_res[Image_Idx + 1] = new_valueG;
	Image_res[Image_Idx + 2] = new_valueR;
}

__global__ void multiplication_arithmetic_operations_CUDA(unsigned char* Image1, unsigned char* Image2, unsigned char* Image_res, int Channels) {

	int x = blockIdx.x;
	int y = blockIdx.y;
	int Image_Idx = (x + y * gridDim.x) * Channels;
	int new_valueB = (Image1[Image_Idx] * Image2[Image_Idx]) / 255;
	int new_valueG = (Image1[Image_Idx + 1] * Image2[Image_Idx + 1]) / 255;
	int new_valueR = (Image1[Image_Idx + 2] * Image2[Image_Idx + 2]) / 255;

	Image_res[Image_Idx] = new_valueB;
	Image_res[Image_Idx + 1] = new_valueG;
	Image_res[Image_Idx + 2] = new_valueR;
}

__global__ void division_arithmetic_operations_CUDA(unsigned char* Image1, unsigned char* Image2, unsigned char* Image_res, int Channels) {

	int x = blockIdx.x;
	int y = blockIdx.y;
	int Image_Idx = (x + y * gridDim.x) * Channels;
	int new_valueB = (Image1[Image_Idx] / Image2[Image_Idx]) * 255;
	int new_valueG = (Image1[Image_Idx + 1] / Image2[Image_Idx + 1]) * 255;
	int new_valueR = (Image1[Image_Idx + 2] / Image2[Image_Idx + 2]) * 255;
	if (new_valueB > 255)
		new_valueB = 255;
	if (new_valueG > 255)
		new_valueG = 255;
	if (new_valueR > 255)
		new_valueR = 255;

	Image_res[Image_Idx] = new_valueB;
	Image_res[Image_Idx + 1] = new_valueG;
	Image_res[Image_Idx + 2] = new_valueR;
}

void arithmetic_operations_CUDA(unsigned char* Image1, unsigned char* Image2, int Height, int Width, int Channels, unsigned char* image_res, int type_operation)
{
	unsigned char* Dev_Image1 = NULL;
	unsigned char* Dev_Image2 = NULL;
	unsigned char* Dev_Image_res = NULL;


	//allocate cuda variable memory
	cudaMalloc((void**)&Dev_Image1, Height * Width * Channels);
	cudaMalloc((void**)&Dev_Image2, Height * Width * Channels);
	cudaMalloc((void**)&Dev_Image_res, Height * Width * Channels);


	//copy CPU data to GPU
	cudaMemcpy(Dev_Image1, Image1, Height * Width * Channels, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Image2, Image2, Height * Width * Channels, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Image_res, image_res, Height * Width * Channels, cudaMemcpyHostToDevice);


	dim3 Grid_Image(Width, Height);
	switch (type_operation)
	{
		case 1:
			addition_arithmetic_operations_CUDA << <Grid_Image, 1 >> > (Dev_Image1, Dev_Image2, Dev_Image_res, Channels);
			break;
		case 2:
			substraction_arithmetic_operations_CUDA << <Grid_Image, 1 >> > (Dev_Image1, Dev_Image2, Dev_Image_res, Channels);
			break;
		case 3:
			multiplication_arithmetic_operations_CUDA << <Grid_Image, 1 >> > (Dev_Image1, Dev_Image2, Dev_Image_res, Channels);
			break;
		case 4:
			division_arithmetic_operations_CUDA << <Grid_Image, 1 >> > (Dev_Image1, Dev_Image2, Dev_Image_res, Channels);
			break;
	default:
		break;
	}
	

	//copy memory back to CPU from GPU
	cudaMemcpy(image_res, Dev_Image_res, Height * Width * Channels, cudaMemcpyDeviceToHost);

	//free up the memory of GPU
	cudaFree(Dev_Image1);
	cudaFree(Dev_Image2);
	cudaFree(Dev_Image_res);

}


__global__ void kernel_convolution_CUDA(unsigned char* Image, unsigned char* Image_res, int Channels) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int Image_Idx = (x + y * gridDim.x)* Channels;

	int new_valueB = Dev_A * Image[Image_Idx] + Dev_B;
	int new_valueG = Dev_A * Image[Image_Idx + 1] + Dev_B;
	int new_valueR = Dev_A * Image[Image_Idx + 2] + Dev_B;

	Image_res[Image_Idx] = new_valueB;
	Image_res[Image_Idx + 1] = new_valueG;
	Image_res[Image_Idx + 2] = new_valueR;
}



__global__ void kernel_median_CUDA(unsigned char *Image, unsigned char *Image_res, int imageWidth, int imageHeight)
{
	// Set row and colum for thread.
	int Idy = blockIdx.y * blockDim.y + threadIdx.y;
	int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	int sum = 0;
	if ((Idy == 0) || (Idx == 0) || (Idy == imageHeight - 1) || (Idx == imageWidth - 1))
		Image_res[Idy*imageWidth + Idx] = 0; //Deal with boundry conditions
	else {
		for (int x = 0; x < WINDOW_SIZE; x++) {
			for (int y = 0; y < WINDOW_SIZE; y++) {
				sum+= Image[(Idy + x - 1)*imageWidth + (Idx + y - 1)];
			}
		}
		Image_res[Idy*imageWidth + Idx] = sum/9;
	}
}
 

__global__ void kernel_sobel_CUDA(unsigned char * Image, unsigned char *Image_res, const unsigned int width, const unsigned int height) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	float dx, dy;
	if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
		dx = (-1 * Image[(y - 1)*width + (x - 1)]) + (-2 * Image[y*width + (x - 1)]) + (-1 * Image[(y + 1)*width + (x - 1)]) +
			(Image[(y - 1)*width + (x + 1)]) + (2 * Image[y*width + (x + 1)]) + (Image[(y + 1)*width + (x + 1)]);
		dy = (Image[(y - 1)*width + (x - 1)]) + (2 * Image[(y - 1)*width + x]) + (Image[(y - 1)*width + (x + 1)]) +
			(-1 * Image[(y + 1)*width + (x - 1)]) + (-2 * Image[(y + 1)*width + x]) + (-1 * Image[(y + 1)*width + (x + 1)]);
		Image_res[y*width + x] = sqrt((dx*dx) + (dy*dy));
	}
}




void convolution_gray_scale_CUDA(unsigned char* Image, int Height, int Width, int Channels, unsigned char* image_res, int type_convolution)
{
	unsigned char* Dev_Image = NULL;
	unsigned char* Dev_Image_res = NULL;
	//allocate cuda variable memory
	cudaMalloc((void**)&Dev_Image, Height * Width * Channels);
	cudaMalloc((void**)&Dev_Image_res, Height * Width * Channels);
	//copy CPU data to GPU
	cudaMemcpy(Dev_Image, Image, Height * Width * Channels, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Image_res, Image, Height * Width * Channels, cudaMemcpyHostToDevice);
	
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	dim3 Grid_Image((int)ceil(Width/ TILE_SIZE), (int)ceil(Height/ TILE_SIZE));
	switch (type_convolution)
	{
	case 1:
		kernel_median_CUDA << <Grid_Image, dimBlock >> > (Dev_Image, Dev_Image_res, Width, Height);
		break;
	case 2:
		kernel_sobel_CUDA << <Grid_Image, dimBlock >> > (Dev_Image, Dev_Image_res, Width, Height);
		break;
	default:
		break;
	}
	//copy memory back to CPU from GPU
	cudaMemcpy(image_res, Dev_Image_res, Height * Width * Channels, cudaMemcpyDeviceToHost);
	//free up the memory of GPU
	cudaFree(Dev_Image_res);
	cudaFree(Dev_Image);
}


__global__ void kernel_bilinear_interpolation_CUDA(unsigned char* Image, unsigned char* Image_eq) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int Image_Idx = x + y * gridDim.x;
	int Image_Idx_ori = x / Dev_size + y * gridDim.x / Dev_size;

	Image_eq[Image_Idx] = Image[Image_Idx_ori];
}


void bilinear_interpolation_CUDA(unsigned char* Image, int Height, int Width, int Channels, unsigned char* image_res, int size)
{

	Dev_size = size;
	unsigned char* Dev_Image = NULL;
	unsigned char* Dev_Image_res = NULL;
	//allocate cuda variable memory
	cudaMalloc((void**)&Dev_Image, Height * Width * Channels);
	cudaMalloc((void**)&Dev_Image_res, Height * Width * size * Channels);
	//copy CPU data to GPU
	cudaMemcpy(Dev_Image, Image, Height * Width * Channels, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Image_res, image_res, Height * Width *size * Channels, cudaMemcpyHostToDevice);
	dim3 Grid_Image(Width*size, Height*size);
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	kernel_bilinear_interpolation_CUDA << <Grid_Image, dimBlock >> > (Dev_Image, Dev_Image_res);
	//copy memory back to CPU from GPU
	cudaMemcpy(image_res, Dev_Image_res, Height * Width * Channels, cudaMemcpyDeviceToHost);
	//free up the memory of GPU
	cudaFree(Dev_Image_res);
	cudaFree(Dev_Image);

}