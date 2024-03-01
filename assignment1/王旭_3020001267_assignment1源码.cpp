#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc.hpp>  
#include<iostream>  
using namespace std;
using namespace cv;
const int seed = 42;

Mat gaussian_noise(Mat& img) {
	Mat imgGaussianNoise(img.size(), img.type());
	Mat gaussianNoise(img.size(), img.type());

	
	RNG rng(seed);
	rng.fill(gaussianNoise, RNG::NORMAL, 0, 30); 
	cv::add(img, gaussianNoise, imgGaussianNoise); 

	imshow("高斯噪声图像", gaussianNoise);
	imshow("加上高斯噪声后的图像", imgGaussianNoise);
	return imgGaussianNoise;
}

Mat saltPepper_noise(Mat& img,int nums) {
	Mat imgSaltPepperNoise = img.clone();

	RNG rng(seed);
	for (int i = 0; i < nums; i++) {
		int x = rng.uniform(0, img.cols);
		int y = rng.uniform(0, img.rows);
		if (i % 2 == 1) {
			imgSaltPepperNoise.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
		}
		else {
			imgSaltPepperNoise.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
		}
	}
	imshow("加上椒盐噪声后的图像", imgSaltPepperNoise);
	return imgSaltPepperNoise;
}

Mat gaussian_filter(Mat& img,int kernelSize,int sigma) {
	Mat ImgGaussianFilter;
	GaussianBlur(img, ImgGaussianFilter, Size(kernelSize, kernelSize), sigma);
	imshow("高斯滤波处理后的图像", ImgGaussianFilter);
	return ImgGaussianFilter;
}

Mat median_filter(Mat& img, int kernelSize) {
	Mat ImgMedianFilter;
	medianBlur(img, ImgMedianFilter, kernelSize);
	imshow("中值滤波处理后的图像", ImgMedianFilter);
	return ImgMedianFilter;
}

Mat median_filter_manual(Mat& img, int kernelSize)
{
	// 对每个像素进行中值滤波
	int border = kernelSize / 2;
	Mat ImgMedianFilter(img.size(), img.type());
	for (int i = border; i < img.rows - border; i++) {
		for (int j = border; j < img.cols - border; j++) {
			for (int c = 0; c < 3; c++) {
				vector<uchar> values;
				for (int k = -border; k <= border; k++) {
					for (int l = -border; l <= border; l++) {
						values.push_back(img.at<Vec3b>(i + k, j + l)[c]);
					}
				}
				sort(values.begin(), values.end());
				ImgMedianFilter.at<Vec3b>(i, j)[c] = values[kernelSize * kernelSize / 2];
			}
			
		}
	}

	imshow("中值滤波处理后的图像", ImgMedianFilter);
	return ImgMedianFilter;
}

Mat gaussian_filter_manual(Mat& img, int kernelSize,double sigma) {
	Mat ImgGaussianFilter(img.size(), img.type());
	vector<vector<double>> kernel(kernelSize, vector<double>(kernelSize));
	double sum = 0;
	double pi = acos(-1);
	double step= 1.0 / kernelSize;
	for (int i = 0; i < kernelSize; i++) {
		for (int j = 0; j < kernelSize; j++) {
			double u = -0.5 + (0.5 + j) * (step);
			double v = -0.5 + (0.5 + i) * (step);
			kernel[i][j] = exp(-(u * u + v * v) / (2 * sigma * sigma))/(2*pi* sigma * sigma);
			sum += kernel[i][j];
		}
	}
	for (int i = 0; i < kernelSize; i++) {
		for (int j = 0; j < kernelSize; j++) {
			kernel[i][j] /= sum;
		}
	}

	int border = kernelSize / 2;
	
	for (int i = border; i < img.rows - border; i++) {
		for (int j = border; j < img.cols - border; j++) {
			for (int c = 0; c < 3; c++) {
				double sum = 0, weight_sum = 0;
				for (int k = -border; k <= border; k++) {
					for (int l = -border; l <= border; l++) {
						double weight = kernel[k + border][l + border];
						sum += weight * img.at<Vec3b>(i + k, j + l)[c];
						weight_sum += weight;
					}
				}
				ImgGaussianFilter.at<Vec3b>(i, j)[c] = saturate_cast<uchar>(sum / weight_sum);
			}
			
		}
	}
	imshow("高斯滤波处理后的图像", ImgGaussianFilter);
	return ImgGaussianFilter;
}

int main()
{
	//源图像  
	Mat img = imread("D:\\study\\CV\\assignment1\\testImg.jpg");
	imshow("原图像", img);
	Mat imgGaussianNoise = gaussian_noise(img);
	Mat imgSaltPepperNoise = saltPepper_noise(img, 5000);

	//Mat Img_GaussianFilter = gaussian_filter(img, 3, 1);
	//Mat ImgGaussianNoise_GaussianFilter = gaussian_filter(imgGaussianNoise,3,1);
	//Mat ImgSaltPepperNoise_GaussianFilter = gaussian_filter(imgSaltPepperNoise, 3, 1);
	//Mat ImgGaussianNoiseManual_GaussianFilter = gaussian_filter_manual(imgGaussianNoise, 3, 1);

	//imwrite("D:\\study\\CV\\assignment1\\imgGaussianNoise.jpg", imgGaussianNoise);
	//imwrite("D:\\study\\CV\\assignment1\\imgSaltPepperNoise.jpg", imgSaltPepperNoise);
	//imwrite("D:\\study\\CV\\assignment1\\Img_GaussianFilter.jpg", Img_GaussianFilter);
	//imwrite("D:\\study\\CV\\assignment1\\ImgGaussianNoise_GaussianFilter.jpg", ImgGaussianNoise_GaussianFilter);
	//imwrite("D:\\study\\CV\\assignment1\\ImgSaltPepperNoise_GaussianFilter.jpg", ImgSaltPepperNoise_GaussianFilter);
	//imwrite("D:\\study\\CV\\assignment1\\ImgGaussianNoiseManual_GaussianFilter.jpg", ImgGaussianNoiseManual_GaussianFilter);

	Mat Img_MedianFilter= median_filter(img, 5);
	Mat ImgGaussianNoise_MedianFilter = median_filter(imgGaussianNoise, 5);
	Mat ImgSaltPepperNoise_MedianFilter = median_filter(imgSaltPepperNoise, 5);
	Mat ImgSaltPepperNoise_MedianFilterManual = median_filter_manual(imgSaltPepperNoise, 5);
	
	imwrite("D:\\study\\CV\\assignment1\\Img_MedianFilter.jpg", Img_MedianFilter);
	imwrite("D:\\study\\CV\\assignment1\\ImgGaussianNoise_MedianFilter.jpg", ImgGaussianNoise_MedianFilter);
	imwrite("D:\\study\\CV\\assignment1\\ImgSaltPepperNoise_MedianFilter.jpg", ImgSaltPepperNoise_MedianFilter);
	imwrite("D:\\study\\CV\\assignment1\\ImgSaltPepperNoise_MedianFilterManual.jpg", ImgSaltPepperNoise_MedianFilterManual);
	waitKey(0);
	
    
}
