#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc.hpp>  
#include<iostream>  
#include<cstdio>
using namespace std;
using namespace cv;
const int seed = 42;

Mat harris_matrix(Mat& img) {
    Mat gray,dst;
    //灰度值转换
    cvtColor(img, gray, COLOR_BGR2GRAY);
    //使用blocksize=2，ksize=3，k=0.04计算角点响应值
    cornerHarris(gray, dst, 2, 3, 0.04);
    return dst;
}



Mat harris_R(Mat& img) {
    Mat gray, dst;
    //彩色三通道图像转换为灰度图像
    cvtColor(img, gray, COLOR_BGR2GRAY);
    //三个参数同opencv的三个参数含义一致
    int blockSize = 2;
    int ksize = 3;
    double k = 0.04;
    //expand_pixel ，对图像边缘处的特殊处理
    int expand_p = blockSize / 2;
    //使用sobel算子,求图像在x、y处的导数
    Mat ImgSobelX, ImgSobelY;
    Sobel(gray, ImgSobelX, CV_32F, 1, 0, ksize);
    Sobel(gray, ImgSobelY, CV_32F, 0, 1, ksize);

    //在图像边缘加上一圈expand_p，对图像边缘处的特殊处理
    Mat Operate_SX = Mat(ImgSobelX.rows + 2 * expand_p, ImgSobelX.cols + 2 * expand_p, CV_32FC1, Scalar(0));
    Mat Operate_SY = Mat(ImgSobelY.rows + 2 * expand_p, ImgSobelY.cols + 2 * expand_p, CV_32FC1, Scalar(0));
    //Mat(Rect) 拿一个矩阵区域去截取Mat这个矩阵，这里是ImgSobelX填充到了Operate_SX的非边缘区域，Operate_SY同理
    Rect rect = Rect(expand_p, expand_p, ImgSobelX.cols, ImgSobelX.rows);
    ImgSobelX.copyTo(Operate_SX(rect)); 
    ImgSobelY.copyTo(Operate_SY(rect));

    Mat resultImage = Mat(ImgSobelX.rows, ImgSobelX.cols, CV_32FC1, Scalar(0));

    //遍历resultImage每一个位置计算响应值
    for (int i = expand_p; i < resultImage.rows; i++)
    {
        for (int j = expand_p; j < resultImage.cols; j++)
        {
            //对于i,j 截取Operate_SX(j - expand_p~j - expand_p+blockSize, i - expand_p~i - expand_p+blockSize)的矩形区域
            //计算响应值，Operate_SY同理
            Rect rec = Rect(j - expand_p, i - expand_p, blockSize, blockSize);
            Mat Ix = Operate_SX(rec);
            Mat Iy = Operate_SY(rec);
            //dot运算，逐位置计算乘积然后累加
            float Ixx = Ix.dot(Ix);
            float Ixy = Ix.dot(Iy);
            float Iyy = Iy.dot(Iy);
            //响应值计算
            float R = Ixx * Iyy - Ixy * Ixy - k * (Ixx + Iyy) * (Ixx + Iyy);
            resultImage.at<int>(i - expand_p, j - expand_p) = (int)R;


        }

    }
    return resultImage;
}


// Harris角点检测的非极大值抑制+阈值抑制
Mat harris_nms(Mat& dst,int WINDOW_SIZE=3,float thers=0.0001)
{
    Mat dst_nms = dst.clone();
    //响应值小于指定阈值的变为0
    for (int i = 0; i < dst_nms.rows; i++)
    {
        for (int j = 0; j < dst_nms.cols; j++)
        {
            if (dst_nms.at<float>(i, j) < thers) {
                dst_nms.at<float>(i, j) = 0;
            }
        }
    }
    

    float eps = 1e-21;
    float max_resp = dst_nms.at<float>(0, 0);
    // 遍历所有像素点
    for (int i = 0; i < dst_nms.rows; i++)
    {
        for (int j = 0; j < dst_nms.cols; j++)
        {
            // 窗口的左上角和右下角坐标
            int x1 = max(j - WINDOW_SIZE / 2, 0);
            int y1 = max(i - WINDOW_SIZE / 2, 0);
            int x2 = min(j + WINDOW_SIZE / 2, dst_nms.cols - 1);
            int y2 = min(i + WINDOW_SIZE / 2, dst_nms.rows - 1);

            // 寻找窗口内的最大值
            float max_val = dst_nms.at<float>(i, j);
            for (int y = y1; y <= y2; y++)
            {
                for (int x = x1; x <= x2; x++)
                {
                    if (dst_nms.at<float>(y, x) > max_val)
                    {
                        max_val = dst_nms.at<float>(y, x);
                    }
                }
            }

            max_resp = max(max_resp, max_val);

            // 如果该点不是局部最大值，则将其响应值设为0
            if (dst_nms.at<float>(i, j)< max_val-eps )
            {
                dst_nms.at<float>(i, j) = 0;
            }
        }
    }
    //将所有留下来的响应值置为255方便显示
    threshold(dst_nms, dst_nms, thers, 255, THRESH_BINARY);

    return dst_nms;
}

Mat harris_draw(Mat& dst, Mat& img1) {
    // 消除函数副作用
	Mat img = img1.clone();
    //遍历所有响应值，若响应值为255则显示
	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			if (dst.at<float>(i, j) == 255) {
				line(img, Point(j - 10, i), Point(j + 10, i), Scalar(0, 255, 0));
				line(img, Point(j, i - 10), Point(j, i + 10), Scalar(0, 255, 0));
			}
		}
	}
	imshow("harris角点检测", img);
    return img;
}

Mat sift_detect(Mat& img1) {
	Mat gray, img;
    //消除副作用
	img = img1.clone();
	cvtColor(img, gray, COLOR_BGR2GRAY);
    //创建sift检测
	Ptr<SIFT> sift = SIFT::create();
	vector<KeyPoint> keypoints;
    //结果存放在keypoints这个vector中
	sift->detect(gray, keypoints);
    //遍历vector，并在响应位置画出红色的圆圈
	for (int i = 0; i < keypoints.size(); i++)
	{
		circle(img, keypoints[i].pt, 5, Scalar(0, 0, 255), 2); // Draw red "o" marker
	}
	imshow("sift角点检测", img);
    return img;
}


int main()
{
    //读入图片
    Mat img = imread("D:\\study\\CV\\assignment2\\testImg.jpg");
    imshow("原图像", img);
    
    //生成角点响应矩阵
    Mat dst = harris_matrix(img);
    //对角点响应矩阵做非极大值抑制
    Mat dst_nms = harris_nms(dst,3,0.0001);
    //工具响应值画出加号
    Mat img_harris_opencv=harris_draw(dst_nms, img);


	Mat img_sift = sift_detect(img);
    
    Mat dst_m = harris_R(img);
    Mat dst_nms_m = harris_nms(dst_m, 3, 0.0001);
    Mat img_harris_m= harris_draw(dst_nms_m, img);

    imwrite("D:\\study\\CV\\assignment2\\img_harris_opencv.jpg", img_harris_opencv);
    imwrite("D:\\study\\CV\\assignment2\\img_sift.jpg", img_sift);
    imwrite("D:\\study\\CV\\assignment2\\img_harris_m.jpg", img_harris_m);
    waitKey(0);
    destroyAllWindows();
    return 0;
}
