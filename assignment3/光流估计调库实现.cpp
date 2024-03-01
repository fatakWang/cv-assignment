#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc.hpp>  
#include<iostream>  
#include<cstdio>
using namespace std;
using namespace cv;
const int seed = 42;




int main()
{
    // 加载视频
    VideoCapture cap("D:\\study\\CV\\assignment3\\testVideo.mp4");

    if (!cap.isOpened())
    {
        cerr << "Failed to open video file." << endl;
        return -1;
    }

    

    Mat old_frame, old_gray;
    vector<Point2f> p0, p1;

    // 获取第一帧并转为灰度图像
    cap >> old_frame;
    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);

    goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
    // 创建一个用来画线的模板，线段将在这个模板上画出来
    Mat mask = Mat::zeros(old_frame.size(), old_frame.type());


    // 使用Lucas-Kanade光流算法跟踪特征点并绘制线段
    while (true)
    {
        Mat frame, frame_gray;
        cap >> frame;
        if (frame.empty())
        {
            break;
        }

        // 转为灰度图像
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15, 15), 2, criteria);

        vector<Point2f> good_new;
        for (uint i = 0; i < p0.size(); i++)
        {
            // 只有状态良好的点会被接着跟踪
            if (status[i] == 1) {
                good_new.push_back(p1[i]);
                // 在模板上画出线段,在原图上将点画出来
                line(mask, p1[i], p0[i], Scalar(0, 0, 255), 2);
                circle(frame, p1[i], 5, Scalar(0, 0, 255), -1);
            }
        }
        Mat img;
        //将模板与图像相加得到有跟踪轨迹的图像
        add(frame, mask, img);
        imshow("Frame", img);

        

        // 按ESC键退出
        if (waitKey(10) == 27)
        {
            break;
        }
        old_gray = frame_gray.clone();
        p0 = good_new;

    }

    // 释放资源
    cap.release();
    destroyAllWindows();
    return 0;


}
