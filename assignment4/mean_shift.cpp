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
    //打开视频文件
    VideoCapture cap;
    cap.open("D:\\study\\CV\\assignment3\\in7.mp4");

    if (!cap.isOpened())
    {
        cerr << "Failed to open video file." << endl;
        return -1;
    }

    Mat  frame;
    //输出结果
    int fps = cap.get(CAP_PROP_FPS);
    Size S = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
        (int)cap.get(CAP_PROP_FRAME_HEIGHT));
    VideoWriter writer("D:\\study\\CV\\assignment3\\out_mean_shift7.mp4", CAP_OPENCV_MJPEG, fps, S, true);

    Mat roi;
    //读入一帧来选择要追踪的物体
    Rect rect;
    cap >> frame;
    rect = selectROI("选择ROI", frame);
    roi = frame(rect);

    //转换到hsv空间
    Mat hsv_roi, mask;
    cvtColor(roi, hsv_roi, COLOR_BGR2HSV);
    //创建掩码,这个操作没意义只是为了满足calcHist的参数要求
    inRange(hsv_roi, Scalar(0, 0, 0), Scalar(255, 255, 255), mask);

    
    //计算直方图
    float range_[] = { 0, 180 };
    const float* range[] = { range_ };
    Mat roi_hist;
    int histSize[] = { 180 };
    int channels[] = { 0 };
    calcHist(&hsv_roi, 1, channels, mask, roi_hist, 1, histSize, range);

    //归一化
    normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX);


    // 设置Mean Shift算法的终止条件
    TermCriteria term_crit(TermCriteria::EPS | TermCriteria::COUNT, 10, 1);
    
    
    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            break;
        }
        
        Mat hsv, dst;
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        // 计算每一帧图像中的直方图的反投影
        calcBackProject(&hsv, 1, channels, roi_hist, dst, range);

        // 应用Mean Shift算法
        meanShift(dst, rect, term_crit);
        
        // 绘制矩形框标记目标物体
        rectangle(frame, rect, Scalar(0, 255, 0), 2);
        
        
        imshow("Video", frame);
        writer.write(frame);

        
        if (waitKey(20) == 27)
        {
            break;
        }
    }

    // 释放资源
    cap.release();
    writer.release();
    destroyAllWindows();
    return 0;


}
