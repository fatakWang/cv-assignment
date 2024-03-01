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
    
    VideoCapture cap;
    cap.open(0);

    if (!cap.isOpened())
    {
        cerr << "Failed to open video file." << endl;
        return -1;
    }

    Mat  frame;
    int fps = cap.get(CAP_PROP_FPS);
    Size S = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
        (int)cap.get(CAP_PROP_FRAME_HEIGHT));
    VideoWriter writer("D:\\study\\CV\\assignment3\\play.mp4", CAP_OPENCV_MJPEG, fps, S, true);

    cv::namedWindow("Object Tracker");
    cap >> frame; // 读取一帧
    cv::Rect track_window = cv::selectROI("Object Tracker", frame);

    Mat roi = frame(track_window);
    Mat hsv_roi,mask;
    cvtColor(roi, hsv_roi, COLOR_BGR2HSV);
    inRange(hsv_roi, Scalar(0, 60, 32), Scalar(180, 255, 255), mask);
    float range_[] = { 0, 180 };
    const float* range[] = { range_ };
    Mat roi_hist;
    int histSize[] = { 180 };
    int channels[] = { 0,1,2 };
    calcHist(&hsv_roi, 1, channels, mask, roi_hist, 1, histSize, range);
    normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX);


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
        calcBackProject(&hsv, 1, channels, roi_hist, dst, range);
        RotatedRect rot_rect = CamShift(dst, track_window, term_crit);

        Point2f points[4];
        rot_rect.points(points);
        for (int i = 0; i < 4; i++)
            line(frame, points[i], points[(i + 1) % 4], 255, 2);

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
