#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc.hpp>  
#include<iostream>  
#include<cstdio>
using namespace std;
using namespace cv;
const int seed = 42;

void LK(Mat& old_gray, Mat& next_gray, vector<Point2f>& p0, vector<Point2f>& p1, cv::Size window_size) {
    int width = next_gray.cols;
    int height = next_gray.rows;
    Mat Ix(height, width, CV_32FC1);
    Mat Iy(height, width, CV_32FC1);
    Mat It(height, width, CV_32FC1);
    //cout << 1 << endl;

    Sobel(next_gray, Ix, CV_32FC1, 1, 0);
    Sobel(next_gray, Iy, CV_32FC1, 0, 1);
    //absdiff(old_gray, next_gray, It);
    It = -next_gray + old_gray;


    int window_width = window_size.width;
    int window_height = window_size.height;
    int area = window_width * window_height;
    p1.clear();
    //对每个上一帧图片的每个跟踪点，计算其偏移值
    for (auto p : p0) {
        Mat A(area, 2, CV_32FC1);
        Mat b(area, 1, CV_32FC1);
        int px = p.x, py = p.y;
        int cnt = 0;
        for (int x = px - window_width / 2; x <= px + window_width / 2; x++) {
            for (int y = py - window_height / 2; y <= py + window_height / 2; y++) {
                //cout << cnt << endl;
                if (x >= 0 && x < width&&y>=0&&y<height) {
                    //cout << Ix.at<float>(y, x);
                    A.at<float>(cnt, 0) = Ix.at<float>(y, x);
                    A.at<float>(cnt, 1) = Iy.at<float>(y, x);
                    b.at<float>(cnt, 0) = (float)It.at<uchar>(y, x);
                }else {
                    A.at<float>(cnt, 0) = 0;
                    A.at<float>(cnt, 1) = 0;
                    b.at<float>(cnt, 0) = 0;
                }
                cnt++;
                
            }
        }
        Mat result = (A.t() * A).inv() * A.t() * b;
        float u = result.at<float>(0, 0),v= result.at<float>(1, 0);
        p1.push_back(Point2f(u*20 + px, v*20 + py));
    }

}



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
    // Create a mask image for drawing purposes
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
        LK(old_gray, frame_gray, p0, p1, Size(15, 15));
        //calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15, 15), 2, criteria);

        vector<Point2f> good_new;
        for (uint i = 0; i < p0.size(); i++)
        {
            good_new.push_back(p1[i]);
            // draw the tracks
            line(mask, p1[i], p0[i], Scalar(0, 0, 255), 2);
            circle(frame, p1[i], 5, Scalar(0, 0, 255), -1);
            
        }
        Mat img;
        add(frame, mask, img);
        imshow("Frame", img);

        // 显示视频帧
        //imshow("Video", frame);

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
