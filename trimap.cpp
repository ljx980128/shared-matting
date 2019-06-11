#include <opencv2/opencv.hpp>   
#include <opencv2/highgui/highgui.hpp>
#include <ctime>

using namespace std;
using namespace cv;

int main() {
	clock_t begin = clock();

	Mat img = imread("mask.jpg");
	cvtColor(img, img, CV_BGR2GRAY);

	Mat out;
	Mat element = getStructuringElement(MORPH_RECT, Size(39, 39)); 

	dilate(img, out, element);
	erode(out, out, element);

	GaussianBlur(out, out, Size(39, 39), 19, 19);

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			if (out.at<uchar>(i, j) < 3)
				out.at<uchar>(i, j) = 0;
			else if (out.at<uchar>(i, j) < 252)
				out.at<uchar>(i, j) = 127;
			else
				out.at<uchar>(i, j) = 255;
		}

	clock_t end = clock();
	cout << double(end - begin) / CLOCKS_PER_SEC << endl;

	imwrite("trimap3.png", out);
}
