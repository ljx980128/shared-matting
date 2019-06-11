#ifndef SHAREDMATTING_H
#define SHAREDMATTING_H

#define KI 10
#define KG 4
#define KC 5

#define PI 3.1415926

#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <ctime>
#include <vector>

using namespace std;
using namespace cv;

struct UnknownPoint {
	CvPoint p;
	vector<CvPoint> fp, bp;
	Vec3d fBest, bBest;
	double sigmaf, sigmab;
	double alpha;
	double confidence;
};

struct FinalTuple {
	CvPoint p;
	Vec3b f;
	Vec3b b;
	double confidence;
	uchar alpha;
};

class SharedMatting {
public:
	SharedMatting();
	~SharedMatting();

	void LoadImage(char *filename);
	void LoadMask(char *filename);

	void SolveAlpha(char *filename);

	void GetTrimap();
	void ExpandKnown();
	void SampleGathering();
	void SampleRefine();
	void LocalSmooth();

	void SaveMatte(char *filename);

private:
	Mat image;
	Mat image_gray;
	Mat mask;
	Mat trimap;
	Mat result;

	map<int, struct UnknownPoint> TU, TU2;
	vector<struct FinalTuple> FinalTU;

	int height;
	int width;
};

#endif
