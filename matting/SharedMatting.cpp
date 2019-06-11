#include "SharedMatting.h"

SharedMatting::SharedMatting() {
}

SharedMatting::~SharedMatting() {
}

void SharedMatting::LoadImage(char *filename) {
	image = imread(filename);
	// transpose(image, image);
	// flip(image, image, 0);

	height = image.rows;
	width = image.cols;
	cvtColor(image, image_gray, CV_BGR2GRAY);
}

void SharedMatting::LoadMask(char *filename) {
	mask = imread(filename);
}

void SharedMatting::SolveAlpha(char *filename) {
	clock_t start, finish;
	
	start = clock();
	cout << "GetTrimap...";
	GetTrimap();
	cout << "    over!!!" << endl;
	finish = clock();
	cout <<  double(finish - start) / CLOCKS_PER_SEC << endl;

	imwrite(filename, trimap);

	start = clock();
	ExpandKnown();
	SampleGathering();
	SampleRefine();
	LocalSmooth();
	finish = clock();
	cout <<  double(finish - start) / CLOCKS_PER_SEC << endl;

	// start = clock();
	// cout << "ExpandKnown...";
	// ExpandKnown();
	// cout << "    over!!!" << endl;
	// finish = clock();
	// cout <<  double(finish - start) / CLOCKS_PER_SEC << endl;

	// start = clock();
	// cout << "SampleGathering...";
	// SampleGathering();
	// cout << "    over!!!" << endl;
	// finish = clock();
	// cout <<  double(finish - start) / CLOCKS_PER_SEC << endl;

	// start = clock();
	// cout << "SampleRefine...";
	// SampleRefine();
	// cout << "    over!!!" << endl;
	// finish = clock();
	// cout <<  double(finish - start) / CLOCKS_PER_SEC << endl;

	// start = clock();
	// cout << "LocalSmooth...";
	// LocalSmooth();
	// cout << "    over!!!" << endl;
	// finish = clock();
	// cout <<  double(finish - start) / CLOCKS_PER_SEC << endl;
}

void SharedMatting::GetTrimap() {
	cvtColor(mask, mask, CV_BGR2GRAY);
	trimap = mask.clone();
	Mat element = getStructuringElement(MORPH_RECT, Size(39, 39)); 

	dilate(mask, trimap, element);
	erode(trimap, trimap, element);

	GaussianBlur(trimap, trimap, Size(39, 39), 19, 19);

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			if (trimap.at<uchar>(i, j) < 3)
				trimap.at<uchar>(i, j) = 0;
			else if (trimap.at<uchar>(i, j) < 252)
				trimap.at<uchar>(i, j) = 127;
			else
				trimap.at<uchar>(i, j) = 255;
		}
}

void SharedMatting::ExpandKnown() {
	struct temp {
		int x;
		int y;
		uchar tri;
	};

	vector<struct temp> vec;

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			uchar gray = trimap.at<uchar>(i, j);
			if (gray != 0 && gray != 255) {
				uchar label;
				bool flag = false;

				for (int k = 0; k <= KI && !flag; k++) {
					int i1 = max(0, i - k);
					int i2 = min(i + k, height - 1);
					int j1 = max(0, j - k);
					int j2 = min(j + k, width - 1);


					for (int ii = i1; ii <= i2 && !flag; ii++) {
						uchar gray = trimap.at<uchar>(ii, j1);
						if (gray == 0 || gray == 255) {
							double PointDis = sqrt((i - ii) * (i - ii) + (j - j1) * (j - j1));
							if (PointDis > KI)
								continue;
							Vec3d temp = image.at<Vec3b>(i, j);
							Vec3d temp1 = image.at<Vec3b>(ii, j1);
							double ColorDis = sqrt((temp - temp1).dot(temp - temp1));
							if (ColorDis <= KC) {
								flag = true;
								label = gray;
							}
						}

						gray = trimap.at<uchar>(ii, j2);
						if (gray == 0 || gray == 255) {
							double PointDis = sqrt((i - ii) * (i - ii) + (j - j2) * (j - j2));
							if (PointDis > KI)
								continue;
							Vec3d temp = image.at<Vec3b>(i, j);
							Vec3d temp1 = image.at<Vec3b>(ii, j2);
							double ColorDis = sqrt((temp - temp1).dot(temp - temp1));
							if (ColorDis <= KC) {
								flag = true;
								label = gray;
							}
						}
					}

					for (int jj = j1; jj <= j2 && !flag; jj++) {
						uchar gray = trimap.at<uchar>(i1, jj);
						if (gray == 0 || gray == 255) {
							double PointDis = sqrt((i - i1) * (i - i1) + (j - jj) * (j - jj));
							if (PointDis > KI)
								continue;
							Vec3d temp = image.at<Vec3b>(i, j);
							Vec3d temp1 = image.at<Vec3b>(i1, jj);
							double ColorDis = sqrt((temp - temp1).dot(temp - temp1));
							if (ColorDis <= KC) {
								flag = true;
								label = gray;
							}
						}

						gray = trimap.at<uchar>(i2, jj);
						if (gray == 0 || gray == 255) {
							double PointDis = sqrt((i - i2) * (i - i2) + (j - jj) * (j - jj));
							if (PointDis > KI)
								continue;
							Vec3d temp = image.at<Vec3b>(i, j);
							Vec3d temp1 = image.at<Vec3b>(i2, jj);
							double ColorDis = sqrt((temp - temp1).dot(temp - temp1));
							if (ColorDis <= KC) {
								flag = true;
								label = gray;
							}
						}
					}
				}


				if(flag) {
					struct temp p;
					p.x = i;
					p.y = j;
					p.tri = label;
					vec.push_back(p);
				}
				else {
					struct UnknownPoint tuple;
					tuple.p = cvPoint(i, j);
					TU.insert(pair<int, struct UnknownPoint>(i * width + j, tuple));
				}
			}
		}

	cout << TU.size() << "   ";

	vector<struct temp>::iterator it;
	for (it = vec.begin(); it != vec.end(); it++)
		trimap.at<uchar>(it->x, it->y) = it->tri;

	result = trimap.clone();
}

void SharedMatting::SampleGathering() {
	double inc = 360 / KG;
	map<int, struct UnknownPoint>::iterator it;

	for (it = TU.begin(); it != TU.end(); it++) {
		int i = it->second.p.x;
		int j = it->second.p.y;

		double angle = (i % 3 * 3 + j % 3) * inc / 9;
		for (int k = 0; k < KG; k++) {
			bool ff = false;
			bool bf = false;

			double _angle = (angle + k * inc) / 180 * PI;

			double step = min(1.0 / (abs(sin(_angle)) + 1e-10), 1.0 / (abs(cos(_angle)) + 1e-10));
			
			for (double s = step; !ff || !bf; s += step) {
				int ii = int(i + sin(_angle) * s + 0.5);
				int jj = int(j + cos(_angle) * s + 0.5);

				if (ii >= height || ii < 0 || jj >= width || jj < 0)
					break;

				uchar gray = trimap.at<uchar>(ii, jj);
				if (!ff && gray == 255) {
					it->second.fp.push_back(cvPoint(ii, jj));
					ff = true;
				}
				else if (!bf && gray == 0) {
					it->second.bp.push_back(cvPoint(ii, jj));
					bf = true;
				}
			}
		}
	}

	for (it = TU.begin(); it != TU.end(); it++) {
		int i = it->second.p.x;
		int j = it->second.p.y;

		double minimal = 1e10;
		double fmin = 1e10;	
		double bmin = 1e10;

		vector<CvPoint>::iterator it1;
		vector<CvPoint>::iterator it2;

		for (it1 = it->second.fp.begin(); it1 != it->second.fp.end(); it1++) {
			double dis = sqrt((i - it1->x) * (i - it1->x) + (j - it1->y) * (j - it1->y));
			double angle_i = (i - it1->x) / (dis + 1e-10);
			double angle_j = (j - it1->y) / (dis + 1e-10);

			double step = min(1.0 / (abs(angle_i) + 1e-10), 1.0 / (abs(angle_j) + 1e-10));

			double Ep = 0;
			Vec2d vec1, vec2;

			vec1 = Vec2d(i - it1->x, j - it1->y);
			vec1 = vec1 / sqrt(vec1.dot(vec1));
			for (double s = step; s < dis; s += step) {
				int ii = int(i + angle_i * s + 0.5);
				int jj = int(j + angle_j * s + 0.5);

				if (ii + 1 >= height || ii < 0 || jj + 1 >= width || jj < 0)
					break;

				vec2 = Vec2d(image_gray.at<uchar>(ii + 1, jj) - image_gray.at<uchar>(ii, jj), image_gray.at<uchar>(ii, jj + 1) - image_gray.at<uchar>(ii, jj));
				Ep += pow(vec1.dot(vec2), 2);
			}
			if (Ep < fmin) 
				fmin = Ep;
		}

		for (it2 = it->second.bp.begin(); it2 != it->second.bp.end(); it2++) {
			double dis = sqrt((i - it2->x) * (i - it2->x) + (j - it2->y) * (j - it2->y));
			double angle_i = (i - it2->x) / (dis + 1e-10);
			double angle_j = (j - it2->y) / (dis + 1e-10);

			double step = min(1.0 / (abs(angle_i) + 1e-10), 1.0 / (abs(angle_j) + 1e-10));

			double Ep = 0;
			Vec2d vec1, vec2;

			vec1 = Vec2d(i - it2->x, j - it2->y);
			vec1 = vec1 / sqrt(vec1.dot(vec1));
			for (double s = step; s < dis; s += step) {
				int ii = int(i + angle_i * s + 0.5);
				int jj = int(j + angle_j * s + 0.5);

				if (ii + 1 >= height || ii < 0 || jj + 1 >= width || jj < 0)
					break;

				vec2 = Vec2d(image_gray.at<uchar>(ii + 1, jj) - image_gray.at<uchar>(ii, jj), image_gray.at<uchar>(ii, jj + 1) - image_gray.at<uchar>(ii, jj));
				Ep += pow(vec1.dot(vec2), 2);
			}
			if (Ep < bmin)
				bmin = Ep;
		}

		double PFp = bmin / (fmin + bmin + 1e-10);

		int fi, fj, bi, bj;
		for (it1 = it->second.fp.begin(); it1 != it->second.fp.end(); it1++)
			for (it2 = it->second.bp.begin(); it2 != it->second.bp.end(); it2++) {
				Vec3d f = image.at<Vec3b>(it1->x, it1->y);
				Vec3d b = image.at<Vec3b>(it2->x, it2->y);

				double Np = 0;
				int i1 = max(0, i - 1);
				int i2 = min(i + 1, height - 1);
				int j1 = max(0, j - 1);
				int j2 = min(j + 1, width - 1);
				for (int ii = i1; ii <= i2; ii++)
					for (int jj = j1; jj <= j2; jj++) {
						Vec3d temp = image.at<Vec3b>(ii, jj);
						double alpha = min(1.0, max(0.0, (temp - b).dot(f - b) / ((f - b).dot(f - b) + 1e-10)));
						double Mp = sqrt((temp - alpha * f - (1 - alpha) * b).dot((temp - alpha * f - (1 - alpha) * b))) / 255;
						Np += Mp * Mp;
					}

				Vec3d temp = image.at<Vec3b>(i, j);
				double alpha = min(1.0, max(0.0, (temp - b).dot(f - b) / ((f - b).dot(f - b) + 1e-10)));
				double Ap = PFp + (1 - 2 * PFp) * alpha;

				double Dpf = sqrt((i - it1->x) * (i - it1->x) + (j - it1->y) * (j - it1->y));
				double Dpb = sqrt((i - it2->x) * (i - it2->x) + (j - it2->y) * (j - it2->y));

				double Gp = pow(Np, 3) * pow(Ap, 2) * Dpf * pow(Dpb, 4);
				if (Gp < minimal){
					minimal = Gp;
					fi = it1->x; fj = it1->y;
					bi = it2->x; bj = it2->y;
					it->second.fBest = f;
					it->second.bBest = b;
				}
			}

		it->second.sigmaf = it->second.sigmab = 0;

		Vec3d temp = it->second.fBest;
		int fi1 = max(0, fi - 2);
		int fi2 = min(fi + 2, height - 1);
		int fj1 = max(0, fj - 2);
		int fj2 = min(fj + 2, width - 1);

		int count = 0;

		for (int fii = fi1; fii <= fi2; fii++)
			for (int fjj = fj1; fjj <= fj2; fjj++) {
				Vec3d temp1 = image.at<Vec3b>(fii, fjj);
				it->second.sigmaf += (temp1 - temp).dot(temp1 - temp);

				count++;
			}
		it->second.sigmaf /= count;

		temp = it->second.bBest;
		int bi1 = max(0, bi - 2);
		int bi2 = min(bi + 2, height - 1);
		int bj1 = max(0, bj - 2);
		int bj2 = min(bj + 2, width - 1);

		count = 0;

		for (int bii = bi1; bii <= bi2; bii++)
			for (int bjj = bj1; bjj <= bj2; bjj++) {
				Vec3d temp1 = image.at<Vec3b>(bii, bjj);
				it->second.sigmab += (temp1 - temp).dot(temp1 - temp);

				count++;
			}
		it->second.sigmab /= count;
	}
}

void SharedMatting::SampleRefine() {
	map<int, struct UnknownPoint>::iterator it;
	for (it = TU.begin(); it != TU.end(); it++) {
		struct UnknownPoint t = it->second;

		int i = it->second.p.x;
		int j = it->second.p.y;
		Vec3d temp = image.at<Vec3b>(i, j);

		int i1 = max(0, i - 6);
		int i2 = min(i + 6, height - 1);
		int j1 = max(0, j - 6);
		int j2 = min(j + 6, width - 1);

		double minvalue[3] = {1e10f, 1e10f, 1e10f};
		CvPoint p[3];
		int count = 0;

		for (int ii = i1; ii <= i2; ii++)
			for (int jj = j1; jj <= j2; jj++) {
				uchar gray = trimap.at<uchar>(ii ,jj);
				if (gray == 0 || gray == 255)
					continue;
				map<int, struct UnknownPoint>::iterator it_temp = TU.find(ii * width + jj);
				Vec3d f = it_temp->second.fBest;
				Vec3d b = it_temp->second.bBest;
				double alpha = min(1.0, max(0.0, (temp - b).dot(f - b) / ((f - b).dot(f - b) + 1e-10)));
				double Mp = (temp - alpha * f - (1 - alpha) * b).dot((temp - alpha * f - (1 - alpha) * b));

				if (Mp > minvalue[2])
					continue;
				else if (Mp < minvalue[0]) {
					minvalue[2] = minvalue[1];
					minvalue[1] = minvalue[0];
					p[2] = p[1];
					p[1] = p[0];

					minvalue[0] = Mp;
					p[0] = cvPoint(ii, jj);
					count++;
				}
				else if (Mp < minvalue[1]) {
					minvalue[2] = minvalue[1];
					p[2] = p[1];

					minvalue[1] = Mp;
					p[1] = cvPoint(ii, jj);
					count++;
				}
				else if(Mp < minvalue[2]) {
					minvalue[2] = Mp;
					p[2] = cvPoint(ii, jj);
					count++;
				}
			}

		count = min(count, 3);

		Vec3d _f, _b;
		double _sigmaf, _sigmab;
		_f[0] = _f[1] = _f[2] = 0;
		_b[0] = _b[1] = _b[2] = 0;
		_sigmaf = _sigmab = 0;


		for (int k = 0; k < count; k++) {
			map<int, struct UnknownPoint>::iterator it_temp = TU.find(p[k].x * width + p[k].y);
			_f += it_temp->second.fBest;
			_b += it_temp->second.bBest;
			_sigmaf += it_temp->second.sigmaf;
			_sigmab += it_temp->second.sigmab;
		}

		_f /= count;
		_b /= count;
		_sigmaf /= count;
		_sigmab /= count;

		if ((temp - _f).dot(temp - _f) <= _sigmaf)
			t.fBest = temp;
		else
			t.fBest = _f;

		if ((temp - _b).dot(temp - _b) <= _sigmab)
			t.bBest = temp;
		else
			t.bBest = _b;

		if (t.fBest != t.bBest) {
			double alpha = min(1.0, max(0.0, (temp - _b).dot(_f - _b) / ((_f - _b).dot(_f - _b) + 1e-10)));
			double Mp = sqrt((temp - alpha * _f - (1 - alpha) * _b).dot((temp - alpha * _f - (1 - alpha) * _b))) / 255;
			t.confidence = exp(-10 * Mp);
		}
		else
			t.confidence = 1e-8;

		t.alpha = min(1.0, max(0.0, (temp - t.bBest).dot(t.fBest - t.bBest) / ((t.fBest - t.bBest).dot(t.fBest - t.bBest) + 1e-10)));
		TU2.insert(pair<int, struct UnknownPoint>(it->first, t));
	}
}

void SharedMatting::LocalSmooth() {
	double r = 3 * sqrt(100.0 / (9 * PI));
	map<int, struct UnknownPoint>::iterator it;
	for (it = TU2.begin(); it != TU2.end(); it++) {
		int i = it->second.p.x;
		int j = it->second.p.y;

		int i1 = max(0, i - (int)r);
		int i2 = min(i + (int)r, height - 1);
		int j1 = max(0, j - (int)r);
		int j2 = min(j + (int)r, width - 1);

		Vec3d Fp, Bp;
		double fp, bp;
		double Dfb, dfb;
		double Alp, alp;

		Fp[0] = Fp[1] = Fp[2] = 0;
		Bp[0] = Bp[1] = Bp[2] = 0;
		fp = bp = 0;
		Dfb = dfb = 0;
		Alp = alp = 0;

		for (int ii = i1; ii <= i2; ii++)
			for (int jj = j1; jj <= j2; jj++) {
				double dis = sqrt((i - ii) * (i - ii) + (j - jj) * (j - jj));
				if (dis > r)
					continue;

				double alpha, confidence;
				Vec3d f, b;

				uchar gray = trimap.at<uchar>(ii, jj);
				if (gray == 0) {
					alpha = 0;
					confidence = 1;
					f = b = image.at<Vec3b>(ii, jj);
				}
				else if (gray == 255) {
					alpha = 1;
					confidence = 1;
					f = b = image.at<Vec3b>(ii, jj);
				}
				else {
					map<int, struct UnknownPoint>::iterator it_temp = TU2.find(ii * width + jj);
					alpha = it_temp->second.alpha;
					confidence = it_temp->second.confidence;
					f = it_temp->second.fBest;
					b = it_temp->second.bBest;
				}

				double Wc;
				if (dis == 0)
					Wc = exp(-dis * dis * 9 * PI / 100.0) * confidence;
				else
					Wc = exp(-dis * dis * 9 * PI / 100.0) * confidence * abs(it->second.alpha - alpha);
			
				Fp += Wc * alpha * f;
				fp += Wc * alpha;
				Bp += Wc * (1 - alpha) * b;
				bp += Wc * (1 - alpha);

				double Wfb = confidence * alpha * (1 - alpha);
				Dfb += Wfb * sqrt((f - b).dot(f - b));
				dfb += Wfb;

				int delta;

				if (gray == 0 || gray == 255)
					delta = 1;
				else
					delta = 0;

				double Wa = confidence * exp(-dis * dis * 9 * PI /100.0) + delta;
				Alp += Wa * alpha;
				alp += Wa;
			}

		Fp = Fp / fp;
		Bp = Bp / bp;
		Dfb /= dfb;
		Alp /= alp;

		Vec3d temp = image.at<Vec3b>(i, j);
		double alpha = min(1.0, max(0.0, (temp - Bp).dot(Fp - Bp) / ((Fp - Bp).dot(Fp - Bp) + 1e-10)));
		double Mp = sqrt((temp - alpha * Fp - (1 - alpha) * Bp).dot((temp - alpha * Fp - (1 - alpha) * Bp))) / 255;

		struct FinalTuple tuple;
		tuple.p = cvPoint(i, j);
		tuple.f = Fp;
		tuple.b = Bp;
		tuple.confidence = min(1.0, sqrt((Fp - Bp).dot(Fp - Bp)) / Dfb) * exp(-10 * Mp); 
		tuple.alpha = (tuple.confidence * min(1.0, max(0.0, (temp - Bp).dot(Fp - Bp) / (Fp - Bp).dot(Fp - Bp))) + (1 - tuple.confidence) * min(1.0, max(0.0, Alp))) * 255;
		FinalTU.push_back(tuple);

		result.at<uchar>(i, j) = tuple.alpha;
	}
}

void SharedMatting::SaveMatte(char *filename) {
	imwrite(filename, result);
}

