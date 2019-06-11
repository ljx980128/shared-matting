#define KI 10
#define KG 4
#define KC 5

#define PI 3.1415926f
#define MaxSize 999999

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vector_arithmetic.cuh"

#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;
using namespace cv;
using namespace cv::gpu;

struct UnknownPoint {
	int x, y;
	int fpx[KG], fpy[KG];
	int bpx[KG], bpy[KG];

	bool flag;
	float3 fBest, bBest;

	float sigmaf, sigmab;
	float alpha;
	float confidence;

	float3 f, b;
	float finalConfidence;
	float finalAlpha;
};

__device__ int GrayTotal, GrayLess;

Mat image, image_gray, trimap, indexing, result;
struct UnknownPoint TU[MaxSize];

GpuMat GpuImage, GpuImage_gray, GpuTrimap, GpuIndexing, GpuResult;
struct UnknownPoint *TU_ptr1, *TU_ptr2, *TU_ptr3;

struct MyPoint {
	int i;
	int j;
};

struct MyPoint *P;

__global__ void GrayCollect(struct MyPoint *P, PtrStepSz<uchar> Trimap);
__global__ void ExpandKnown(struct MyPoint *P, struct UnknownPoint *TU, PtrStepSz<uchar3> Image, PtrStepSz<uchar> Trimap, PtrStepSz<int> Indexing, PtrStepSz<uchar> Result);

__global__ void SampleGathering1(struct UnknownPoint *TU, PtrStepSz<uchar3> Image, PtrStepSz<uchar> Image_gray, PtrStepSz<uchar> Trimap, PtrStepSz<uchar> Result);
__global__ void SampleGathering2(struct UnknownPoint *TU, PtrStepSz<uchar3> Image, PtrStepSz<uchar> Image_gray, PtrStepSz<uchar> Trimap, PtrStepSz<uchar> Result);

__global__ void SampleRefine(struct UnknownPoint *TU_ptr1, struct UnknownPoint *TU_ptr2, PtrStepSz<uchar3> Image, PtrStepSz<uchar> Trimap, PtrStepSz<int> Indexing, PtrStepSz<uchar> Result);
__global__ void LocalSmooth(struct UnknownPoint *TU_ptr2, struct UnknownPoint *TU_ptr3, PtrStepSz<uchar3> Image, PtrStepSz<uchar> Trimap, PtrStepSz<int> Indexing, PtrStepSz<uchar> Result);

int main(void) {
	char fileAddr[64] = {0};

    for (int n = 1; n < 65; n++) {
	    sprintf(fileAddr, "input/input%d%d.jpg", n / 10, n % 10);
	    image = imread(fileAddr);

	    cvtColor(image, image_gray, CV_BGR2GRAY);

		int height = image.rows;
		int width = image.cols;
	    
	    sprintf(fileAddr, "trimap/trimap%d%d.png", n / 10, n % 10);
	    trimap = imread(fileAddr);
		cvtColor(trimap, trimap, CV_BGR2GRAY);

		indexing = Mat(height, width, CV_32SC1, Scalar(-1));

		result = trimap.clone();
		GpuImage.upload(image);
		GpuImage_gray.upload(image_gray);
		GpuTrimap.upload(trimap);
		GpuIndexing.upload(indexing);
		GpuResult.upload(result);

		cudaMalloc((void **)&TU_ptr1, MaxSize * sizeof(struct UnknownPoint));
		cudaMalloc((void **)&TU_ptr2, MaxSize * sizeof(struct UnknownPoint));
		cudaMalloc((void **)&TU_ptr3, MaxSize * sizeof(struct UnknownPoint));

		cudaMalloc((void **)&P, MaxSize * sizeof(struct MyPoint));
		int value1 = 0;
		cudaMemcpyToSymbol(GrayTotal, &value1, sizeof(int));
		GrayCollect<<<ceil(1.0 * height * width / 1024), 1024>>>(P, GpuTrimap);
		cudaMemcpyFromSymbol(&value1, GrayTotal, sizeof(int));

		int value2 = 0;
		cudaMemcpyToSymbol(GrayLess, &value2, sizeof(int));
		ExpandKnown<<<ceil(1.0 * value1 / 1024), 1024>>>(P, TU_ptr1, GpuImage, GpuTrimap, GpuIndexing, GpuResult);
		cudaMemcpyFromSymbol(&value2, GrayLess, sizeof(int));

		printf("%d %d\n", value1, value2);

		GpuTrimap = GpuResult.clone();

		SampleGathering1<<<ceil(1.0 * value2 / 512), 512>>>(TU_ptr1, GpuImage, GpuImage_gray, GpuTrimap, GpuResult);
		SampleGathering2<<<ceil(1.0 * value2 / 512), 512>>>(TU_ptr1, GpuImage, GpuImage_gray, GpuTrimap, GpuResult);
		SampleRefine<<<ceil(1.0 * value2 / 512), 512>>>(TU_ptr1, TU_ptr2, GpuImage, GpuTrimap, GpuIndexing, GpuResult);
		LocalSmooth<<<ceil(1.0 * value2 / 512), 512>>>(TU_ptr2, TU_ptr3, GpuImage, GpuTrimap, GpuIndexing, GpuResult);

		GpuResult.download(result);
		// cudaMemcpy(TU, TU_ptr3, MaxSize * sizeof(struct UnknownPoint), cudaMemcpyDeviceToHost);

	    sprintf(fileAddr, "result/result%d%d.png", n / 10, n % 10);
	    imwrite(fileAddr, result);

	    cudaFree(TU_ptr1);
	    cudaFree(TU_ptr2);
	    cudaFree(TU_ptr3);
	    cudaFree(P);
    }
	return 0;
}

__global__ void GrayCollect(struct MyPoint *P, PtrStepSz<uchar> Trimap) {
	int height = Trimap.rows;
	int width = Trimap.cols;

	int m = blockIdx.x * 1024 + threadIdx.x;
	int i = m / width;
	int j = m % width;

	if(i >= height || i < 0 || j >= width || j < 0)
		return;

	uchar gray = Trimap(i, j);

	if (gray != 0 && gray != 255) {
		int old = atomicAdd(&GrayTotal, 1);
		P[old].i = i;
		P[old].j = j;
	}
}

__global__ void ExpandKnown(struct MyPoint *P, struct UnknownPoint *TU, PtrStepSz<uchar3> Image, PtrStepSz<uchar> Trimap, PtrStepSz<int> Indexing, PtrStepSz<uchar> Result) {
	int height = Image.rows;
	int width = Image.cols;

	int m = blockIdx.x * 1024 + threadIdx.x;
	if (m >= GrayTotal)
		return;

	struct MyPoint p = P[m];

	int i = p.i;
	int j = p.j;

	bool flag = false;

	for (int k = 0; k <= KI && !flag; k++) {
		int i1 = max(0, i - k);
		int i2 = min(i + k, height - 1);
		int j1 = max(0, j - k);
		int j2 = min(j + k, width - 1);


		for (int ii = i1; ii <= i2 && !flag; ii++) {
			uchar gray = Trimap(ii, j1);
			if (gray == 0 || gray == 255) {
				float PointDis = (i - ii) * (i - ii) + (j - j1) * (j - j1);
				if (PointDis > KI * KI)
					continue;
				float3 temp = make_float3(Image(i, j));
				float3 temp1 = make_float3(Image(ii, j1));
				float ColorDis = dot(temp - temp1, temp - temp1);
				if (ColorDis <= KC * KC) {
					flag = true;
					Result(i, j) = gray;
				}
			}

			gray = Trimap(ii, j2);
			if (gray == 0 || gray == 255) {
				float PointDis = ((i - ii) * (i - ii) + (j - j2) * (j - j2));
				if (PointDis > KI * KI)
					continue;
				float3 temp = make_float3(Image(i, j));
				float3 temp1 = make_float3(Image(ii, j2));
				float ColorDis = dot(temp - temp1, temp - temp1);
				if (ColorDis <= KC * KC) {
					flag = true;
					Result(i, j) = gray;
				}
			}
		}

		for (int jj = j1; jj <= j2 && !flag; jj++) {
			uchar gray = Trimap(i1, jj);
			if (gray == 0 || gray == 255) {
				float PointDis = (i - i1) * (i - i1) + (j - jj) * (j - jj);
				if (PointDis > KI * KI)
					continue;
				float3 temp = make_float3(Image(i, j));
				float3 temp1 = make_float3(Image(i1, jj));
				float ColorDis = dot(temp - temp1, temp - temp1);
				if (ColorDis <= KC * KC) {
					flag = true;
					Result(i, j) = gray;
				}
			}

			gray = Trimap(i2, jj);
			if (gray == 0 || gray == 255) {
				float PointDis = (i - i2) * (i - i2) + (j - jj) * (j - jj);
				if (PointDis > KI * KI)
					continue;
				float3 temp = make_float3(Image(i, j));
				float3 temp1 = make_float3(Image(i2, jj));
				float ColorDis = dot(temp - temp1, temp - temp1);
				if (ColorDis <= KC * KC) {
					flag = true;
					Result(i, j) = gray;
				}
			}
		}
	}

	if (!flag) {
		int old = atomicAdd(&GrayLess, 1);
		TU[old].x = i;
		TU[old].y = j;
		Indexing(i, j) = old;
	}
}

__global__ void SampleGathering1(struct UnknownPoint *TU, PtrStepSz<uchar3> Image, PtrStepSz<uchar> Image_gray, PtrStepSz<uchar> Trimap, PtrStepSz<uchar> Result) {
	int m = 512 * blockIdx.x + threadIdx.x;

	if (m >= GrayLess)
		return;
	struct UnknownPoint UP = TU[m];

	int height = Trimap.rows;
	int width = Trimap.cols;

	int i = UP.x;
	int j = UP.y;

	float inc = 360 / KG;
	float angle = (i % 3 * 3 + j % 3) * inc / 9;

	for (int k = 0; k < KG; k++) {
		UP.fpx[k] = UP.fpy[k] = -1;
		UP.bpx[k] = UP.bpy[k] = -1;

		bool ff = false;
		bool bf = false;

		float _angle = (angle + k * inc) / 180 * PI;

		float step = fminf(1.0f / (fabsf(sinf(_angle)) + 1e-10f), 1.0f / (fabsf(cosf(_angle)) + 1e-10f));

		for (float s = step; !ff || !bf; s += step) {
			int ii = int(i + sinf(_angle) * s + 0.5f);
			int jj = int(j + cosf(_angle) * s + 0.5f);

			if(ii >= height || ii < 0 || jj >= width || jj < 0)
				break;

			uchar gray = Trimap(ii, jj);
			if (!ff && gray == 255) {
				UP.fpx[k] = ii;
				UP.fpy[k] = jj;
				ff = true;
			}
			else if(!bf && gray == 0) {
				UP.bpx[k] = ii;
				UP.bpy[k] = jj;
				bf = true;
			}
		}
	}
	TU[m] = UP;
}

__global__ void SampleGathering2(struct UnknownPoint *TU, PtrStepSz<uchar3> Image, PtrStepSz<uchar> Image_gray, PtrStepSz<uchar> Trimap, PtrStepSz<uchar> Result) {
	int m = 512 * blockIdx.x + threadIdx.x;

	if (m >= GrayLess)
		return;
	struct UnknownPoint UP = TU[m];

	int height = Trimap.rows;
	int width = Trimap.cols;

	int i = UP.x;
	int j = UP.y;

	float fmin = 1e10f;
	float bmin = 1e10f;

	for (int k = 0; k < KG; k++)
		if (UP.fpx[k] != -1 && UP.fpy[k] != -1) {
			float dis = sqrtf((i - UP.fpx[k]) * (i - UP.fpx[k]) + (j - UP.fpy[k]) * (j - UP.fpy[k]));
			float angle_i = (i - UP.fpx[k]) / (dis + 1e-10f);
			float angle_j = (j - UP.fpy[k]) / (dis + 1e-10f);

			float step = fminf(1.0f / (fabsf(angle_i) + 1e-10f), 1.0f / (fabsf(angle_j) + 1e-10f));

			float Ep = 0;
			float2 temp1, temp2;

			temp1 = make_float2(i - UP.fpx[k], j - UP.fpy[k]);
			temp1 = temp1 / sqrtf(dot(temp1, temp1));

			for (float s = step; s < dis; s += step) {
				int ii = int(i + angle_i * s + 0.5f);
				int jj = int(j + angle_j * s + 0.5f);

				if (ii + 1 >= height || ii < 0 || jj + 1 >= width || jj < 0)
					break;
				
				temp2 = make_float2((int)(Image_gray(ii + 1, jj) - Image_gray(ii, jj)), (int)(Image_gray(ii, jj + 1) - Image_gray(ii, jj)));
				Ep += powf(dot(temp1, temp2), 2);
			}
			if (Ep < fmin) 
				fmin = Ep;
		}

	for (int k = 0; k < KG; k++)
		if (UP.bpx[k] != -1 && UP.bpy[k] != -1) {
			float dis = sqrtf((i - UP.bpx[k]) * (i - UP.bpx[k]) + (j - UP.bpy[k]) * (j - UP.bpy[k]));
			float angle_i = (i - UP.bpx[k]) / (dis + 1e-10f);
			float angle_j = (j - UP.bpy[k]) / (dis + 1e-10f);

			float step = fminf(1.0f / (fabsf(angle_i) + 1e-10f), 1.0f / (fabsf(angle_j) + 1e-10f));

			float Ep = 0;
			float2 temp1, temp2;

			temp1 = make_float2(i - UP.bpx[k], j - UP.bpy[k]);
			temp1 = temp1 / sqrtf(dot(temp1, temp1));

			for (float s = step; s < dis; s += step) {
				int ii = int(i + angle_i * s + 0.5f);
				int jj = int(j + angle_j * s + 0.5f);

				if (ii + 1 >= height || ii < 0 || jj + 1 >= width || jj < 0)
					break;

				temp2 = make_float2((int)(Image_gray(ii + 1, jj) - Image_gray(ii, jj)), (int)(Image_gray(ii, jj + 1) - Image_gray(ii, jj)));
				Ep += powf(dot(temp1, temp2), 2);
			}
			if (Ep < bmin) 
				bmin = Ep;
		}

	float PFp = bmin / (fmin + bmin + 1e-10f);

	int fi, fj, bi, bj;
	float minimal = 1e10f;
	bool flag = false;

	for (int ki = 0; ki < KG; ki++) 
		if (UP.fpx[ki] != -1 && UP.fpy[ki] != -1) 
			for (int kj = 0; kj < KG; kj++) 
				if (UP.bpx[kj] != -1 && UP.bpy[kj] != -1) {
					float3 f = make_float3(Image(UP.fpx[ki], UP.fpy[ki]));
					float3 b = make_float3(Image(UP.bpx[ki], UP.bpy[ki]));

					float Np = 0;
					int i1 = max(0, i - 1);
					int i2 = min(i + 1, height - 1);
					int j1 = max(0, j - 1);
					int j2 = min(j + 1, width - 1);

					for (int ii = i1; ii <= i2; ii++)
						for (int jj = j1; jj <= j2; jj++) {
							float3 temp = make_float3(Image(ii, jj));
							float alpha = fminf(1.0f, fmaxf(0.0f, dot(temp - b, f - b) / (dot(f - b, f - b) + 1e-10f)));
 							float Mp = sqrtf(dot(temp - alpha * f - (1 - alpha) * b, temp - alpha * f - (1 - alpha) * b)) / 255;
							Np += Mp * Mp;
						}

					float3 temp = make_float3(Image(i, j));
					float alpha = fminf(1.0f, fmaxf(0.0f, dot(temp - b, f - b) / (dot(f - b, f - b) + 1e-10f)));

					float Ap = PFp + (1 - 2 * PFp) * alpha;
					float Dpf = sqrtf((i - UP.fpx[ki]) * (i - UP.fpx[ki]) + (j - UP.fpy[ki]) * (j - UP.fpy[ki]));
					float Dpb = sqrtf((i - UP.bpx[kj]) * (i - UP.bpx[kj]) + (j - UP.bpy[kj]) * (j - UP.bpy[kj]));

					float Gp = powf(Np, 3) * powf(Ap, 2) * Dpf * powf(Dpb, 4);
					if (Gp < minimal){
						minimal = Gp;
						fi = UP.fpx[ki]; fj = UP.fpy[ki];
						bi = UP.bpx[kj]; bj = UP.bpy[kj];
						// UP.fBest = f;
						// UP.bBest = b;
						flag = true;
					}
				}


	UP.flag = false;

	if (flag) {
		UP.flag = true;

		UP.fBest = make_float3(Image(fi, fj));
		UP.bBest = make_float3(Image(bi, bj));

		UP.sigmaf = UP.sigmab = 0;

		float3 temp = UP.fBest;
		int fi1 = max(0, fi - 2);
		int fi2 = min(fi + 2, height - 1);
		int fj1 = max(0, fj - 2);
		int fj2 = min(fj + 2, width - 1);

		int count = 0;

		for (int fii = fi1; fii <= fi2; fii++)
			for (int fjj = fj1; fjj <= fj2; fjj++) {
				float3 temp1 = make_float3(Image(fii, fjj));
				UP.sigmaf += dot(temp1 - temp, temp1 - temp);

				count++;
			}
		UP.sigmaf /= count;

		temp = UP.bBest;
		int bi1 = max(0, bi - 2);
		int bi2 = min(bi + 2, height - 1);
		int bj1 = max(0, bj - 2);
		int bj2 = min(bj + 2, width - 1);

		count = 0;

		for (int bii = bi1; bii <= bi2; bii++)
			for (int bjj = bj1; bjj <= bj2; bjj++) {
				float3 temp1 = make_float3(Image(bii, bjj));
				UP.sigmab += dot(temp1 - temp, temp1 - temp);

				count++;
			}
		UP.sigmab /= count;
	}
	TU[m] = UP;
}

__global__ void SampleRefine(struct UnknownPoint *TU_ptr1, struct UnknownPoint *TU_ptr2, PtrStepSz<uchar3> Image, PtrStepSz<uchar> Trimap, PtrStepSz<int> Indexing, PtrStepSz<uchar> Result) {
	int m = 512 * blockIdx.x + threadIdx.x;
	if (m >= GrayLess)
		return;
	struct UnknownPoint UP = TU_ptr1[m];

	int height = Trimap.rows;
	int width = Trimap.cols;

	int i = UP.x;
	int j = UP.y;

	float3 temp = make_float3(Image(i, j));

	//at most 200
	int i1 = max(0, i - 6);
	int i2 = min(i + 6, height - 1);
	int j1 = max(0, j - 6);
	int j2 = min(j + 6, width - 1);

	float minvalue[3] = {1e10f, 1e10f, 1e10f};
	int px[3], py[3];
	int count = 0;

	for (int ii = i1; ii <= i2; ii++)
		for (int jj = j1; jj <= j2; jj++) {
			uchar gray = Trimap(ii, jj);
			if (gray == 0 || gray == 255)
				continue;
			if (!UP.flag)
				continue;

			int index = Indexing(ii, jj);

			float3 f = TU_ptr1[index].fBest;
			float3 b = TU_ptr1[index].bBest;
			float alpha = fminf(1.0f, fmaxf(0.0f, dot(temp - b, f - b) / (dot(f - b, f - b) + 1e-10f)));
			float Mp = dot(temp - alpha * f - (1 - alpha) * b, temp - alpha * f - (1 - alpha) * b);

			if (Mp > minvalue[2])
				continue;
			else if (Mp < minvalue[0]) {
				minvalue[2] = minvalue[1];
				minvalue[1] = minvalue[0];
				px[2] = px[1];
				py[2] = py[1];
				px[1] = px[0];
				py[1] = py[0];

				minvalue[0] = Mp;
				px[0] = ii;
				py[0] = jj;
				count++;
			}
			else if (Mp < minvalue[1]) {
				minvalue[2] = minvalue[1];
				px[2] = px[1];
				py[2] = py[1];

				minvalue[1] = Mp;
				px[1] = ii;
				py[1] = jj;
				count++;
			}
			else if(Mp < minvalue[2]) {
				minvalue[2] = Mp;
				px[2] = ii;
				py[2] = jj;
				count++;
			}
		}

	count = min(count, 3);

	float3 _f, _b;
	float _sigmaf, _sigmab;
	_f.x = _f.y = _f.z = 0;
	_b.x = _b.y = _b.z = 0;
	_sigmaf = _sigmab = 0;

	for (int k = 0; k < count; k++) {
		int index = Indexing(px[k], py[k]);
		_f += TU_ptr1[index].fBest;
		_b += TU_ptr1[index].bBest;
		_sigmaf += TU_ptr1[index].sigmaf;
		_sigmab += TU_ptr1[index].sigmab;
	}

	_f /= count;
	_b /= count;
	_sigmaf /= count;
	_sigmab /= count;

	if (dot(temp - _f, temp - _f) <= _sigmaf) 
		UP.fBest = temp;
	else
		UP.fBest = _f;

	if (dot(temp - _b, temp - _b) <= _sigmab)
		UP.bBest = temp;
	else 
		UP.bBest = _b;

	if (UP.fBest != UP.bBest) {
		float alpha = fminf(1.0f, fmaxf(0.0f, dot(temp - _b, _f - _b) / (dot(_f - _b, _f - _b) + 1e-10f)));
		float Mp = sqrtf(dot(temp - alpha * _f - (1 - alpha) * _b, temp - alpha * _f - (1 - alpha) * _b)) / 255;
		UP.confidence = expf(-10 * Mp);
	}
	else
		UP.confidence = 1e-8f;

	UP.alpha = fminf(1.0f, fmaxf(0.0f, dot(temp - UP.bBest, UP.fBest - UP.bBest) / (dot(UP.fBest - UP.bBest, UP.fBest - UP.bBest) + 1e-10f)));
	TU_ptr2[m] = UP;
}

__global__ void LocalSmooth(struct UnknownPoint *TU_ptr2, struct UnknownPoint *TU_ptr3, PtrStepSz<uchar3> Image, PtrStepSz<uchar> Trimap, PtrStepSz<int> Indexing, PtrStepSz<uchar> Result) {
	int m = 512 * blockIdx.x + threadIdx.x;
	if (m >= GrayLess)
		return;
	struct UnknownPoint UP = TU_ptr2[m];

	int height = Trimap.rows;
	int width = Trimap.cols;

	int i = UP.x;
	int j = UP.y;

	float sigma2 = 100.0f / (9 * PI);
	float r = 3 * sigma2;

	int i1 = max(0, i - (int)r);
	int i2 = min(i + (int)r, height - 1);
	int j1 = max(0, j - (int)r);
	int j2 = min(j + (int)r, width - 1);

	float3 Fp, Bp;
	float fp, bp;
	float Dfb, dfb;
	float Alp, alp;

	Fp.x = Fp.y = Fp.z = 0;
	Bp.x = Bp.y = Bp.z = 0;
	fp = bp = 0;
	Dfb = dfb = 0;
	Alp = alp = 0;

	for (int ii = i1; ii <= i2; ii++)
		for (int jj = j1; jj <= j2; jj++) {
			float dis = sqrtf((i - ii) * (i - ii) + (j - jj) * (j - jj));
			if (dis > r)
				continue;

			float alpha, confidence;
			float3 f, b;

			uchar gray = Trimap(ii, jj);
			if (gray == 0) {
				alpha = 0;
				confidence = 1;
				f = b = make_float3(Image(ii, jj));
			}
			else if (gray == 255) {
				alpha = 1;
				confidence = 1;
				f = b = make_float3(Image(ii, jj));
			}
			else {
				int index = Indexing(ii, jj);
				struct UnknownPoint UP_index = TU_ptr2[index];
				
				alpha = UP_index.alpha;
				confidence = UP_index.confidence;
				f = UP_index.fBest;
				b = UP_index.bBest;		
			}

			float t = (dis == 0) ? 1 : fabsf(UP.alpha - alpha);
			float Wc = expf(-(dis * dis) / sigma2) * confidence * t;
			Fp += Wc * alpha * f;
			fp += Wc * alpha;
			Bp += Wc * (1 - alpha) * b;
			bp += Wc * (1 - alpha);

			float Wfb = confidence * alpha * (1 - alpha);
			Dfb += Wfb * sqrtf(dot(f - b, f - b));
			dfb += Wfb;

			int delta = (gray == 0 || gray == 255) ? 1 : 0;
			float Wa = confidence * expf(-(dis * dis) / sigma2) + delta;
			Alp += Wa * alpha;
			alp += Wa;
		}

	Fp /= fp;
	Bp /= bp;
	Dfb /= dfb;
	Alp /= alp;

	float3 temp = make_float3(Image(i, j));
	float alpha = fminf(1.0f, fmaxf(0.0f, dot(temp - Bp, Fp - Bp) / (dot(Fp - Bp, Fp - Bp) + 1e-10f)));
	float Mp = sqrtf(dot(temp - alpha * Fp - (1 - alpha) * Bp, temp - alpha * Fp - (1 - alpha) * Bp)) / 255;

	UP.f = Fp;
	UP.b = Bp;
	UP.finalConfidence = fminf(1.0f, sqrtf(dot(Fp - Bp, Fp - Bp)) / Dfb) * expf(-10 * Mp);
	UP.finalAlpha = UP.finalConfidence * fminf(1.0f, fmaxf(0.0f, dot(temp - Bp, Fp - Bp) / dot(Fp - Bp, Fp - Bp))) + (1 - UP.finalConfidence) * fminf(1.0f, fmaxf(0.0f, Alp));

	Result(i, j) = UP.finalAlpha * 255;
	TU_ptr3[m] = UP;
}
