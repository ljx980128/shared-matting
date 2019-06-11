# shared-matting
matting matting_improve CPU版shared-matting抠图程序
	input文件夹存放输入图像                     格式为inputxx.jpg
	mask文件夹存放输入的0-1mask           格式为maskxx.jpg
	trimap文件夹存放输出的trimap结果     格式为trimapxx.png
	result文件夹存放输出的matting结果    格式为resultxx.png
	运行环境为Ubuntu16.04 opencv2.4.13 
	编译指令g++ -std=c++11 -O3 main.cpp SharedMatting.cpp -o SharedMatting `pkg-config opencv --cflags --libs`

matting_gpu matting_gpu_improve GPU版shared-matting抠图程序
	input文件夹存放输入图像                 格式为inputxx.jpg
	trimap文件夹存放输入trimap           格式为trimapxx.png
	result文件夹存放输出结果                格式为resultxx.png
	运行环境为Ubuntu16.04 opencv2.4.13 cuda8 需编译gpu版OpenCV
	编译指令
nvcc matting.cu -o matting `pkg-config opencv --cflags --libs`

	注：GPU版本没有实现trimap获取，但OpenCV3.0以上的版本（我使用的是OpenCV2）提供了膨胀、腐蚀、高斯模糊的GPU版，可直接调用。
	获取trimap的实现细节可参考trimap.cpp

	matting.cu中 宏定义MaxSize是与trimap中unknown区域大小相关的量 预设值为999999 可满足现有图像抠图的需求 若今后图像分辨率加大 导致抠图过程程序崩溃 可增加此参数
