#include "opencv2/opencv.hpp"
#include "opencv2/ml.hpp"

#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <cctype>
#include<iostream>
#include <fstream>

using namespace cv;
using namespace cv::ml;
using namespace std;

#define FILEPATH  "D:/Pedestrians64x128/"

int Traain_Flag = 0;
int Distinguish_Flag = 1;

int Predect_num(Mat test_image, Ptr<ANN_MLP> bp);
///////////////////////////////////HOG+SVM识别方式2///////////////////////////////////////////////////	
void Train()
{
	////////////////////////////////读入训练样本图片路径和类别///////////////////////////////////////////////////
	//图像路径和类别
	vector<string> imagePath;
	vector<int> imageClass;
	int numberOfLine = 0;
	string buffer;
	ifstream trainingData(string(FILEPATH) + "TrainData.txt");
	unsigned long n;

	while (!trainingData.eof())
	{
		getline(trainingData, buffer);
		if (!buffer.empty())
		{
			++numberOfLine;
			if (numberOfLine % 2 == 0)
			{
				//读取样本类别
				imageClass.push_back(atoi(buffer.c_str()));
			}
			else
			{
				//读取图像路径
				imagePath.push_back(buffer);
			}
		}
	}

	//关闭文件  
	trainingData.close();


	////////////////////////////////获取样本的HOG特征///////////////////////////////////////////////////
	//样本特征向量矩阵
	int numberOfSample = numberOfLine / 2;
	Mat featureVectorOfSample(numberOfSample, 3780, CV_32FC1);//矩阵中每行为一个样本

	//样本的类别
	Mat classOfSample(numberOfSample, 1, CV_32SC1);

	Mat convertedImage;
	Mat trainImage;

	// 计算HOG特征
	for (vector<string>::size_type i = 0; i <= imagePath.size() - 1; ++i)
	{
		//读入图片
		Mat src = imread(imagePath[i], -1);
		if (src.empty())
		{
			cout << "can not load the image:" << imagePath[i] << endl;
			continue;
		}
		cout << "processing:" << imagePath[i] << endl;

		// 归一化
		resize(src, trainImage, Size(64, 128));

		// 提取HOG特征
		HOGDescriptor hog(cvSize(64, 128), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
		vector<float> descriptors;
		double time1 = getTickCount();
		hog.compute(trainImage, descriptors);//这里可以设置检测窗口步长，如果图片大小超过64×128，可以设置winStride
		double time2 = getTickCount();
		double elapse_ms = (time2 - time1) * 1000 / getTickFrequency();
		cout << "HOG dimensions:" << descriptors.size() << endl;
		//cout << "Compute time:" << elapse_ms << endl;


		//保存到特征向量矩阵中
		for (vector<float>::size_type j = 0; j <= descriptors.size() - 1; ++j)
		{
			featureVectorOfSample.at<float>(i, j) = descriptors[j];
		}

		//保存类别到类别矩阵
		//!!注意类别类型一定要是int 类型的
		classOfSample.at<int>(i, 0) = imageClass[i];
	}


	///////////////////////////////////使用SVM分类器训练///////////////////////////////////////////////////	
	//设置参数，注意Ptr的使用
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);//注意必须使用线性SVM进行训练，因为HogDescriptor检测函数只支持线性检测！！！
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON));

	//使用SVM学习         
	svm->train(featureVectorOfSample, ROW_SAMPLE, classOfSample);

	//保存分类器(里面包括了SVM的参数，支持向量,α和rho)
	svm->save(string(FILEPATH) + "Classifier.xml");

	/*
	SVM训练完成后得到的XML文件里面，有一个数组，叫做support vector，还有一个数组，叫做alpha,有一个浮点数，叫做rho;
	将alpha矩阵同support vector相乘，注意，alpha*supportVector,将得到一个行向量，将该向量前面乘以-1。之后，再该行向量的最后添加一个元素rho。
	如此，变得到了一个分类器，利用该分类器，直接替换opencv中行人检测默认的那个分类器（cv::HOGDescriptor::setSVMDetector()），
	*/
	//获取支持向量机：矩阵默认是CV_32F
	Mat supportVector = svm->getSupportVectors();//

	//获取alpha和rho
	Mat alpha;//每个支持向量对应的参数α(拉格朗日乘子)，默认alpha是float64的
	Mat svIndex;//支持向量所在的索引
	float rho = svm->getDecisionFunction(0, alpha, svIndex);

	//转换类型:这里一定要注意，需要转换为32的
	Mat alpha2;
	alpha.convertTo(alpha2, CV_32FC1);

	//结果矩阵，两个矩阵相乘
	Mat result(1, 3780, CV_32FC1);
	result = alpha2*supportVector;

	//乘以-1，这里为什么会乘以-1？
	//注意因为svm.predict使用的是alpha*sv*another-rho，如果为负的话则认为是正样本，在HOG的检测函数中，使用rho+alpha*sv*another(another为-1)
	for (int i = 0; i < 3780; ++i)
		result.at<float>(0, i) *= -1;

	//将分类器保存到文件，便于HOG识别
	//这个才是真正的判别函数的参数(ω)，HOG可以直接使用该参数进行识别
	FILE *fp = fopen((string(FILEPATH) + "HOG_SVM.txt").c_str(), "wb");
	for (int i = 0; i<3780; i++)
	{
		fprintf(fp, "%f \n", result.at<float>(0, i));
	}
	fprintf(fp, "%f", rho);

	fclose(fp);

}
// 使用训练好的分类器识别
void Detect()
{
	Mat img;
	FILE* f = 0;
	char _filename[1024];

	// 获取测试图片文件路径
	f = fopen((string(FILEPATH) + "TestData.txt").c_str(), "rt");
	if (!f)
	{
		fprintf(stderr, "ERROR: the specified file could not be loaded\n");
		return;
	}


	//加载训练好的判别函数的参数(注意，与svm->save保存的分类器不同)
	vector<float> detector;
	ifstream fileIn(string(FILEPATH) + "HOG_SVM.txt", ios::in);
	float val = 0.0f;
	while (!fileIn.eof())
	{
		fileIn >> val;
		detector.push_back(val);
	}
	fileIn.close();

	//设置HOG
	HOGDescriptor hog;
	hog.setSVMDetector(detector);// 使用自己训练的分类器
	//hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//可以直接使用05 CVPR已训练好的分类器,这样就不用Train()这个步骤了
	namedWindow("people detector", 1);

	// 检测图片
	for (;;)
	{/*
		// 读取文件名
		char* filename = _filename;
		if (f)
		{
			if (!fgets(filename, (int)sizeof(_filename)-2, f))
				break;
			//while(*filename && isspace(*filename))
			//  ++filename;
			if (filename[0] == '#')
				continue;

			//去掉空格
			int l = (int)strlen(filename);
			while (l > 0 && isspace(filename[l - 1]))
				--l;
			filename[l] = '\0';
			img = imread(filename);
		}
		printf("%s:\n", filename);
		*/
		img = imread("test4.jpg");
		if (!img.data)
			continue;

		fflush(stdout);
		vector<Rect> found, found_filtered;
		double t = (double)getTickCount();
		// run the detector with default parameters. to get a higher hit-rate
		// (and more false alarms, respectively), decrease the hitThreshold and
		// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
		//多尺度检测
		hog.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
		t = (double)getTickCount() - t;
		printf("detection time = %gms\n", t*1000. / cv::getTickFrequency());
		size_t i, j;

		//去掉空间中具有内外包含关系的区域，保留大的
		for (i = 0; i < found.size(); i++)
		{
			Rect r = found[i];
			for (j = 0; j < found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
			if (j == found.size())
				found_filtered.push_back(r);
		}

		// 适当缩小矩形
		for (i = 0; i < found_filtered.size(); i++)
		{
			Rect r = found_filtered[i];
			// the HOG detector returns slightly larger rectangles than the real objects.
			// so we slightly shrink the rectangles to get a nicer output.
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);
			rectangle(img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
		}

		imshow("people detector", img);
		int c = waitKey(0) & 255;
		if (c == 'q' || c == 'Q' || !f)
			break;
	}
	if (f)
		fclose(f);
	return;
}

void HOG_SVM2()
{
	//如果使用05 CVPR提供的默认分类器，则不需要Train(),直接使用Detect检测图片
	Train();
	Detect();
}

#define PATH "E:/opencv/opencv3.0/MyProject/res/BigBossPic/"
///////////////////////////////////HOG+SVM识别方式1///////////////////////////////////////////////////
void HOG_SVM1()
{
	////////////////////////////////读入训练样本图片路径和类别///////////////////////////////////////////////////
	//图像路径和类别
	vector<string> imagePath;
	vector<int> imageClass;
	int numberOfLine = 0;
	string buffer;
	ifstream trainingData(string(PATH) + "TrainData.txt", ios::in);
	unsigned long n;

	

	while (!trainingData.eof())
	{
		getline(trainingData, buffer);
		if (!buffer.empty())
		{
			++numberOfLine;
			if (numberOfLine % 2 == 0)
			{
				//读取样本类别
				imageClass.push_back(atoi(buffer.c_str()));
			}
			else
			{
				//读取图像路径
				imagePath.push_back(buffer);
			}
		}
	}
	trainingData.close();


	////////////////////////////////获取样本的HOG特征///////////////////////////////////////////////////
	//样本特征向量矩阵
	int numberOfSample = numberOfLine / 2;
	Mat featureVectorOfSample(numberOfSample, 324, CV_32FC1);//矩阵中每行为一个样本

	//样本的类别
	Mat classOfSample(numberOfSample, 1, CV_32SC1);

	//开始计算训练样本的HOG特征
	for (vector<string>::size_type i = 0; i <= imagePath.size() - 1; ++i)
	{
		//读入图片
		Mat src = imread(imagePath[i], -1);
		if (src.empty())
		{
			cout << "can not load the image:" << imagePath[i] << endl;
			continue;
		}
		cout << "processing" << imagePath[i] << endl;

		//缩放
		Mat trainImage;
		resize(src, trainImage, Size(28, 28));

		//提取HOG特征
	//	HOGDescriptor hog(Size(128, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
		HOGDescriptor hog(Size(28, 28), Size(14, 14), Size(7, 7), Size(7, 7), 9);
		vector<float> descriptors;
		hog.compute(trainImage, descriptors);//这里可以设置检测窗口步长，如果图片大小超过64×128，可以设置winStride
		//hog.compute(trainImage, descriptors);//这里可以设置检测窗口步长，如果图片大小超过64×128，可以设置winStride
		cout << "HOG dimensions:" << descriptors.size() << endl;

		//保存特征向量矩阵中
		for (vector<float>::size_type j = 0; j <= descriptors.size() - 1; ++j)
		{
			featureVectorOfSample.at<float>(i, j) = descriptors[j];
		}

		//保存类别到类别矩阵
		//!!注意类别类型一定要是int 类型的
		classOfSample.at<int>(i, 0) = imageClass[i];
	}


	///////////////////////////////////使用SVM分类器训练///////////////////////////////////////////////////	
	//设置参数
	//参考3.0的Demo
	Ptr<SVM> svm = SVM::create();
	/*
	svm->setKernel(SVM::RBF);
	svm->setType(SVM::C_SVC);
	svm->setC(10);
	svm->setCoef0(1.0);
	svm->setP(1.0);
	svm->setNu(0.5);
	*/
	svm->setType(SVM::C_SVC);
	svm->setGamma(0.01);
	svm->setDegree(2);
	svm->setC(10);
	svm->setKernel(SVM::POLY);
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON));

	//使用SVM学习         
	svm->train(featureVectorOfSample, ROW_SAMPLE, classOfSample);

	//保存分类器
	svm->save("Classifier.xml");
	cout << "Train Done!" << endl;

	
	
	/*
	///////////////////////////////////使用训练好的分类器进行识别///////////////////////////////////////////////////
	vector<string> testImagePath;
	ifstream testData(string(PATH) + "TestData.txt", ios::out);
	while (!testData.eof())
	{
		getline(testData, buffer);
		//读取
		if (!buffer.empty())
			testImagePath.push_back(buffer);

	}
	testData.close();

	ofstream fileOfPredictResult(string(PATH) + "PredictResult.txt"); //最后识别的结果
	for (vector<string>::size_type i = 0; i <= testImagePath.size() - 1; ++i)
	{
		//读取测试图片
		Mat src = imread(testImagePath[i], -1);
		if (src.empty())
		{
			cout << "Can not load the image:" << testImagePath[i] << endl;
			continue;
		}

		//缩放
		Mat testImage;
		resize(src, testImage, Size(128, 128));

		//测试图片提取HOG特征
		HOGDescriptor hog(cvSize(128, 128), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
		vector<float> descriptors;
		hog.compute(testImage, descriptors);
		cout << "HOG dimensions:" << descriptors.size() << endl;

		Mat featureVectorOfTestImage(1, descriptors.size(), CV_32FC1);
		for (int j = 0; j <= descriptors.size() - 1; ++j)
		{
			featureVectorOfTestImage.at<float>(0, j) = descriptors[j];
		}

		//对测试图片进行分类并写入文件
		int predictResult = svm->predict(featureVectorOfTestImage);
		char line[512];
		printf("%s %d\r\n", testImagePath[i].c_str(), predictResult);
		std::sprintf(line, "%s %d\n", testImagePath[i].c_str(), predictResult);
		fileOfPredictResult << line;

	}
	fileOfPredictResult.close();


	*/

	
}
int Num_Predect(void)
{

	Ptr<SVM> svm = StatModel::load<SVM>("Classifier.xml");
//	Ptr<SVM> svm = SVM::create();
	// svm->load("Classifier.xml");
	//svm = SVM::load("HOG_SVM_DATA.xml");;//加载训练好的xml文件，这里训练的是10K个手写数字  
	//检测样本      
	cv::Mat test;
	char result[300]; //存放预测结果   
	test = cv::imread("D:/Pedestrians64x128/Positive/30.jpg", -1); //待预测图片，用系统自带的画图工具随便手写  
	if (!test.data)
	{
		cout << "Can not load the image" << endl;
		return -1;
	}
	
	//缩放
	Mat testImage;
	resize(test, testImage, Size(64, 128));

	// 提取HOG特征
	HOGDescriptor hog(cvSize(64, 128), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
	vector<float> descriptors;
	hog.compute(testImage, descriptors);
	cout << "HOG dimensions:" << descriptors.size() << endl;

	Mat featureVectorOfTestImage(1, descriptors.size(), CV_32FC1);
	for (int j = 0; j <= descriptors.size() - 1; ++j)
	{
		featureVectorOfTestImage.at<float>(0, j) = descriptors[j];
	}

	//对测试图片进行分类并写入文件
	int predictResult = svm->predict(featureVectorOfTestImage);
	cout << "The Num Is:" << predictResult << endl;
	//MessageBox(NULL, result, TEXT("预测结果"), MB_OK);
	
	return 0;
}
void Detect_People()
{

	////////////////////////////////读入训练样本图片路径和类别///////////////////////////////////////////////////
	//图像路径和类别
	vector<string> imagePath;
	vector<int> imageClass;
	int numberOfLine = 0;
	string buffer;
	ifstream trainingData(string(FILEPATH) + "TrainData.txt", ios::in);
	unsigned long n;



	while (!trainingData.eof())
	{
		getline(trainingData, buffer);
		if (!buffer.empty())
		{
			++numberOfLine;
			if (numberOfLine % 2 == 0)
			{
				//读取样本类别
				imageClass.push_back(atoi(buffer.c_str()));
			}
			else
			{
				//读取图像路径
				imagePath.push_back(buffer);
			}
		}
	}
	trainingData.close();


	////////////////////////////////获取样本的HOG特征///////////////////////////////////////////////////
	//样本特征向量矩阵
	int numberOfSample = numberOfLine / 2;
	Mat featureVectorOfSample(numberOfSample, 3780, CV_32FC1);//矩阵中每行为一个样本

	//样本的类别
	Mat classOfSample(numberOfSample, 1, CV_32SC1);

	//开始计算训练样本的HOG特征
	for (vector<string>::size_type i = 0; i <= imagePath.size() - 1; ++i)
	{
		//读入图片
		Mat src = imread(imagePath[i], -1);
		if (src.empty())
		{
			cout << "can not load the image:" << imagePath[i] << endl;
			continue;
		}
		cout << "processing" << imagePath[i] << endl;

		//缩放
		Mat trainImage;

		resize(src, trainImage, Size(64, 128));

		// 提取HOG特征
		HOGDescriptor hog(cvSize(64, 128), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
		vector<float> descriptors;
		hog.compute(trainImage, descriptors);//这里可以设置检测窗口步长，如果图片大小超过64×128，可以设置winStride
		//hog.compute(trainImage, descriptors);//这里可以设置检测窗口步长，如果图片大小超过64×128，可以设置winStride
		cout << "HOG dimensions:" << descriptors.size() << endl;

		//保存特征向量矩阵中
		for (vector<float>::size_type j = 0; j <= descriptors.size() - 1; ++j)
		{
			featureVectorOfSample.at<float>(i, j) = descriptors[j];
		}

		//保存类别到类别矩阵
		//!!注意类别类型一定要是int 类型的
		classOfSample.at<int>(i, 0) = imageClass[i];
	}


	///////////////////////////////////使用SVM分类器训练///////////////////////////////////////////////////	
	//设置参数
	//参考3.0的Demo
	Ptr<SVM> svm = SVM::create();
	/*
	svm->setKernel(SVM::RBF);
	svm->setType(SVM::C_SVC);
	svm->setC(10);
	svm->setCoef0(1.0);
	svm->setP(1.0);
	svm->setNu(0.5);
	*/
	svm->setType(SVM::C_SVC);
	svm->setGamma(0.01);
	svm->setDegree(2);
	svm->setC(10);
	svm->setKernel(SVM::POLY);
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON));

	//使用SVM学习         
	svm->train(featureVectorOfSample, ROW_SAMPLE, classOfSample);

	//保存分类器
	svm->save("Classifier.xml");
	cout << "Train Done!" << endl;



	/*
	///////////////////////////////////使用训练好的分类器进行识别///////////////////////////////////////////////////
	vector<string> testImagePath;
	ifstream testData(string(PATH) + "TestData.txt", ios::out);
	while (!testData.eof())
	{
	getline(testData, buffer);
	//读取
	if (!buffer.empty())
	testImagePath.push_back(buffer);

	}
	testData.close();

	ofstream fileOfPredictResult(string(PATH) + "PredictResult.txt"); //最后识别的结果
	for (vector<string>::size_type i = 0; i <= testImagePath.size() - 1; ++i)
	{
	//读取测试图片
	Mat src = imread(testImagePath[i], -1);
	if (src.empty())
	{
	cout << "Can not load the image:" << testImagePath[i] << endl;
	continue;
	}

	//缩放
	Mat testImage;
	resize(src, testImage, Size(128, 128));

	//测试图片提取HOG特征
	HOGDescriptor hog(cvSize(128, 128), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
	vector<float> descriptors;
	hog.compute(testImage, descriptors);
	cout << "HOG dimensions:" << descriptors.size() << endl;

	Mat featureVectorOfTestImage(1, descriptors.size(), CV_32FC1);
	for (int j = 0; j <= descriptors.size() - 1; ++j)
	{
	featureVectorOfTestImage.at<float>(0, j) = descriptors[j];
	}

	//对测试图片进行分类并写入文件
	int predictResult = svm->predict(featureVectorOfTestImage);
	char line[512];
	printf("%s %d\r\n", testImagePath[i].c_str(), predictResult);
	std::sprintf(line, "%s %d\n", testImagePath[i].c_str(), predictResult);
	fileOfPredictResult << line;

	}
	fileOfPredictResult.close();


	*/
}
void XuLie_Test()
{
	Ptr<SVM> svm = StatModel::load<SVM>("Classifier.xml");
	Ptr<ANN_MLP> bp = StatModel::load<ANN_MLP>("bpcharModel.xml");
	Ptr<BackgroundSubtractorMOG2> pBgmodel = createBackgroundSubtractorMOG2(100, 25, false);
	//bgsubtractor->setVarThreshold(20);  
	Mat frame, image, foreGround, backGround, fgMask, Mask_Count, ROI;
	//vector<RotatedRect> People_Vec;
	vector<Rect> People_Vec;
	int trainCounter = 0,Coumt = 0;
	bool dynamicDetect = true;
	long frameNo = 0;
	char str[10];
	char save_file[200];
	char result[300]; //存放预测结果  
	String Num;
	

	cv::Mat test, ROI_Head;
	VideoCapture capture("Ved.mp4");//如果是笔记本，0打开的是自带的摄像头，1 打开外接的相机    
	while (1)
	{
		capture >> frame;

		resize(frame, image, Size(frame.cols / 2, frame.rows / 2), INTER_LINEAR);

		flip(image, image, 0);


		/***************************混合高斯处理*****************************************************/
		if (foreGround.empty())
			foreGround.create(image.size(), image.type());
		//得到前景图像，是黑白灰 3种灰度值的图
		pBgmodel->apply(image, fgMask);

		// 下面是根据前景图的操作，和原图像融合得到有纹理的前景图
		GaussianBlur(fgMask, fgMask, Size(5, 5), 0);
		threshold(fgMask, fgMask, 10, 255, THRESH_BINARY);
		// 将foreGraound 所有像素置为0
		foreGround = Scalar::all(0);
		//fgMask对应点像素值为255则 foreGround像素为image里的像素，为0则直接为0
		image.copyTo(foreGround, fgMask);

		pBgmodel->getBackgroundImage(backGround);


		//foreGround.copyTo(findc);
		vector<vector<Point> > contours;

		Mat element = getStructuringElement(MORPH_RECT, Size(6, 6));  //6 6
		Mat element2 = getStructuringElement(MORPH_RECT, Size(2, 2));  //6 6
		fgMask.copyTo(Mask_Count);

		//GaussianBlur(Mask_Count, Mask_Count, Size(5, 5), 0);
		//	dilate(Mask_Count, Mask_Count, element2);
		erode(Mask_Count, Mask_Count, element);
		dilate(Mask_Count, Mask_Count, element);

		/*
		dilate(Mask_Count, Mask_Count, Mat(), Point(-1, -1), 3);
		erode(Mask_Count, Mask_Count, Mat(), Point(-1, -1), 6);
		dilate(Mask_Count, Mask_Count, Mat(), Point(-1, -1), 3);
		*/
		cv::findContours(Mask_Count.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		//targets.clear();  
		const int maxArea = 5000;
		size_t s = contours.size();
		RotatedRect RectS;   //定义旋转矩形
		sprintf(str, "%d", Coumt);
		Num = str;
		sprintf(save_file, "D:\\Pedestrians64x128\\BpHead\\%d.jpg", Coumt);

		for (size_t i = 0; i < s; i++)
		{
			double area = abs(contourArea(contours[i]));
			//	printf("Are: %f\n", area);
			if (area > maxArea)
			{
				//	RectS = fitEllipse(contours[i]);
				//RectS = minAreaRect(contours[i]);
				//People_Vec.push_back(RectS);
				Rect mr = boundingRect(Mat(contours[i]));
				ROI = image(Rect(mr.x, mr.y, mr.width, mr.height));
				// imwrite(save_file, ROI);
				//rectangle(image, mr, Scalar(0, 0, 255), 2, 8, 0);
				//缩放
				Mat testImage;
				resize(ROI, testImage, Size(64, 128));

				// 提取HOG特征
				HOGDescriptor hog(cvSize(64, 128), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
				vector<float> descriptors;
				hog.compute(testImage, descriptors);
				cout << "HOG dimensions:" << descriptors.size() << endl;

				Mat featureVectorOfTestImage(1, descriptors.size(), CV_32FC1);
				for (int j = 0; j <= descriptors.size() - 1; ++j)
				{
					featureVectorOfTestImage.at<float>(0, j) = descriptors[j];
				}

				//对测试图片进行分类并写入文件
				int predictResult = svm->predict(featureVectorOfTestImage);
				if (predictResult == 1)
				{
					// ROI_Head = ROI(Rect((mr.width / 5), 0, (mr.width / 2), (mr.height/6)));
					 ROI_Head = ROI(Rect(0, 0, mr.width, (mr.height / 6)));
			
					
					imshow("ROI_Head", ROI_Head);
					int Predct = Predect_num(ROI_Head, bp);
				//	imwrite(save_file, ROI_Head);
					if (Predct == 0)
					{
						//Size text_size = cv::getTextSize("Safe", cv::FONT_HERSHEY_COMPLEX, 1, 1,0);

						rectangle(image, mr, Scalar(0, 255, 0), 2, 8, 0);
						putText(image, "Safe", Point(mr.x, mr.y), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 1, 8, 0);
						
					}
					 if (Predct == 1)
					{
						rectangle(image, mr, Scalar(0, 0, 255), 2, 8, 0);
						putText(image, "Dangerous", Point(mr.x, mr.y), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 1, 8, 0);

					}
					Coumt++;
			
				}
				cout << "The Num Is:" << predictResult << endl;
				People_Vec.push_back(mr);
			}
		}


		/***************************End*************************************************************/


		imshow("前景", fgMask);
		//imshow("背景", backGround);
		imshow("image", image);
		//imshow("12", ROI);
		People_Vec.clear();

		waitKey(10);
	}
	
}
/********************************BP****************************/
char* WcharToChar(const wchar_t* wp)
{
	char *m_char;
	int len = WideCharToMultiByte(CP_ACP, 0, wp, wcslen(wp), NULL, 0, NULL, NULL);
	m_char = new char[len + 1];
	WideCharToMultiByte(CP_ACP, 0, wp, wcslen(wp), m_char, len, NULL, NULL);
	m_char[len] = '\0';
	return m_char;
}

wchar_t* CharToWchar(const char* c)
{
	wchar_t *m_wchar;
	int len = MultiByteToWideChar(CP_ACP, 0, c, strlen(c), NULL, 0);
	m_wchar = new wchar_t[len + 1];
	MultiByteToWideChar(CP_ACP, 0, c, strlen(c), m_wchar, len);
	m_wchar[len] = '\0';
	return m_wchar;
}

wchar_t* StringToWchar(const string& s)
{
	const char* p = s.c_str();
	return CharToWchar(p);
}
int BPXublian(void)
{

	const string fileform = "*.jpg";
	const string perfileReadPath = "BpHead";

	const int sample_mun_perclass = 400;//训练字符每类数量  
	const int class_mun = 2;//训练字符类数 0-9 A-Z 除了I、O  

	const int image_cols = 8;
	const int image_rows = 16;
	string  fileReadName,
		fileReadPath;
	char temp[256];

	float trainingData[class_mun*sample_mun_perclass][image_rows*image_cols] = { { 0 } };//每一行一个训练样本  
	float labels[class_mun*sample_mun_perclass][class_mun] = { { 0 } };//训练样本标签  

	for (int i = 0; i <= class_mun - 1; i++)//不同类  
	{
		//读取每个类文件夹下所有图像  
		int j = 0;//每一类读取图像个数计数  

		if (i <= 9)//0-9  
		{
			sprintf(temp, "%d", i);
			//printf("%d\n", i);  
		}
		else//A-Z  
		{
			sprintf(temp, "%c", i + 55);
			//printf("%c\n", i+55);  
		}

		fileReadPath = perfileReadPath + "/" + temp + "/" + fileform;
		cout << "文件夹" << temp << endl;

		HANDLE hFile;
		LPCTSTR lpFileName = StringToWchar(fileReadPath);//指定搜索目录和文件类型，如搜索d盘的音频文件可以是"D:\\*.mp3"  
		WIN32_FIND_DATA pNextInfo;  //搜索得到的文件信息将储存在pNextInfo中;  
		hFile = FindFirstFile(lpFileName, &pNextInfo);//请注意是 &pNextInfo , 不是 pNextInfo;  
		if (hFile == INVALID_HANDLE_VALUE)
		{
			continue;//搜索失败  
		}
		//do-while循环读取  
		do
		{
			if (pNextInfo.cFileName[0] == '.')//过滤.和..  
				continue;
			j++;//读取一张图  
			//wcout<<pNextInfo.cFileName<<endl;  
			//printf("%s\n",WcharToChar(pNextInfo.cFileName));  
			//对读入的图片进行处理  
			Mat srcImage = imread(perfileReadPath + "/" + temp + "/" + WcharToChar(pNextInfo.cFileName), CV_LOAD_IMAGE_GRAYSCALE);
			Mat resizeImage;
			Mat trainImage;
			Mat result;

			resize(srcImage, resizeImage, Size(image_cols, image_rows), (0, 0), (0, 0), CV_INTER_AREA);//使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现  
			threshold(resizeImage, trainImage, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

			for (int k = 0; k<image_rows*image_cols; ++k)
			{
				trainingData[i*sample_mun_perclass + (j - 1)][k] = (float)trainImage.data[k];
				//trainingData[i*sample_mun_perclass+(j-1)][k] = (float)trainImage.at<unsigned char>((int)k/8,(int)k%8);//(float)train_image.data[k];  
				//cout<<trainingData[i*sample_mun_perclass+(j-1)][k] <<" "<< (float)trainImage.at<unsigned char>(k/8,k%8)<<endl;  
			}

		} while (FindNextFile(hFile, &pNextInfo) && j<sample_mun_perclass);//如果设置读入的图片数量，则以设置的为准，如果图片不够，则读取文件夹下所有图片  

	}

	// Set up training data Mat  
	Mat trainingDataMat(class_mun*sample_mun_perclass, image_rows*image_cols, CV_32FC1, trainingData);
	cout << "trainingDataMat——OK！" << endl;

	// Set up label data   
	for (int i = 0; i <= class_mun - 1; ++i)
	{
		for (int j = 0; j <= sample_mun_perclass - 1; ++j)
		{
			for (int k = 0; k < class_mun; ++k)
			{
				if (k == i)
				if (k == 18)
				{
					labels[i*sample_mun_perclass + j][1] = 1;
				}
				else if (k == 24)
				{
					labels[i*sample_mun_perclass + j][0] = 1;
				}
				else
				{
					labels[i*sample_mun_perclass + j][k] = 1;
				}
				else
					labels[i*sample_mun_perclass + j][k] = 0;
			}
		}
	}
	Mat labelsMat(class_mun*sample_mun_perclass, class_mun, CV_32FC1, labels);
	cout << "labelsMat:" << endl;
	ofstream outfile("out.txt");
	outfile << labelsMat;
	//cout<<labelsMat<<endl;  
	cout << "labelsMat——OK！" << endl;

	//训练代码  

	cout << "training start...." << endl;

	Ptr<ANN_MLP> bp = ANN_MLP::create();
	Mat layerSizes = (Mat_<int>(1, 5) << image_rows*image_cols, 128, 128, 128, class_mun);
	bp->setLayerSizes(layerSizes);//  
	bp->setActivationFunction(ANN_MLP::SIGMOID_SYM);
	bp->setTrainMethod(ANN_MLP::BACKPROP, 0.0001, 0.1);
	bp->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001));


	cout << "training...." << endl;
	Ptr<TrainData> tData = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);


	bp->train(tData);
	//bp->train(trainingDataMat, labelsMat, Mat(), Mat());

	bp->save("bpcharModel.xml"); //save classifier

	cout << "training finish...bpModel1.xml saved " << endl;


	//测试神经网络  
	cout << "测试：" << endl;
	Mat test_image = imread("test.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat test_temp;
	resize(test_image, test_temp, Size(image_cols, image_rows), (0, 0), (0, 0), CV_INTER_AREA);//使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现  
	threshold(test_temp, test_temp, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	Mat_<float>sampleMat(1, image_rows*image_cols);
	for (int i = 0; i<image_rows*image_cols; ++i)
	{
		sampleMat.at<float>(0, i) = (float)test_temp.at<uchar>(i / 8, i % 8);
	}

	Mat responseMat;
	bp->predict(sampleMat, responseMat);
	Point maxLoc;
	double maxVal = 0;
	minMaxLoc(responseMat, NULL, &maxVal, NULL, &maxLoc);

	if (maxLoc.x <= 9)//0-9  
	{
		sprintf(temp, "%d", maxLoc.x);
		//printf("%d\n", i);  
	}
	else//A-Z  
	{
		sprintf(temp, "%c", maxLoc.x + 55);
		//printf("%c\n", i+55);  
	}

	cout << "识别结果：" << temp << "    相似度:" << maxVal * 100 << "%" << endl;
	imshow("test_image", test_image);

	waitKey(0);



	return 0;
}
int Predect_num(Mat test_image, Ptr<ANN_MLP> bp)
{
	char temp[256];
	const int image_cols = 8;
	const int image_rows = 16;
	//	Ptr<ANN_MLP> bp = ANN_MLP::create();
	Mat test_temp;

	//测试神经网络  
	cout << "测试：" << endl;
	cvtColor(test_image, test_image, CV_BGR2GRAY);
	//Mat test_image = imread("D:/Pedestrians64x128/BpHead/0/b25.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	resize(test_image, test_temp, Size(image_cols, image_rows), (0, 0), (0, 0), CV_INTER_AREA);//使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现  
	threshold(test_temp, test_temp, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	Mat_<float>sampleMat(1, image_rows*image_cols);
	for (int i = 0; i<image_rows*image_cols; ++i)
	{
		sampleMat.at<float>(0, i) = (float)test_temp.at<uchar>(i / 8, i % 8);
	}

	Mat responseMat;
	bp->predict(sampleMat, responseMat);
	Point maxLoc;
	double maxVal = 0;
	minMaxLoc(responseMat, NULL, &maxVal, NULL, &maxLoc);


	sprintf(temp, "%d", maxLoc.x);
	

	cout << "识别结果：" << temp << "    相似度:" << maxVal * 100 << "%" << endl;
	//imshow("test_image", test_image);
	if ((maxVal * 100) < 90)
	{
		return 2;

		
	}
	return maxLoc.x;
	
}
int main()
{


	XuLie_Test();


	return 0;
}