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
///////////////////////////////////HOG+SVMʶ��ʽ2///////////////////////////////////////////////////	
void Train()
{
	////////////////////////////////����ѵ������ͼƬ·�������///////////////////////////////////////////////////
	//ͼ��·�������
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
				//��ȡ�������
				imageClass.push_back(atoi(buffer.c_str()));
			}
			else
			{
				//��ȡͼ��·��
				imagePath.push_back(buffer);
			}
		}
	}

	//�ر��ļ�  
	trainingData.close();


	////////////////////////////////��ȡ������HOG����///////////////////////////////////////////////////
	//����������������
	int numberOfSample = numberOfLine / 2;
	Mat featureVectorOfSample(numberOfSample, 3780, CV_32FC1);//������ÿ��Ϊһ������

	//���������
	Mat classOfSample(numberOfSample, 1, CV_32SC1);

	Mat convertedImage;
	Mat trainImage;

	// ����HOG����
	for (vector<string>::size_type i = 0; i <= imagePath.size() - 1; ++i)
	{
		//����ͼƬ
		Mat src = imread(imagePath[i], -1);
		if (src.empty())
		{
			cout << "can not load the image:" << imagePath[i] << endl;
			continue;
		}
		cout << "processing:" << imagePath[i] << endl;

		// ��һ��
		resize(src, trainImage, Size(64, 128));

		// ��ȡHOG����
		HOGDescriptor hog(cvSize(64, 128), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
		vector<float> descriptors;
		double time1 = getTickCount();
		hog.compute(trainImage, descriptors);//����������ü�ⴰ�ڲ��������ͼƬ��С����64��128����������winStride
		double time2 = getTickCount();
		double elapse_ms = (time2 - time1) * 1000 / getTickFrequency();
		cout << "HOG dimensions:" << descriptors.size() << endl;
		//cout << "Compute time:" << elapse_ms << endl;


		//���浽��������������
		for (vector<float>::size_type j = 0; j <= descriptors.size() - 1; ++j)
		{
			featureVectorOfSample.at<float>(i, j) = descriptors[j];
		}

		//�������������
		//!!ע���������һ��Ҫ��int ���͵�
		classOfSample.at<int>(i, 0) = imageClass[i];
	}


	///////////////////////////////////ʹ��SVM������ѵ��///////////////////////////////////////////////////	
	//���ò�����ע��Ptr��ʹ��
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);//ע�����ʹ������SVM����ѵ������ΪHogDescriptor��⺯��ֻ֧�����Լ�⣡����
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON));

	//ʹ��SVMѧϰ         
	svm->train(featureVectorOfSample, ROW_SAMPLE, classOfSample);

	//���������(���������SVM�Ĳ�����֧������,����rho)
	svm->save(string(FILEPATH) + "Classifier.xml");

	/*
	SVMѵ����ɺ�õ���XML�ļ����棬��һ�����飬����support vector������һ�����飬����alpha,��һ��������������rho;
	��alpha����ͬsupport vector��ˣ�ע�⣬alpha*supportVector,���õ�һ������������������ǰ�����-1��֮���ٸ���������������һ��Ԫ��rho��
	��ˣ���õ���һ�������������ø÷�������ֱ���滻opencv�����˼��Ĭ�ϵ��Ǹ���������cv::HOGDescriptor::setSVMDetector()����
	*/
	//��ȡ֧��������������Ĭ����CV_32F
	Mat supportVector = svm->getSupportVectors();//

	//��ȡalpha��rho
	Mat alpha;//ÿ��֧��������Ӧ�Ĳ�����(�������ճ���)��Ĭ��alpha��float64��
	Mat svIndex;//֧���������ڵ�����
	float rho = svm->getDecisionFunction(0, alpha, svIndex);

	//ת������:����һ��Ҫע�⣬��Ҫת��Ϊ32��
	Mat alpha2;
	alpha.convertTo(alpha2, CV_32FC1);

	//������������������
	Mat result(1, 3780, CV_32FC1);
	result = alpha2*supportVector;

	//����-1������Ϊʲô�����-1��
	//ע����Ϊsvm.predictʹ�õ���alpha*sv*another-rho�����Ϊ���Ļ�����Ϊ������������HOG�ļ�⺯���У�ʹ��rho+alpha*sv*another(anotherΪ-1)
	for (int i = 0; i < 3780; ++i)
		result.at<float>(0, i) *= -1;

	//�����������浽�ļ�������HOGʶ��
	//��������������б����Ĳ���(��)��HOG����ֱ��ʹ�øò�������ʶ��
	FILE *fp = fopen((string(FILEPATH) + "HOG_SVM.txt").c_str(), "wb");
	for (int i = 0; i<3780; i++)
	{
		fprintf(fp, "%f \n", result.at<float>(0, i));
	}
	fprintf(fp, "%f", rho);

	fclose(fp);

}
// ʹ��ѵ���õķ�����ʶ��
void Detect()
{
	Mat img;
	FILE* f = 0;
	char _filename[1024];

	// ��ȡ����ͼƬ�ļ�·��
	f = fopen((string(FILEPATH) + "TestData.txt").c_str(), "rt");
	if (!f)
	{
		fprintf(stderr, "ERROR: the specified file could not be loaded\n");
		return;
	}


	//����ѵ���õ��б����Ĳ���(ע�⣬��svm->save����ķ�������ͬ)
	vector<float> detector;
	ifstream fileIn(string(FILEPATH) + "HOG_SVM.txt", ios::in);
	float val = 0.0f;
	while (!fileIn.eof())
	{
		fileIn >> val;
		detector.push_back(val);
	}
	fileIn.close();

	//����HOG
	HOGDescriptor hog;
	hog.setSVMDetector(detector);// ʹ���Լ�ѵ���ķ�����
	//hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//����ֱ��ʹ��05 CVPR��ѵ���õķ�����,�����Ͳ���Train()���������
	namedWindow("people detector", 1);

	// ���ͼƬ
	for (;;)
	{/*
		// ��ȡ�ļ���
		char* filename = _filename;
		if (f)
		{
			if (!fgets(filename, (int)sizeof(_filename)-2, f))
				break;
			//while(*filename && isspace(*filename))
			//  ++filename;
			if (filename[0] == '#')
				continue;

			//ȥ���ո�
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
		//��߶ȼ��
		hog.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
		t = (double)getTickCount() - t;
		printf("detection time = %gms\n", t*1000. / cv::getTickFrequency());
		size_t i, j;

		//ȥ���ռ��о������������ϵ�����򣬱������
		for (i = 0; i < found.size(); i++)
		{
			Rect r = found[i];
			for (j = 0; j < found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
			if (j == found.size())
				found_filtered.push_back(r);
		}

		// �ʵ���С����
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
	//���ʹ��05 CVPR�ṩ��Ĭ�Ϸ�����������ҪTrain(),ֱ��ʹ��Detect���ͼƬ
	Train();
	Detect();
}

#define PATH "E:/opencv/opencv3.0/MyProject/res/BigBossPic/"
///////////////////////////////////HOG+SVMʶ��ʽ1///////////////////////////////////////////////////
void HOG_SVM1()
{
	////////////////////////////////����ѵ������ͼƬ·�������///////////////////////////////////////////////////
	//ͼ��·�������
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
				//��ȡ�������
				imageClass.push_back(atoi(buffer.c_str()));
			}
			else
			{
				//��ȡͼ��·��
				imagePath.push_back(buffer);
			}
		}
	}
	trainingData.close();


	////////////////////////////////��ȡ������HOG����///////////////////////////////////////////////////
	//����������������
	int numberOfSample = numberOfLine / 2;
	Mat featureVectorOfSample(numberOfSample, 324, CV_32FC1);//������ÿ��Ϊһ������

	//���������
	Mat classOfSample(numberOfSample, 1, CV_32SC1);

	//��ʼ����ѵ��������HOG����
	for (vector<string>::size_type i = 0; i <= imagePath.size() - 1; ++i)
	{
		//����ͼƬ
		Mat src = imread(imagePath[i], -1);
		if (src.empty())
		{
			cout << "can not load the image:" << imagePath[i] << endl;
			continue;
		}
		cout << "processing" << imagePath[i] << endl;

		//����
		Mat trainImage;
		resize(src, trainImage, Size(28, 28));

		//��ȡHOG����
	//	HOGDescriptor hog(Size(128, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
		HOGDescriptor hog(Size(28, 28), Size(14, 14), Size(7, 7), Size(7, 7), 9);
		vector<float> descriptors;
		hog.compute(trainImage, descriptors);//����������ü�ⴰ�ڲ��������ͼƬ��С����64��128����������winStride
		//hog.compute(trainImage, descriptors);//����������ü�ⴰ�ڲ��������ͼƬ��С����64��128����������winStride
		cout << "HOG dimensions:" << descriptors.size() << endl;

		//������������������
		for (vector<float>::size_type j = 0; j <= descriptors.size() - 1; ++j)
		{
			featureVectorOfSample.at<float>(i, j) = descriptors[j];
		}

		//�������������
		//!!ע���������һ��Ҫ��int ���͵�
		classOfSample.at<int>(i, 0) = imageClass[i];
	}


	///////////////////////////////////ʹ��SVM������ѵ��///////////////////////////////////////////////////	
	//���ò���
	//�ο�3.0��Demo
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

	//ʹ��SVMѧϰ         
	svm->train(featureVectorOfSample, ROW_SAMPLE, classOfSample);

	//���������
	svm->save("Classifier.xml");
	cout << "Train Done!" << endl;

	
	
	/*
	///////////////////////////////////ʹ��ѵ���õķ���������ʶ��///////////////////////////////////////////////////
	vector<string> testImagePath;
	ifstream testData(string(PATH) + "TestData.txt", ios::out);
	while (!testData.eof())
	{
		getline(testData, buffer);
		//��ȡ
		if (!buffer.empty())
			testImagePath.push_back(buffer);

	}
	testData.close();

	ofstream fileOfPredictResult(string(PATH) + "PredictResult.txt"); //���ʶ��Ľ��
	for (vector<string>::size_type i = 0; i <= testImagePath.size() - 1; ++i)
	{
		//��ȡ����ͼƬ
		Mat src = imread(testImagePath[i], -1);
		if (src.empty())
		{
			cout << "Can not load the image:" << testImagePath[i] << endl;
			continue;
		}

		//����
		Mat testImage;
		resize(src, testImage, Size(128, 128));

		//����ͼƬ��ȡHOG����
		HOGDescriptor hog(cvSize(128, 128), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
		vector<float> descriptors;
		hog.compute(testImage, descriptors);
		cout << "HOG dimensions:" << descriptors.size() << endl;

		Mat featureVectorOfTestImage(1, descriptors.size(), CV_32FC1);
		for (int j = 0; j <= descriptors.size() - 1; ++j)
		{
			featureVectorOfTestImage.at<float>(0, j) = descriptors[j];
		}

		//�Բ���ͼƬ���з��ಢд���ļ�
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
	//svm = SVM::load("HOG_SVM_DATA.xml");;//����ѵ���õ�xml�ļ�������ѵ������10K����д����  
	//�������      
	cv::Mat test;
	char result[300]; //���Ԥ����   
	test = cv::imread("D:/Pedestrians64x128/Positive/30.jpg", -1); //��Ԥ��ͼƬ����ϵͳ�Դ��Ļ�ͼ���������д  
	if (!test.data)
	{
		cout << "Can not load the image" << endl;
		return -1;
	}
	
	//����
	Mat testImage;
	resize(test, testImage, Size(64, 128));

	// ��ȡHOG����
	HOGDescriptor hog(cvSize(64, 128), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
	vector<float> descriptors;
	hog.compute(testImage, descriptors);
	cout << "HOG dimensions:" << descriptors.size() << endl;

	Mat featureVectorOfTestImage(1, descriptors.size(), CV_32FC1);
	for (int j = 0; j <= descriptors.size() - 1; ++j)
	{
		featureVectorOfTestImage.at<float>(0, j) = descriptors[j];
	}

	//�Բ���ͼƬ���з��ಢд���ļ�
	int predictResult = svm->predict(featureVectorOfTestImage);
	cout << "The Num Is:" << predictResult << endl;
	//MessageBox(NULL, result, TEXT("Ԥ����"), MB_OK);
	
	return 0;
}
void Detect_People()
{

	////////////////////////////////����ѵ������ͼƬ·�������///////////////////////////////////////////////////
	//ͼ��·�������
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
				//��ȡ�������
				imageClass.push_back(atoi(buffer.c_str()));
			}
			else
			{
				//��ȡͼ��·��
				imagePath.push_back(buffer);
			}
		}
	}
	trainingData.close();


	////////////////////////////////��ȡ������HOG����///////////////////////////////////////////////////
	//����������������
	int numberOfSample = numberOfLine / 2;
	Mat featureVectorOfSample(numberOfSample, 3780, CV_32FC1);//������ÿ��Ϊһ������

	//���������
	Mat classOfSample(numberOfSample, 1, CV_32SC1);

	//��ʼ����ѵ��������HOG����
	for (vector<string>::size_type i = 0; i <= imagePath.size() - 1; ++i)
	{
		//����ͼƬ
		Mat src = imread(imagePath[i], -1);
		if (src.empty())
		{
			cout << "can not load the image:" << imagePath[i] << endl;
			continue;
		}
		cout << "processing" << imagePath[i] << endl;

		//����
		Mat trainImage;

		resize(src, trainImage, Size(64, 128));

		// ��ȡHOG����
		HOGDescriptor hog(cvSize(64, 128), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
		vector<float> descriptors;
		hog.compute(trainImage, descriptors);//����������ü�ⴰ�ڲ��������ͼƬ��С����64��128����������winStride
		//hog.compute(trainImage, descriptors);//����������ü�ⴰ�ڲ��������ͼƬ��С����64��128����������winStride
		cout << "HOG dimensions:" << descriptors.size() << endl;

		//������������������
		for (vector<float>::size_type j = 0; j <= descriptors.size() - 1; ++j)
		{
			featureVectorOfSample.at<float>(i, j) = descriptors[j];
		}

		//�������������
		//!!ע���������һ��Ҫ��int ���͵�
		classOfSample.at<int>(i, 0) = imageClass[i];
	}


	///////////////////////////////////ʹ��SVM������ѵ��///////////////////////////////////////////////////	
	//���ò���
	//�ο�3.0��Demo
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

	//ʹ��SVMѧϰ         
	svm->train(featureVectorOfSample, ROW_SAMPLE, classOfSample);

	//���������
	svm->save("Classifier.xml");
	cout << "Train Done!" << endl;



	/*
	///////////////////////////////////ʹ��ѵ���õķ���������ʶ��///////////////////////////////////////////////////
	vector<string> testImagePath;
	ifstream testData(string(PATH) + "TestData.txt", ios::out);
	while (!testData.eof())
	{
	getline(testData, buffer);
	//��ȡ
	if (!buffer.empty())
	testImagePath.push_back(buffer);

	}
	testData.close();

	ofstream fileOfPredictResult(string(PATH) + "PredictResult.txt"); //���ʶ��Ľ��
	for (vector<string>::size_type i = 0; i <= testImagePath.size() - 1; ++i)
	{
	//��ȡ����ͼƬ
	Mat src = imread(testImagePath[i], -1);
	if (src.empty())
	{
	cout << "Can not load the image:" << testImagePath[i] << endl;
	continue;
	}

	//����
	Mat testImage;
	resize(src, testImage, Size(128, 128));

	//����ͼƬ��ȡHOG����
	HOGDescriptor hog(cvSize(128, 128), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
	vector<float> descriptors;
	hog.compute(testImage, descriptors);
	cout << "HOG dimensions:" << descriptors.size() << endl;

	Mat featureVectorOfTestImage(1, descriptors.size(), CV_32FC1);
	for (int j = 0; j <= descriptors.size() - 1; ++j)
	{
	featureVectorOfTestImage.at<float>(0, j) = descriptors[j];
	}

	//�Բ���ͼƬ���з��ಢд���ļ�
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
	char result[300]; //���Ԥ����  
	String Num;
	

	cv::Mat test, ROI_Head;
	VideoCapture capture("Ved.mp4");//����ǱʼǱ���0�򿪵����Դ�������ͷ��1 ����ӵ����    
	while (1)
	{
		capture >> frame;

		resize(frame, image, Size(frame.cols / 2, frame.rows / 2), INTER_LINEAR);

		flip(image, image, 0);


		/***************************��ϸ�˹����*****************************************************/
		if (foreGround.empty())
			foreGround.create(image.size(), image.type());
		//�õ�ǰ��ͼ���Ǻڰ׻� 3�ֻҶ�ֵ��ͼ
		pBgmodel->apply(image, fgMask);

		// �����Ǹ���ǰ��ͼ�Ĳ�������ԭͼ���ںϵõ��������ǰ��ͼ
		GaussianBlur(fgMask, fgMask, Size(5, 5), 0);
		threshold(fgMask, fgMask, 10, 255, THRESH_BINARY);
		// ��foreGraound ����������Ϊ0
		foreGround = Scalar::all(0);
		//fgMask��Ӧ������ֵΪ255�� foreGround����Ϊimage������أ�Ϊ0��ֱ��Ϊ0
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
		RotatedRect RectS;   //������ת����
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
				//����
				Mat testImage;
				resize(ROI, testImage, Size(64, 128));

				// ��ȡHOG����
				HOGDescriptor hog(cvSize(64, 128), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
				vector<float> descriptors;
				hog.compute(testImage, descriptors);
				cout << "HOG dimensions:" << descriptors.size() << endl;

				Mat featureVectorOfTestImage(1, descriptors.size(), CV_32FC1);
				for (int j = 0; j <= descriptors.size() - 1; ++j)
				{
					featureVectorOfTestImage.at<float>(0, j) = descriptors[j];
				}

				//�Բ���ͼƬ���з��ಢд���ļ�
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


		imshow("ǰ��", fgMask);
		//imshow("����", backGround);
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

	const int sample_mun_perclass = 400;//ѵ���ַ�ÿ������  
	const int class_mun = 2;//ѵ���ַ����� 0-9 A-Z ����I��O  

	const int image_cols = 8;
	const int image_rows = 16;
	string  fileReadName,
		fileReadPath;
	char temp[256];

	float trainingData[class_mun*sample_mun_perclass][image_rows*image_cols] = { { 0 } };//ÿһ��һ��ѵ������  
	float labels[class_mun*sample_mun_perclass][class_mun] = { { 0 } };//ѵ��������ǩ  

	for (int i = 0; i <= class_mun - 1; i++)//��ͬ��  
	{
		//��ȡÿ�����ļ���������ͼ��  
		int j = 0;//ÿһ���ȡͼ���������  

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
		cout << "�ļ���" << temp << endl;

		HANDLE hFile;
		LPCTSTR lpFileName = StringToWchar(fileReadPath);//ָ������Ŀ¼���ļ����ͣ�������d�̵���Ƶ�ļ�������"D:\\*.mp3"  
		WIN32_FIND_DATA pNextInfo;  //�����õ����ļ���Ϣ��������pNextInfo��;  
		hFile = FindFirstFile(lpFileName, &pNextInfo);//��ע���� &pNextInfo , ���� pNextInfo;  
		if (hFile == INVALID_HANDLE_VALUE)
		{
			continue;//����ʧ��  
		}
		//do-whileѭ����ȡ  
		do
		{
			if (pNextInfo.cFileName[0] == '.')//����.��..  
				continue;
			j++;//��ȡһ��ͼ  
			//wcout<<pNextInfo.cFileName<<endl;  
			//printf("%s\n",WcharToChar(pNextInfo.cFileName));  
			//�Զ����ͼƬ���д���  
			Mat srcImage = imread(perfileReadPath + "/" + temp + "/" + WcharToChar(pNextInfo.cFileName), CV_LOAD_IMAGE_GRAYSCALE);
			Mat resizeImage;
			Mat trainImage;
			Mat result;

			resize(srcImage, resizeImage, Size(image_cols, image_rows), (0, 0), (0, 0), CV_INTER_AREA);//ʹ�����ع�ϵ�ز�������ͼ����Сʱ�򣬸÷������Ա��Ⲩ�Ƴ���  
			threshold(resizeImage, trainImage, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

			for (int k = 0; k<image_rows*image_cols; ++k)
			{
				trainingData[i*sample_mun_perclass + (j - 1)][k] = (float)trainImage.data[k];
				//trainingData[i*sample_mun_perclass+(j-1)][k] = (float)trainImage.at<unsigned char>((int)k/8,(int)k%8);//(float)train_image.data[k];  
				//cout<<trainingData[i*sample_mun_perclass+(j-1)][k] <<" "<< (float)trainImage.at<unsigned char>(k/8,k%8)<<endl;  
			}

		} while (FindNextFile(hFile, &pNextInfo) && j<sample_mun_perclass);//������ö����ͼƬ�������������õ�Ϊ׼�����ͼƬ���������ȡ�ļ���������ͼƬ  

	}

	// Set up training data Mat  
	Mat trainingDataMat(class_mun*sample_mun_perclass, image_rows*image_cols, CV_32FC1, trainingData);
	cout << "trainingDataMat����OK��" << endl;

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
	cout << "labelsMat����OK��" << endl;

	//ѵ������  

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


	//����������  
	cout << "���ԣ�" << endl;
	Mat test_image = imread("test.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat test_temp;
	resize(test_image, test_temp, Size(image_cols, image_rows), (0, 0), (0, 0), CV_INTER_AREA);//ʹ�����ع�ϵ�ز�������ͼ����Сʱ�򣬸÷������Ա��Ⲩ�Ƴ���  
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

	cout << "ʶ������" << temp << "    ���ƶ�:" << maxVal * 100 << "%" << endl;
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

	//����������  
	cout << "���ԣ�" << endl;
	cvtColor(test_image, test_image, CV_BGR2GRAY);
	//Mat test_image = imread("D:/Pedestrians64x128/BpHead/0/b25.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	resize(test_image, test_temp, Size(image_cols, image_rows), (0, 0), (0, 0), CV_INTER_AREA);//ʹ�����ع�ϵ�ز�������ͼ����Сʱ�򣬸÷������Ա��Ⲩ�Ƴ���  
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
	

	cout << "ʶ������" << temp << "    ���ƶ�:" << maxVal * 100 << "%" << endl;
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