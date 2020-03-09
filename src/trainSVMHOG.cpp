// opencv includes
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
// #include <opencv2/ocl.hpp>

#include <iostream>
#include <vector>
#include <math.h>
#include <string>
#include "../include/SVMHOGClassification.h"
// cv::svm tutorial:
// https://docs.opencv.org/3.4/d1/d73/tutorial_introduction_to_svm.html
using namespace std;
using namespace cv;
using namespace cv::ml;

void computeRGBfeatures(Mat frame, vector<float>* featurePtr);
int main(){
    
    cout<<"Hello"<<endl;
    // int classifieridx = 2;
    for (int classifieridx = 0; classifieridx < 7; classifieridx++){
    string imgdir = "./src/robot_rgb_classification/TrainingImages/";
    vector<string> ObjList = {"Box", "Sphere", "BrownBox", "BlackBox", "OrangeSphere", "GreenSphere", "CardboardBox", "WoodenBox", 
    "Computer", "Book", "Orange", "Basketball", "Watermelon","Apple"};
    imgdir = imgdir + ObjList[2*classifieridx] + ObjList[2*classifieridx+1];
    // for (int i = 0; i<7; i++){
    // dir = "TrainingSet_" + to_string(i);
    // }
    cout<<"Training Object: "<<ObjList[2*classifieridx]<<"(0) and "<<ObjList[2*classifieridx+1]
        <<"(1), at "<<imgdir<<endl;
    // Train
    string trainingObj;
    Mat trainingDataMat, labelsMat, CurImage, trainingImg;
    Rect roi(160, 240, 480, 240);
    int imgidx = 0;
    // Initialize HOGDescriptor
    vector<float> HOGfeatures;
    vector<float> RGBfeatures;
    vector<float> features;

    vector<Point> locations;
    cv::HOGDescriptor *hog = new HOGDescriptor();
    // load images & extract HOG feature
    int total_num = (classifieridx < 3) ? 400:200;
    for (int i = 1; i < total_num+1; i++){
        if (i < total_num/2+1){
            trainingObj = ObjList[2*classifieridx];
            labelsMat.push_back(0);
            imgidx = i;
        }
        else{
            trainingObj = ObjList[2*classifieridx+1];
            labelsMat.push_back(1);
            imgidx = i - total_num/2;
        }
        // cout<<imgidx<<" ";
        CurImage = imread(imgdir + "/" + trainingObj + to_string(imgidx) + ".jpg");
        trainingImg = CurImage(roi);
        // Compute HOG features
        if (classifieridx == 0){
            hog->compute(trainingImg,features,Size(HOG_SIZE,HOG_SIZE), Size(0,0),locations);
            trainingDataMat.push_back( Mat(features).t() );
        }
        else if (classifieridx == 1 || classifieridx == 2) {
            cv::resize(trainingImg, trainingImg, cv::Size(), 0.2, 0.2);
            computeRGBfeatures(trainingImg,&features);      
            trainingDataMat.push_back( Mat(features).t() );
        }
        else{
            hog->compute(trainingImg,features,Size(HOG_SIZE,HOG_SIZE), Size(0,0),locations);
            trainingDataMat.push_back( Mat(features).t() );
        }
        // trainingDataMat.push_back( Mat(features).t() );
        
    }
    // Train the SVM
    cout<<"Feature Size: "<< features.size()<<endl;
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));
    svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
    // Save Classifier
    string classifiername = "./src/robot_rgb_classification/models/SVMClassifier_" + std::to_string(classifieridx) + ".xml";
    svm->save(classifiername);
    cout<<"Success"<<endl;
    }
    cout<<"Training Complete"<<endl;
    return 0;
};

void computeRGBfeatures(Mat frame, vector<float>* featurePtr){
    featurePtr->clear();
    // cv::cvtColor(frame, frame, CV_BGR2HSV);
    vector<Mat> rgb_planes;
    split( frame, rgb_planes );
    int histSize = COLOR_SIZE;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange = { range };
    bool uniform = true, accumulate = false;
    // OpenCV stores images in BGR order
    // Mat b_hist, g_hist, r_hist;
    // cv::calcHist( &rgb_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
    // cv::calcHist( &rgb_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    // cv::calcHist( &rgb_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    Mat channels = rgb_planes[2];
    vector<uchar> vec;
    if (channels.isContinuous()){
        vec.assign(channels.data, channels.data + channels.total());
        // vec.assign(r_hist.begin<float>(), r_hist.end<float>());
        featurePtr->insert(featurePtr->end(), vec.begin(), vec.end());
    }
    
    channels = rgb_planes[1];
    if (channels.isContinuous()){
        vec.assign(channels.data, channels.data + channels.total());
        // vec.assign(g_hist.begin<float>(), g_hist.end<float>());
        featurePtr->insert(featurePtr->end(), vec.begin(), vec.end());
    }

    channels = rgb_planes[0];
    if (channels.isContinuous()){
        vec.assign(channels.data, channels.data + channels.total());
        // vec.assign(b_hist.begin<float>(), b_hist.end<float>());
        featurePtr->insert(featurePtr->end(), vec.begin(), vec.end());
    }

}
