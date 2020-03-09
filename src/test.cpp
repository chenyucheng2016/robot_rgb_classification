#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include <vector>
#include <math.h>
#include <string>

/*                     0 - 1
Classifier 0:        Box - Sphere
Classifier 1:      Brown - Black
Classifier 2:     Orange - Green
Classifier 3:  Cardboard - Wooden Box
Classifier 4:   Computer - Book
Classifier 5:     Orange - Basketball
Classifier 6: Watermelon - Apple
*/
using namespace cv;
using namespace std;
using namespace cv::ml;

typedef Ptr<ml::SVM> SVMPtr;
void computeRGBfeatures(Mat frame, vector<float>* featurePtr);


int main(int argc, char* argv[]){


    cout<<"Hello"<<endl;
    std::vector<SVMPtr> SVMClassifier;
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
    Mat trainingDataMat, labelsMat, CurImage;
    int imgidx = 0;
    // Initialize HOGDescriptor
    vector<float> features;
    vector<Point> locations;
    cv::HOGDescriptor *hog = new HOGDescriptor();
    ;
    // load images & extract HOG feature
    int total_num = 100;
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
        
        CurImage = imread(imgdir + "/" + trainingObj + to_string(imgidx) + ".jpg");
        // Compute HOG features
        if (classifieridx == 1 || classifieridx == 2) {
            computeRGBfeatures(CurImage,&features);      
        }
        else{
            hog->compute(CurImage,features,Size(32,32), Size(0,0),locations);
        }
        // trainingDataMat.push_back( Mat(features).t() );
        trainingDataMat.push_back( Mat(features).t() );
    }
    // Train the SVM
    cout<<"features"<< features.size()<<endl;
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));
    svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
    // Save Classifier
    
    SVMClassifier.push_back(svm);
    
    cout<<"Success"<<endl;
    }
    cout<<"Training Complete"<<endl;



    /*


    Read


    */

    // Read file image
    string imgdir = "./src/robot_rgb_classification/TrainingImages/WatermelonApple";
    

    int label;
    vector<int> Distrib(8);
    string Filein;
    cv::Mat frame;
    std::vector<float> HOGfeatures;
    std::vector<float> RGBfeatures;
    std::vector<Point> locations;
    cv::HOGDescriptor *hog = new HOGDescriptor();
    int total = 20;
    float response;
    for (int i = 1; i < total; i++){
        Filein = imgdir + "/"+ "Apple" + to_string(i) + ".jpg";

        frame = cv::imread(Filein, cv::IMREAD_COLOR);
        cout<<frame.type();
        if(! frame.data )               // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
            return -1;
        }
        hog->compute(frame,HOGfeatures,Size(32,32), Size(0,0),locations);
        computeRGBfeatures(frame, &RGBfeatures);
        int classifier_idx = 0, label = 0;
        SVMPtr CurClassifier;
        // Layer 1
        CurClassifier = SVMClassifier[classifier_idx];
        response = CurClassifier->predict(HOGfeatures);
        classifier_idx = classifier_idx*2 + response + 1;

        // Layer 2
        CurClassifier = SVMClassifier[classifier_idx];
        response = CurClassifier->predict(RGBfeatures);
        classifier_idx = classifier_idx*2 + response + 1;

        // Layer 3
        CurClassifier = SVMClassifier[classifier_idx];
        response = CurClassifier->predict(HOGfeatures);
        
        label = 2* classifier_idx - 6 + response;
        cout<<endl;
        // cout<<"Predicted Label: "<<label<<endl;
        Distrib[label]++;
    }
    
    for (int i = 0; i < 8; i++) 
        cout<<Distrib[i]<<" "<<endl;
    
    return 0;
}

void computeRGBfeatures(Mat frame, vector<float>* featurePtr){
    featurePtr->clear();
    vector<Mat> rgb_planes;
    split( frame, rgb_planes );
    int histSize = 32;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange = { range };
    bool uniform = true, accumulate = false;
    Mat b_hist, g_hist, r_hist;
    cv::calcHist( &rgb_planes[0], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
    cv::calcHist( &rgb_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    cv::calcHist( &rgb_planes[2], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    vector<float> vec;
    vec.assign(r_hist.begin<float>(), r_hist.end<float>());
    featurePtr->insert(featurePtr->end(), vec.begin(), vec.end());

    vec.assign(g_hist.begin<float>(), g_hist.end<float>());
    featurePtr->insert(featurePtr->end(), vec.begin(), vec.end());

    vec.assign(b_hist.begin<float>(), b_hist.end<float>());
    featurePtr->insert(featurePtr->end(), vec.begin(), vec.end());

}