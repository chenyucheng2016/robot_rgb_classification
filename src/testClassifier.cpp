#include <string>
#include "../include/SVMHOGClassification.h"
/*                     0 - 1
Classifier 0:        Box - Sphere
Classifier 1:      Brown - Black
Classifier 2:     Orange - Green
Classifier 3:  Cardboard - Wooden Box
Classifier 4:   Computer - Book
Classifier 5:     Orange - Basketball
Classifier 6: Watermelon - Apple
*/

void computeRGBfeatures(Mat frame, vector<float>* featurePtr);

int main(int argc, char* argv[]){
    using namespace cv;
    using namespace std;
    // Read file image
    cout<<"Hello"<<endl;
    string filename = "./src/robot_rgb_classification/models/SVMClassifier_0.xml";
    Ptr<ml::SVM> svmNew = Algorithm::load<ml::SVM>(filename);
    bool testColorHist = 0;
    cout<<"Successfully Read All Models"<<endl;
    HOGDescriptor *hog = new HOGDescriptor();
    string imgdir = "./src/robot_rgb_classification/TrainingImages/CardboardBoxWoodenBox/";
    
    int classifier_idx = 0, response = 0;
    vector<int> Distrib(2);
    string Filein;
    cv::Mat frame;
    std::vector<float> features;
    std::vector<Point> locations;
    for (int i = 1; i < 20; i++){
        Filein = imgdir + "CardboardBox" + to_string(i) + ".jpg";
        frame = cv::imread(Filein, cv::IMREAD_COLOR);
        if(! frame.data )               // Check for invalid input
        {
            cout <<  "Could not open or find the image" << endl ;
            return -1;
        }
        if (testColorHist) {
            computeRGBfeatures(frame,&features);      
        }
        else{
            hog->compute(frame,features,Size(16,16), Size(0,0),locations);
        }
        response = svmNew->predict(features);
        cout<<response<<" ";
    }
    cout<<endl;
    
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
