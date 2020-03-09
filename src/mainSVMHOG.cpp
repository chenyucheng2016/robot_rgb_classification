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
int main(int argc, char* argv[]){
    using namespace std;
    // Read file image
    string imgdir = "./src/robot_rgb_classification/TrainingImages/BoxSphere";
    
    SVMHOGClassification Classifier;
    vector<string> ObjList = {"CardboardBox", "WoodenBox", 
    "Computer", "Book", "Orange", "Basketball", "Watermelon","Apple"};
    int label;
    vector<int> Distrib(8);
    string Filein;
    cv::Mat frame;
    int total = 200;
    for (int i = 1; i < total; i++){
        Filein = imgdir + "/"+ "Sphere" + to_string(i) + ".jpg";

        frame = cv::imread(Filein, cv::IMREAD_COLOR);
        if(! frame.data )               // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
            return -1;
        }
        label = Classifier.Detect_SSD(frame);
        // cout<<"Predicted Label: "<<label<<endl;
        cout<<ObjList[label]<<endl;
    }
    

    return 0;
}

		
		
