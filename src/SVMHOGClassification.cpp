#include "../include/SVMHOGClassification.h"
SVMHOGClassification::SVMHOGClassification(){
    // Load Classifiers
    string path = "./src/robot_rgb_classification/models/";
    string filename;
    SVMPtr svmNew;
    for (int i = 0; i < 7; i++){
        cout<<"Reading Model "<<i<<endl;
        filename = path + "SVMClassifier_" + to_string(i) + ".xml";
        svmNew = Algorithm::load<ml::SVM>(filename);
        // check if loaded
        SVMClassifier.push_back(svmNew);
    }
    cout<<"Successfully Read All Models"<<endl;
    hog = new HOGDescriptor();
    roi = {160, 240, 480, 240};
}

int SVMHOGClassification::Detect_SSD(Mat frame){
    // frame.convertTo(frame, CV_8U);
    //https://answers.opencv.org/question/70491/matching-hog-images-with-opencv-in-c/
    Mat trainingImg = frame(roi);
    

    hog->compute(trainingImg,HOGfeatures,Size(HOG_SIZE,HOG_SIZE), Size(0,0),locations);
    std::vector<float> features = HOGfeatures;

    cv::resize(trainingImg, trainingImg, cv::Size(), 0.2, 0.2);
    computeRGBfeatures(trainingImg, &RGBfeatures);
    // features.insert(features.end(), RGBfeatures.begin(), RGBfeatures.end() );
    int classifier_idx = 0, label = 0;
    SVMPtr CurClassifier;
    // Layer 1
    // std::cout<<classifier_idx<<" ";
    CurClassifier = SVMClassifier[classifier_idx];
    response = CurClassifier->predict(HOGfeatures);
    classifier_idx = classifier_idx*2 + response + 1;

    // Layer 2

    // std::cout<<classifier_idx<<" ";
    CurClassifier = SVMClassifier[classifier_idx];
    response = CurClassifier->predict(RGBfeatures);
    classifier_idx = classifier_idx*2 + response + 1;

    // std::cout<<classifier_idx<<" "<<std::endl;
    // Layer 3
    CurClassifier = SVMClassifier[classifier_idx];
    response = CurClassifier->predict(HOGfeatures);
    
    label = 2* classifier_idx - 6 + response;
    return label;
}

int SVMHOGClassification::Detect_SSD(Mat frame, int key_pressed, int label_int){
    // frame.convertTo(frame, CV_8U);
    //https://answers.opencv.org/question/70491/matching-hog-images-with-opencv-in-c/
    Mat trainingImg = frame(roi);
    SVMPtr CurClassifier = SVMClassifier[label_int];
    if (label_int == 0) {
        key_pressed = 1;
    }else if (label_int > 0 && label_int <= 2){
        key_pressed = 2;
    }else {
        key_pressed = 3;
    }
    switch (key_pressed){
        case 1:
            hog->compute(trainingImg,HOGfeatures,Size(HOG_SIZE,HOG_SIZE), Size(0,0),locations);
            response = CurClassifier->predict(HOGfeatures);
            break;
        case 2:
            cv::resize(trainingImg, trainingImg, cv::Size(), 0.2, 0.2);
            computeRGBfeatures(trainingImg, &RGBfeatures);
            response = CurClassifier->predict(RGBfeatures);
            break;
        case 3:
            hog->compute(trainingImg,HOGfeatures,Size(HOG_SIZE,HOG_SIZE), Size(0,0),locations);
            response = CurClassifier->predict(HOGfeatures);
            break;
        default:
            std::cout<<"Wrong Key";
    }
    label_int = label_int*2 + response + 1;

    return label_int;
}

void SVMHOGClassification::computeRGBfeatures(Mat frame, vector<float>* featurePtr){
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