#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include "../include/SimpleClassification.h"
#include "../include/SVMHOGClassification.h"
#include "std_msgs/Int32.h"
#include <cmath>

static const std::string OPENCV_WINDOW = "Image window";

class ImageClassifier
{
private:
  ros::NodeHandle nh_;
  // Image Transport
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  cv_bridge::CvImagePtr cv_ptr;
  
  // Classification
  //  SimpleClassification Classifier;
  ros::Publisher label_pub = nh_.advertise<std_msgs::Int32>("SVM_labels", 10);
  SVMHOGClassification Classifier;
  std_msgs::Int32 label; 
  int imgIdx = 0;
  int key_pressed = -1, key_pre = 0;
  //encode num_meausre*19 + label_int
  int start_label_int = -1;
  int label_int = -1;
  int num_measure = 0;
  int started = 0;
  bool DEBUG = 0;
  std::vector<string> ObjList_key = {"Box", "Sphere", "BrownBox", "BlackBox", "OrangeSphere", "GreenSphere", "CardboardBox", "WoodenBox", 
    "Computer", "Book", "Orange", "Basketball", "Watermelon","Apple"};
  std::vector<string> CategoryList = {"Level1: ", "Levle2: ", "Level3: "};  
  std::vector<string> ObjList = {"CardboardBox", "WoodenBox", "Computer", "Book", "Orange", "Basketball", "Watermelon","Apple"};
  std::vector<double> ProbTable = {0.5097,0.4663,0.5490,0.59,0.2543,0.4189,0.6467,0.69,0.39,0.14,0.94,0.88,0.05,0.59,0.93};
  std::vector<int> classResults;
  // Keyboard Input
  ros::Subscriber key_sub_;
  
public:
  ImageClassifier()
    : it_(nh_)
  {
	  std::cout<<"Constructor"<<std::endl;
    
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/camera/rgb/image_rect_color", 1,
      &ImageClassifier::imageCb, this);
    key_sub_ = nh_.subscribe("rgb_classification/keypressed", 1,
      &ImageClassifier::keyCb, this);
  }

  ~ImageClassifier()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void keyCb(const std_msgs::Int32& msg){
      start_label_int = msg.data % 19;
      num_measure = msg.data / 19;
      std::cout<<"start_label_int "<<start_label_int<<std::endl;
      std::cout<<"num_measure "<<num_measure<<std::endl;
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    
   // ======= Collect Image ======= 
	// cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
 //   cv::Mat outFrame = cv_ptr->image ;
 //   cv::imshow(OPENCV_WINDOW, outFrame);
 //   std::string Fileout = "./src/robot_rgb_classification/TrainingImages/Watermelon" + std::to_string(imgIdx++) + ".jpg"; 
 //   cv::imwrite(Fileout, outFrame);

   // ======= Neural Net Prediction =======  
   // cv::Rect boundingBox = Classifier.Detect_SSD(outFrame);
   // outFrame = Classifier.PostProcessImage(outFrame, boundingBox);
   // // Update GUI Window
   // cv::imshow(OPENCV_WINDOW, outFrame);
   
    // ======= SVM Binary Prediction =======
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    cv::Mat outFrame = cv_ptr->image;
    if (started == num_measure) {
      started = 0;
      num_measure = 0;
      start_label_int = -1;
      label_int = -1;
    }
    // Auto Mode
    if (DEBUG){
    	label_int = Classifier.Detect_SSD(outFrame);
    	std::cout<<label_int<<std::endl;
	    cv::putText(outFrame, ObjList[label_int], cv::Point2i(50,50), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 3);
	    cv::imshow(OPENCV_WINDOW, outFrame);
    }
    else{
	    //std::cout<<"Classification Node Got "<<key_pressed<<" in callback "<<std::endl;
	    if (start_label_int > -1 && started < num_measure){
        if (label_int == -1) {
          label_int = start_label_int;
        }
        started++;
        //label_int: 0: object, 1: Box, 2:Sphere...
        label_int = Classifier.Detect_SSD(outFrame, 0, label_int);
        if (started == num_measure) {
          if (ProbTable[label_int] > 0.5)
            classResults.push_back(1);
          else
            classResults.push_back(0);
        }
        std::cout<<"label_int "<<label_int<<std::endl;
	    }
	    if (start_label_int < 0){
	      cv::imshow(OPENCV_WINDOW, outFrame);
        cv::waitKey(20);
      }
	    else{
	      cv::putText(outFrame, "Classification Mode", cv::Point2i(420,50), FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
        if (label_int > 0 && label_int <= 2) {
          key_pressed = 1;
        }else if (label_int > 2 && label_int <= 8){
          key_pressed = 2;
        }else {
          key_pressed = 3;
        }
	      cv::putText(outFrame, CategoryList[key_pressed-1]+ObjList_key[label_int-1], cv::Point2i(50,50), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 3);
        string class_label;
        if (started == num_measure) {
          if (ProbTable[label_int] > 0.5)
            class_label = "Yes";
          else
            class_label = "No";
          cv::putText(outFrame, "Treasure?: " + class_label, cv::Point2i(50,100), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 3);
        }
	      cv::imshow(OPENCV_WINDOW, outFrame);
        cv::waitKey(2400);
	    }
	      // Publish Label
	    //label.data = label_int;
	    //label_pub.publish(label);
	    //key_pre = key_pressed;
	}
  cv::waitKey(5);

  }

};


int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_classification");
  ImageClassifier ic;

  ros::spin();
  return 0;
}
