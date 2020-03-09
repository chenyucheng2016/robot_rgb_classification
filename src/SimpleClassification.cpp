#include "../include/SimpleClassification.h"
SimpleClassification::SimpleClassification(){
    // pre-trained CNN file names:
	// cv::String modelConfiguration = "‎⁨../src/deploy.prototxt.txt";
	// cv::String modelBinary = "‎⁨../src/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel";
	/* 
	SSD:
		Single Shot: this means that the tasks of object localization and classification are done in a single forward pass of the network
		MultiBox: this is the name of a technique for bounding box regression developed by Szegedy et al. (we will briefly cover it shortly)
		Detector: The network is an object detector that also classifies those detected objects

	Understanding SSD MultiBox — Real-Time Object Detection In Deep Learning:
		https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab

	The PASCAL Visual Object Classes: PASCAL VOC
	Common Objects in Context: COCO
	ImageNet Large Scale Visual Recognition Challenge: ILSVRC
	*/
	cv::String modelConfiguration = "/Users/wenguai/Downloads/Cornell/Research/Classification/src/deploy.prototxt";
	cv::String modelBinary = "/Users/wenguai/Downloads/Cornell/Research/Classification/src/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel";
	//! [Initialize network]
	net = cv::dnn::readNetFromCaffe(modelConfiguration, modelBinary);
	//GEngine->AddOnScreenDebugMessage(1, 15.0f, FColor::Green, FString::Printf(TEXT("[SSD] model loaded!")));
	cout<<"Successfully Read Model"<<endl;
	
}

static cv::Mat getMean(const size_t& imageHeight, const size_t& imageWidth)
{
    using namespace std;
	cv::Mat mean;

	const int meanValues[3] = { 104, 117, 123 };
	vector<cv::Mat> meanChannels;
	for (int i = 0; i < 3; i++)
	{
		cv::Mat channel((int)imageHeight, (int)imageWidth, CV_32F, cv::Scalar(meanValues[i]));
		meanChannels.push_back(channel);
	}
	cv::merge(meanChannels, mean);
	return mean;
}

static cv::Mat preprocess(const cv::Mat& frame)
{
    using namespace std;
	// image size required as input to the CNN
	const size_t width = 300;
	const size_t height = 300;

	cv::Mat preprocessed;
	frame.convertTo(preprocessed, CV_32F);
	//SSD accepts 300x300 RGB-images
	cv::resize(preprocessed, preprocessed, cv::Size(width, height));

	cv::Mat mean = getMean(width, height);
	cv::subtract(preprocessed, mean, preprocessed);

	return preprocessed;
}

cv::Rect SimpleClassification::Detect_SSD(cv::Mat frame)
{
	// List of classes to be detected and classified (cannot be changed, since pre-trained):
	const char* classNames[] = { "background",
		"aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair",
		"cow", "diningtable", "dog", "horse",
		"motorbike", "person", "pottedplant",
		"sheep", "sofa", "train", "tvmonitor" };

/*
	if (frame.empty())
	{
		std::cerr << "Can't read the image!" << std::endl;
		exit(-1);
	}
*/
	//! [Prepare blob from image for input to CNN]
	cv::Mat preprocessedFrame = preprocess(frame);

	//Convert Mat to batch of images
	cv::Mat inputBlob = cv::dnn::blobFromImage(preprocessedFrame, 1.0f, cv::Size(), cv::Scalar(), false);

	//set the network input				
	net.setInput(inputBlob, "data");

	// compute output with a forward pass through the CNN
	cv::Mat detection = net.forward("detection_out");

	//vector<double> layersTimings;
	/*double freq = getTickFrequency() / 1000;
	double time = net.getPerfProfile(layersTimings) / freq;*/
	ostringstream ss;
	/*ss << "FPS: " << 1000 / time << " ; time: " << time << " ms";
	putText(frame, ss.str(), Point(20, 20), 0, 0.5, Scalar(0, 0, 255));*/

	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	// user parameter; should be made visible:
	float confidenceThreshold = 0.1f; // threshold to keep bounding box
	float b_conf = 0.0; // confidence score associated with the target

	// initialize bounding box at center of image. Should use desired width and height ** TODO ******************
	cv::Rect b(0,0,0,0); // bounding box of the target


	// Loop over detections
	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confidence = detectionMat.at<float>(i, 2);

		// only show output for people with confidence above threshold
		if (confidence > confidenceThreshold && detectionMat.at<float>(i, 1) == 5)
		{
			size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

			

			int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
			int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
			int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
			int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

			ss.str("");

			ss << confidence;

			cv::String conf(ss.str());

			cv::Rect object(xLeftBottom, yLeftBottom,
				xRightTop - xLeftBottom,
				yRightTop - yLeftBottom);

			// if this is the highest-confidence human, call it the target
			if (confidence > b_conf) {
				b_conf = confidence;
				b = object;
			}

			//cv::rectangle(frame, object, cv::Scalar(0, 255, 0));


			// uncomment the following lines to label bounding boxes:

			/*
			String label = String(classNames[objectClass]) + ": " + conf;

			int baseLine = 0;

			Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

			rectangle(frame, cv::Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
			Size(labelSize.width, labelSize.height + baseLine)),
			Scalar(255, 255, 255), CV_FILLED);

			putText(frame, label, Point(xLeftBottom, yLeftBottom),
			FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
			*/

		}
	}


	return b;
}

cv::Mat SimpleClassification::PostProcessImage(cv::Mat frame, cv::Rect b){
	// draw the highest-confidence bounding box on the image
	cv::rectangle(frame, b, cv::Scalar(0, 255, 0), 2, 8);
	// draw the desired bounding box, not needed here 
	// cv::rectangle(frame, cv::Rect(frame.cols / 2 - w_desired / 2, frame.rows / 2 - h_desired/2, w_desired, h_desired), cv::Scalar(0, 165, 255), 2, 8);
	return frame;
}