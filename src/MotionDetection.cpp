#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include "helper.hpp"
#include "timing.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

int main(int, char**)
{
	// Compute and save the coefficients required for filtering
	computeFilterCoeff();

	// Train, test and save SVM models for both motion and grasping detection
	trainAllModels();

	// Apply the trained models to some concrete problem, where only given the raw data
	startPredictions();


    return 0;
}
