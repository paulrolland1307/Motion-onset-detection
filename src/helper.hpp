/*
 * helper.hpp
 *
 *  Created on: 21 nov. 2016
 *      Author: paul
 */

#ifndef HELPER_HPP_
#define HELPER_HPP_

#include <string>
#include <vector>
#include <array>
#include <cv.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

struct MLData {
	std::vector<std::vector<double>> features;
	std::vector<int> labels;
};

std::vector<std::vector<double>> readMatrix(std::string filename);
std::vector<double> readVector(std::string filename);

MLData split_data(std::vector<std::vector<double>> const& data, bool standardize);

void standardizeBy(std::vector<std::vector<double>>& data, std::vector<double> const& means, std::vector<double> const& stds);
void standardize(std::vector<std::vector<double>>& features, std::string const& fileMeans, std::string const& fileStds); // standardize and stores means and stds
std::vector<std::vector<double>> applyPCA(std::vector<std::vector<double>> const& data, std::vector<std::vector<double>> const& pcaMatrix, int numPCAVectors);

double correctness(std::vector<int> labels, std::vector<int> predictions);

void outputResult(std::vector<int> const& predictions);

void trainSVM(std::string const& fileTrain, std::string const& fileSVM, std::string const& fileMeans, std::string const& fileStds, std::string const& filePCA, std::string const& kernel, double C, double gamma=1.0);
void testSVM(std::string const& fileTest, std::string const& fileSVM, std::string const& fileMeans, std::string const& fileStds, std::string const& filePCA);
float predict(std::vector<double> features, cv::Ptr<cv::ml::SVM> svm, std::vector<std::vector<double>> const& pcaMatrix, std::vector<double> const& means, std::vector<double> const& stds);

std::vector<double> computeFeatures(std::vector<std::vector<double>> signal, std::vector<double> Bbdp, std::vector<double> Abdp, std::vector<double> Blp, std::vector<double> Alp, std::vector<double> MVC, std::vector<std::vector<double>> Xs_start);
int computeZeroCrossing(std::vector<double> signal);
double computeRMS(std::vector<double> const& signal);
double computeWaveformLength(std::vector<double> const& signal);


std::vector<std::vector<double>> exctractTW(std::vector<std::vector<double>> data, int jStart, int jEnd);


void computeFilterCoeff();
void trainAllModels();
void startPredictions();




#endif /* HELPER_HPP_ */
