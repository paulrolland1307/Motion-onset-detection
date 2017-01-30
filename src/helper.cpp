/*
 * helper.cpp
 *
 *  Created on: 22 nov. 2016
 *      Author: paul
 */

#include "helper.hpp"
#include "timing.h"
#include "filters.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

vector<vector<double>> readMatrix(string filename) {
	ifstream input(filename.c_str());
	vector<vector<double>> data;

	if (input) {
		while(!input.eof()) {
			string line;
			getline(input, line);
			istringstream stm(line);
			vector<double> timeWindow;
			while(!stm.eof()) {
				double val;
				stm >> val;
				timeWindow.push_back(val);
			}
			timeWindow.pop_back();
			input >> ws;
			if (!timeWindow.empty()) data.push_back(timeWindow);
		}
	} else {
		cerr << "Couldn't open " << filename << " for reading." << endl;
	}

	return data;
}

vector<double> readVector(string filename) {
	ifstream input(filename.c_str());
	vector<double> data;

	if (input) {
		double value(0.0);
		while(input >> value) {
			data.push_back(value);
		}
	} else {
		cerr << "Couldn't open " << filename << " for reading." << endl;
	}
	return data;
}

void standardizeBy(vector<vector<double>>& data, vector<double> const& means, vector<double> const& stds) {
	for (size_t i(0); i < data.size(); ++i) {
		for (size_t j(0); j < data[0].size(); ++j) {
			data[i][j] = (data[i][j] - means[j]) / stds[j];
		}
	}
}

void standardize(vector<vector<double>>& features, string const& fileMeans, string const& fileStds) {
	vector<double> featureMeans(features[0].size(), 0.0);
	vector<double> featureStds(features[0].size(), 0.0);
	ofstream meanOut(fileMeans);
	ofstream stdOut(fileStds);
	for(size_t j(0); j < features[0].size(); ++j) {
		for (size_t i(0); i < features.size(); ++i) {
			featureMeans[j] += features[i][j];
			featureStds[j] += features[i][j]*features[i][j];
		}
		featureMeans[j] /= features.size();
		featureStds[j] = sqrt(featureStds[j]/features.size() - featureMeans[j]*featureMeans[j]);
		meanOut << featureMeans[j] << ' ';
		stdOut << featureStds[j] << ' ';
	}
	meanOut.close();
	stdOut.close();

	for (size_t i(0); i < features.size(); ++i) {
		for (size_t j(0); j < features[0].size(); ++j) {
			features[i][j] = (features[i][j] - featureMeans[j]) / featureStds[j];
		}
	}


}

vector<vector<double>> applyPCA(vector<vector<double>> const& data, vector<vector<double>> const& pcaMatrix, int numPCAVectors) {
	vector<vector<double>> result(data.size(), vector<double>(numPCAVectors, 0));
	for (int i(0); i < result.size(); ++i) {
		for (int j(0); j < result[0].size(); ++j) {
			double value(0);
			for (int k(0); k < data[0].size(); ++k) {
				value += data[i][k] * pcaMatrix[k][j];
			}
			result[i][j] = value;
		}
	}
	return result;
}

// Seperate features from labels
MLData split_data(vector<vector<double>> const& data, bool standardize) {
	MLData mldata;

	vector<vector<double>> features(data.size(), vector<double>(data[0].size()-1, 0.0));
	vector<int> labels(data.size(), 0);

	for (size_t i(0); i < features.size(); ++i) {
		for (size_t j(0); j < features[0].size(); ++j) {
			features[i][j] = data[i][j];
		}
	}

	for (size_t i(0); i < labels.size(); ++i) {
		labels[i] = data[i][data[0].size()-1];
	}

	mldata.features = features;
	mldata.labels = labels;
	return mldata;
}

double correctness(vector<int> labels, vector<int> predictions) {
	double percentage(0);
	for (size_t i(0); i < labels.size(); ++i) {
		if (labels[i] == predictions[i]) {
			percentage += 1;
		}
	}
	return percentage / labels.size();
}

void outputResult(vector<int> const& predictions) {
	ofstream out("Resources/predictions.txt");
	for (auto i : predictions) {
		out << i << endl;
	}
	out.close();
}

// Compute PCA, train the SVM model from data within file "fileTrain" and save the results
void trainSVM(string const& fileTrain, string const& fileSVM, string const& fileMeans, string const& fileStds, string const& filePCA, string const& kernel, double C, double gamma) {
	int nFeaturesPerMuscle = 3;
	int nMuscles = 12;
	int nPCAVectors = 18;

	vector<vector<double>> training = readMatrix(fileTrain);

	MLData trainData = split_data(training, true);
	standardize(trainData.features, fileMeans, fileStds);

	// Compute PCA
	if (!filePCA.empty()) {
		ofstream pcaOut(filePCA);
		vector<vector<double>> pcaMatrix = vector<vector<double>>(nFeaturesPerMuscle*nMuscles, vector<double>(nFeaturesPerMuscle*nMuscles, 0.0));
		Mat featuresMat = Mat(trainData.features.size(), trainData.features[0].size(), CV_64F);
		for (size_t i(0); i < trainData.features.size(); ++i) {
			for (size_t j(0); j < trainData.features[0].size(); ++j) {
				featuresMat.at<double>(i,j) = trainData.features[i][j];
			}
		}

		PCA pca(featuresMat, Mat(), PCA::DATA_AS_ROW);
		Mat pcaEigenvectors = pca.eigenvectors;
		for (int i(0); i < pcaMatrix.size(); ++i) {
			for (int j(0); j < pcaMatrix[0].size(); ++j) {
				pcaMatrix[i][j] = pcaEigenvectors.at<double>(i, j);
				pcaOut << pcaMatrix[i][j] << ' ';
			}
			pcaOut << endl;
		}
		pcaOut.close();

		trainData.features = applyPCA(trainData.features, pcaMatrix, nPCAVectors);
	}

	cout << trainData.labels.size() << " items in training set." << endl;

	int nbData = trainData.features.size();
	int nbFeatures = trainData.features[0].size();

	float trainingFeaturesArray[nbData][nbFeatures];
	for(size_t i(0); i < nbData; ++i) {
		for(size_t j(0); j < nbFeatures; ++j) {
			trainingFeaturesArray[i][j] = trainData.features[i][j];
		}
	}
	int *trainingLabelsArray = &trainData.labels[0];


	Mat trainingFeaturesMat(nbData, nbFeatures, CV_32FC1, trainingFeaturesArray);
	Mat trainingLabelsMat(nbData, 1, CV_32SC1, trainingLabelsArray);

	Ptr<SVM> svm;
	svm = SVM::create();
	svm->setC(C);
	svm->setType(SVM::C_SVC);
	if (kernel == "RBF") {
		svm->setKernel(SVM::RBF);
		svm->setGamma(gamma);
	}
	else {
		svm->setKernel(SVM::LINEAR);
	}
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100000, 1e-6));
	svm->train(trainingFeaturesMat, ROW_SAMPLE, trainingLabelsMat);

	svm->save(fileSVM);
}

// Evaluate the performance of an SVM (fileTest must also contain the labels)
void testSVM(string const& fileTest, string const& fileSVM, string const& fileMeans, string const& fileStds, string const& filePCA) {
	Ptr<SVM> svm = SVM::load(fileSVM);

	int nFeaturesPerMuscle = 3;
	int nMuscles = 12;
	int nPCAVectors = 18;

	vector<vector<double>> testing = readMatrix(fileTest);
	MLData testData = split_data(testing, false);
	vector<double> means = readVector(fileMeans);
	vector<double> stds = readVector(fileStds);
	standardizeBy(testData.features, means,stds);

	if (!filePCA.empty()) {
		vector<vector<double>> pcaMatrix = readMatrix(filePCA);
		testData.features = applyPCA(testData.features, pcaMatrix, nPCAVectors);
	}

	cout << testData.labels.size() << " items in testing set." << endl;

	int nbData = testData.features.size();
	int nbFeatures = testData.features[0].size();

	vector<int> predictions;
	for(size_t i(0); i < nbData; ++i) { // iterate over all samples
		float row[nbFeatures];
		for(size_t j(0); j < nbFeatures; ++j) { // record all corresponding features
			row[j] = testData.features[i][j];
		}
		Mat sampleData(1, nbFeatures, CV_32FC1, row);
		float response = svm->predict(sampleData);
		if (response >= 1.0) {
			predictions.push_back(1);
		}
		else {
			predictions.push_back(0);
		}
	}

	cout << "Percentage of correctness : " << correctness(testData.labels, predictions) << endl;

	outputResult(predictions);
}


vector<double> computeFeatures(vector<vector<double>> signal, vector<double> Bbdp, vector<double> Abdp, vector<double> Blp, vector<double> Alp, vector<double> MVC, vector<vector<double>> Xs_start) {
	vector<double> f(3*signal.size());
	vector<vector<double>> filteredSignal = preProcessTW(Bbdp, Abdp, Blp, Alp, MVC, signal, Xs_start);

	for (size_t i(0); i < signal.size(); ++i) {
		f[3*i] = computeRMS(filteredSignal[i]);
		f[3*i+1] = computeWaveformLength(filteredSignal[i]);
		f[3*i+2] = computeZeroCrossing(signal[i]); // the Zero Crossing feature is computed on raw signal
	}

	return f;
}

int computeZeroCrossing(vector<double> signal) {
	int zc(0);
	for(size_t i(1); i < signal.size(); ++i) {
		if (signal[i] * signal[i-1] < 0) {
			++zc;
		}
	}
	return zc;
}

double computeRMS(std::vector<double> const& signal) {
	double rms(0.0);
	for (auto i : signal) {
		rms += i*i;
	}
	return sqrt(rms / signal.size());
}

double computeWaveformLength(vector<double> const& signal) {
	double wfl(0.0);
	for (size_t i(1); i < signal.size(); ++i) {
		wfl += abs(signal[i] - signal[i-1]);
	}
	return wfl;
}


void computeFilterCoeff() {
	// the sample rate of the acquisition
	int fs = 1500;

	int filt_order = 7;

	// High pass frequency
	double	cof_h = 50;

	// Low pass frequency
	double cof_l = 450;

	double W_hp = 2 * cof_h / fs;
	double W_lp = 2 * cof_l / fs;

	double FrequencyBands[2] = { W_hp, W_lp };

	double *Atmp = 0;				// temporary denominator
	double *Btmp = 0;				// temporary numerator
	std::vector<double> A(2 * filt_order + 1, 0);			// denominator
	std::vector<double> B(2 * filt_order + 1, 0);			// numerator

	Atmp = ComputeDenCoeffs(filt_order, FrequencyBands[0], FrequencyBands[1]);

	for (int k = 0; k<2 * filt_order + 1; k++)
	{
		A[k] = Atmp[k];

	}

	Btmp = ComputeNumCoeffs(filt_order, FrequencyBands[0], FrequencyBands[1], Atmp);

	for (int k = 0; k<2 * filt_order + 1; k++)
	{
		B[k] = Btmp[k];
	}

	ofstream outA("Resources/A_BP.txt");
	for (int k = 0; k<2 * filt_order + 1; k++) {
		outA << A[k] << ' ';
	}
	outA.close();
	ofstream outB("Resources/B_BP.txt");
	for (int k = 0; k<2 * filt_order + 1; k++) {
		outB << B[k] << ' ';
	}
	outB.close();
}

vector<vector<double>> exctractTW(vector<vector<double>> data, int jStart, int jEnd) {
	vector<vector<double>> tw(data.size());
	for (size_t i(0); i < tw.size(); ++i) {
		for (int j(jStart); j <= jEnd; ++j) {
			tw[i].push_back(data[i][j]);
		}
	}
	return tw;
}

float predict(vector<double> features, Ptr<SVM> svm, vector<vector<double>> const& pcaMatrix, vector<double> const& means, vector<double> const& stds) {
	// Standardize the features
	for (size_t i(0); i < features.size(); ++i) {
		features[i] = (features[i] - means[i]) / stds[i];
	}

	// Apply PCA
	float featuresPCA[pcaMatrix[0].size()];
	for (int i(0); i < pcaMatrix[0].size(); ++i) {
		float val(0.0);
		for (int k(0); k < features.size(); ++k) {
			val += features[k] * pcaMatrix[k][i];
		}
		featuresPCA[i] = val;
	}

	// Predict using SVM
	Mat sample(1, pcaMatrix[0].size(), CV_32FC1, featuresPCA);

	return svm->predict(sample);
}

void startPredictions() {
	// Get the parameters for the filter
	vector<double> Abdp = readVector("Resources/A_BP.txt");
	vector<double> Bbdp = readVector("Resources/B_BP.txt");
	vector<double> Alp = readVector("Resources/A_low.txt");
	vector<double> Blp = readVector("Resources/B_low.txt");
	vector<double> MVC = readVector("Resources/MVC.txt");

	// Get the SVM model and the data needed for preprocessing
	vector<double> meansMotion = readVector("Resources/Mean_Motion.txt");
	vector<double> stdsMotion = readVector("Resources/Std_Motion.txt");
	vector<vector<double>> fullPCAMatrixMotion = readMatrix("Resources/coeff_PCA_Motion.txt");
	Ptr<SVM> svmMotion = SVM::load("Resources/SVM_Motion.xml");

	vector<double> meansGrasp = readVector("Resources/Mean_Grasp.txt");
	vector<double> stdsGrasp = readVector("Resources/Std_Grasp.txt");
	vector<vector<double>> fullPCAMatrixGrasp = readMatrix("Resources/coeff_PCA_Grasp.txt");
	Ptr<SVM> svmGrasp = SVM::load("Resources/SVM_Grasp.xml");

	// Only keep a limited number of PCA eigenvectors
	int numPCAVectors = 18;
	int nFeatures = 12*3;
	int nMuscles = 12;
	vector<vector<double>> pcaMatrixMotion(nFeatures, vector<double>(numPCAVectors, 0));
	vector<vector<double>> pcaMatrixGrasp(nFeatures, vector<double>(numPCAVectors, 0));
	for (size_t i(0); i < pcaMatrixGrasp.size(); ++i) {
		for (size_t j(0); j < pcaMatrixGrasp[0].size(); ++j) {
			pcaMatrixMotion[i][j] = fullPCAMatrixMotion[i][j];
			pcaMatrixGrasp[i][j] = fullPCAMatrixGrasp[i][j];
		}
	}

	// Get the data on which the models will be applied
	vector<vector<double>> fullRawData = readMatrix("Resources/testingRaw.txt");
	vector<vector<double>> rawData; //nMuscles x T matrix
	for (int i(0); i < nMuscles; ++i) {
		rawData.push_back(fullRawData[i]);
	}

	// Some variables we will need
	bool motionDetection(true);
	double detectedStartMotion(-1);
	double detectedEndMotion(-1);
	double detectedStartGrasp(-1);
	double detectedEndGrasp(-1);
	double meanComputationTime = 0.0;
	double meanFeatureComputationTime = 0.0;
	double meanPredictionTime = 0.0;
	int numberTW = 0;
	int count(0);
	vector<vector<double>> timeWindow_prev = vector<vector<double>>(nMuscles, vector<double>(1, 0.0));


	// Parameters of for the detection
	int timeWindowsToWaitMotion_Start = 6;
	int timeWindowsToWaitMotion_End = 6;
	int timeWindowsToWaitGrasp_Start = 6;
	int timeWindowsToWaitGrasp_End = 7;

	// Parameters for time discretization
	int fs = 1500;
	double timeWindowDuration = 0.05;
	int indexStart = 0;
	int indexEnd = fs * timeWindowDuration - 1;

	cout << endl << "Apply the models to real problem." << endl;
	while(indexEnd < rawData[0].size()) {
		vector<vector<double>> timeWindow = exctractTW(rawData, indexStart, indexEnd);

		double startPrediction = second();
		vector<double> features = computeFeatures(timeWindow, Bbdp, Abdp, Blp, Alp, MVC, timeWindow_prev);
		meanFeatureComputationTime += second() - startPrediction;
		float prediction;
		if (motionDetection) { // Apply motion detection model
			prediction = predict(features, svmMotion, pcaMatrixMotion, meansMotion, stdsMotion);
			if (detectedStartMotion < 0) {
				if (prediction >= 1.0) ++ count;
				else count = max(count-1, 0);
			} else {
				if (prediction <= 0.0) ++ count;
				else count = max(count-1, 0);
			}
			if (detectedStartMotion < 0 and count >= timeWindowsToWaitMotion_Start) {
				detectedStartMotion = indexEnd / double(fs);
				motionDetection = false; // switch to grasping detection once motion start has been detected
				count = 0;
				//cout << "Motion start detected" << endl;
			}
			if (detectedStartMotion >= 0 and detectedEndMotion < 0 and count >= timeWindowsToWaitMotion_End) {
				detectedEndMotion = indexEnd / double(fs);
				count = 0;
				//cout << "Motion end detected" << endl;
			}
		}
		else { // Apply grasping detection model
			prediction = predict(features, svmGrasp, pcaMatrixGrasp, meansGrasp, stdsGrasp);
			if (detectedStartGrasp< 0) {
				if (prediction >= 1.0) ++ count;
				else count = max(count-1, 0);
			} else {
				if (prediction <= 0.0) ++ count;
				else count = max(count-1, 0);
			}
			if (detectedStartGrasp < 0 and count >= timeWindowsToWaitGrasp_Start) {
				detectedStartGrasp = indexEnd / double(fs);
				count = 0;
				//cout << "Grasp start detected" << endl;
			}
			if (detectedStartGrasp >= 0 and detectedEndGrasp < 0 and count >= timeWindowsToWaitGrasp_End) {
				detectedEndGrasp = indexEnd / double(fs);
				motionDetection = true; // switch to motion detection once grasping end has been detected
				count = 0;
				//cout << "Grasp end detected" << endl;
			}
		}
		double timePrediction = second() - startPrediction;
		meanComputationTime += timePrediction;

		timeWindow_prev = timeWindow; // we record the last time window for the filter

		indexStart += fs * timeWindowDuration / 2;
		indexEnd += fs * timeWindowDuration / 2;
		++numberTW;
	}
	meanComputationTime /= numberTW;
	meanFeatureComputationTime /= numberTW;

	cout << "Detected start of motion : " << detectedStartMotion << endl;
	cout << "Detected end of motion : " << detectedEndMotion << endl;
	cout << "Detected start of grasp : " << detectedStartGrasp << endl;
	cout << "Detected end of grasp : " << detectedEndGrasp << endl;
	cout << endl;

	cout << "Mean features computation time : " << meanFeatureComputationTime << " s" << endl;
	cout << "Mean prediction time : " << meanComputationTime - meanFeatureComputationTime << " s" << endl;
	cout << "Mean computation time : " << meanComputationTime << " s" << endl;
}


void trainAllModels() {
	// Define the files into which the data will be saved
	string detectionType = "Motion";
	string fileSVM = "Resources/SVM_"+detectionType+".xml";
	string filePCA = "Resources/coeff_PCA_"+detectionType+".txt";
	//string filePCA = "Resources/identity.txt";
	string fileMeans = "Resources/Mean_"+detectionType+".txt";
	string fileStds = "Resources/Std_"+detectionType+".txt";
	string fileTrain = "Resources/training_"+detectionType+".txt";
	string fileTest = "Resources/testing_"+detectionType+".txt";
	string kernel = "linear";
	double C(1.0);
	double startTrain = second();
	cout << "Training for motion detection" << endl;
	trainSVM(fileTrain, fileSVM, fileMeans, fileStds, filePCA, kernel, C);
	cout << "Training time for motion : " << second() - startTrain << " s" << endl;
	testSVM(fileTest, fileSVM, fileMeans, fileStds, filePCA);

	cout << endl << "Training for grasping detection" << endl;
	detectionType = "Grasp";
	fileSVM = "Resources/SVM_"+detectionType+".xml";
	filePCA = "Resources/coeff_PCA_"+detectionType+".txt";
	fileMeans = "Resources/Mean_"+detectionType+".txt";
	fileStds = "Resources/Std_"+detectionType+".txt";
	fileTrain = "Resources/training_"+detectionType+".txt";
	fileTest = "Resources/testing_"+detectionType+".txt";
	kernel = "linear";
	C = 1.0;
	double gamma = 6.0;
	startTrain = second();
	trainSVM(fileTrain, fileSVM, fileMeans, fileStds, filePCA, kernel, C, gamma);
	//trainSVM(fileTrain, fileSVM, fileMeans, fileStds, "", kernel, C, gamma);
	cout << "Training time for grasp : " << second() - startTrain << " s" << endl;
	testSVM(fileTest, fileSVM, fileMeans, fileStds, filePCA);
}
