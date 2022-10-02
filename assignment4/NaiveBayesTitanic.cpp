#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>

const double PI = 3.1415926535897932384626433832795028841971693993751058209;
const int NUM_TRAIN = 800;

struct Passenger {
	int id = -1;
	double pclass = -1.0, survived = -1.0, sex = -1.0, age = -1.0;
};

bool readInputFile(std::vector<Passenger>& trainSurvived, std::vector<Passenger>& trainDied, std::vector<Passenger>& test) {
	std::ifstream inFile("titanic_project.csv");
	if (!inFile.is_open()) {
		return false;
	}
	std::string heading;
	getline(inFile, heading);
	int count = 0;
	while (inFile.good()) {
		Passenger curEntry;
		std::string curStr;
		getline(inFile, curStr, ',');
		std::string curSubstr = curStr.substr(1, curStr.size() - 2);
		curEntry.id = std::stoi(curSubstr);
		getline(inFile, curStr, ',');
		curEntry.pclass = std::stof(curStr);
		getline(inFile, curStr, ',');
		curEntry.survived = std::stof(curStr);
		getline(inFile, curStr, ',');
		curEntry.sex = std::stof(curStr);
		getline(inFile, curStr, '\n');
		curEntry.age = std::stof(curStr);
		if (count < NUM_TRAIN) {
			if (curEntry.survived > 0) {
				trainSurvived.push_back(curEntry);
			}
			else {
				trainDied.push_back(curEntry);
			}
		} else {
			test.push_back(curEntry);
		}
		++count;
	}
	return true;
}

struct PassengerSummary {
	double prior = -1.0;
	double pclassMean = -1.0, pclassVar = -1.0;
	double sexMean = -1.0, sexVar = -1.0;
	double ageMean = -1.0, ageVar = -1.0;
};

void computeSummary(std::vector<Passenger>& data, PassengerSummary &summary) {
	double count = static_cast<double>(data.size());
	summary.prior = count / static_cast<double>(NUM_TRAIN);

	summary.pclassMean = 0.0;
	summary.sexMean = 0.0;
	summary.ageMean = 0.0;
	for (auto it = data.begin(); it != data.end(); ++it) {
		summary.pclassMean += it->pclass;
		summary.sexMean += it->sex;
		summary.ageMean += it->age;
	}
	summary.pclassMean /= count;
	summary.sexMean /= count;
	summary.ageMean /= count;

	summary.pclassVar = 0.0;
	summary.sexVar = 0.0;
	summary.ageVar = 0.0;
	for (auto it = data.begin(); it != data.end(); ++it) {
		summary.pclassVar += pow((it->pclass - summary.pclassMean), 2.0);
		summary.sexVar += pow((it->sex - summary.sexMean), 2.0);
		summary.ageVar += pow((it->age - summary.ageMean), 2.0);
	}
	summary.pclassVar /= count;
	summary.sexVar /= count;
	summary.ageVar /= count;
}

double computeFeatureProbability(double x, double mean, double variance) {
	return exp(-pow((x - mean), 2.0) / (variance * 2.0)) / sqrt(variance * PI * 2.0);
}

void predict(std::vector<Passenger> &testData, std::vector<Passenger> &predicted, PassengerSummary summarySurvived, PassengerSummary summaryDied) {
	for (auto it = testData.begin(); it != testData.end(); ++it) {
		Passenger cur;
		cur.id = it->id;
		cur.pclass = it->pclass;
		cur.sex = it->sex;
		cur.age = it->age;
		
		double pSurvived = summarySurvived.prior *
			computeFeatureProbability(cur.pclass, summarySurvived.pclassMean, summarySurvived.pclassVar) *
			computeFeatureProbability(cur.sex, summarySurvived.sexMean, summarySurvived.sexVar) *
			computeFeatureProbability(cur.pclass, summarySurvived.ageMean, summarySurvived.ageVar);

		double pDied = summaryDied.prior *
			computeFeatureProbability(cur.pclass, summaryDied.pclassMean, summaryDied.pclassVar) *
			computeFeatureProbability(cur.sex, summaryDied.sexMean, summaryDied.sexVar) *
			computeFeatureProbability(cur.pclass, summaryDied.ageMean, summaryDied.ageVar);

		if (pSurvived > pDied) {
			cur.survived = 1.0;
		}
		else {
			cur.survived = 0.0;
		}

		predicted.push_back(cur);
	}
}

void computeTestMetrics(std::vector<Passenger>& testData, std::vector<Passenger>& predicted, double &accuracy, double &sensitivity, double &specificity) {
	double truePos = 0.0, trueNeg = 0.0, falsePos = 0.0, falseNeg = 0.0;
	for (size_t i = 0; i < testData.size(); ++i) {
		if (predicted[i].survived == testData[i].survived) {
			if (predicted[i].survived) {
				truePos += 1.0;
			}
			else {
				trueNeg += 1.0;
			}
		}
		else {
			if (predicted[i].survived) {
				falsePos += 1.0;
			}
			else {
				falseNeg += 1.0;
			}
		}
	}
	double trueTotal = truePos + trueNeg;

	double count = static_cast<double>(testData.size());
	accuracy = trueTotal / count;
	sensitivity = truePos / (truePos + falseNeg);
	specificity = trueNeg / (trueNeg + falsePos);
}

int main(int argc, char* argv[]) {
	std::vector<Passenger> trainSurvived, trainDied, testData;
	if (!readInputFile(trainSurvived, trainDied, testData)) {
		std::cerr << "Unable to read titanic_project.csv" << std::endl;
		return -1;
	}
	std::cout << "Read " << trainSurvived.size() + trainDied.size() << " train rows and " << testData.size() << " test rows." << std::endl << std::endl;

	PassengerSummary summarySurvived, summaryDied;
	auto trainStart = std::chrono::high_resolution_clock::now();
	computeSummary(trainSurvived, summarySurvived);
	computeSummary(trainDied, summaryDied);
	auto trainEnd = std::chrono::high_resolution_clock::now();
	std::cout << "Survived" << std::endl << "====================" << std::endl;
	std::cout << "A-priori: " << summarySurvived.prior << std::endl;
	std::cout << "pclass: Mean = " << summarySurvived.pclassMean << "; variance = " << summarySurvived.pclassVar << std::endl;
	std::cout << "sex: Mean = " << summarySurvived.sexMean << "; variance = " << summarySurvived.sexVar << std::endl;
	std::cout << "age: Mean = " << summarySurvived.ageMean << "; variance = " << summarySurvived.ageVar << std::endl << std::endl;
	std::cout << "Died" << std::endl << "====================" << std::endl;
	std::cout << "A-priori: " << summaryDied.prior << std::endl;
	std::cout << "pclass: Mean = " << summaryDied.pclassMean << "; variance = " << summaryDied.pclassVar << std::endl;
	std::cout << "sex: Mean = " << summaryDied.sexMean << "; variance = " << summaryDied.sexVar << std::endl;
	std::cout << "age: Mean = " << summaryDied.ageMean << "; variance = " << summaryDied.ageVar << std::endl << std::endl;
	std::cout << "Train time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(trainEnd - trainStart).count() << " nanoseconds" << std::endl << std::endl;

	std::vector<Passenger> predicted;
	predict(testData, predicted, summarySurvived, summaryDied);

	double accuracy = -1.0, sensitivity = -1.0, specificity = -1.0;
	computeTestMetrics(testData, predicted, accuracy, sensitivity, specificity);
	std::cout << "Test metrics: Accuracy = " << accuracy << "; sensitivity = " << sensitivity << "; specificity = " << specificity << std::endl;

	return 0;
}
