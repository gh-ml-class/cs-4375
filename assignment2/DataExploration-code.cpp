#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>

using namespace std;

double sum(vector<double> &vec)
{
	double result = 0.0;
	for (auto &x : vec) {
		result += x;
	}
	return result;
}

double mean(vector<double> &vec)
{
	return sum(vec) / static_cast<double>(vec.size());
}

double median(vector<double> &vec)
{
	sort(vec.begin(), vec.end());
	if (vec.size() % 2 == 0) {
		return (vec[vec.size() / 2 - 1] + vec[vec.size() / 2]) / 2.0;
	} else {
		return vec[vec.size() / 2];
	}
}

pair<double, double> range(vector<double> &vec)
{
	double min = 9999999.0;
	double max = -9999999.0;
	for (auto &x : vec) {
		if (x < min) {
			min = x;
		}
		if (x > max) {
			max = x;
		}
	}
	return make_pair(min, max);
}

void print_stats(vector<double> &vec)
{
	cout << "Sum: " << sum(vec) << endl;
	cout << "Mean: " << mean(vec) << endl;
	cout << "Median: " << median(vec) << endl;
	auto r = range(vec);
	cout << "Range: " << r.first << ", " << r.second << endl;
}

double covar(vector<double> &rm, vector<double> &medv)
{
	double rmMean = mean(rm);
	double medvMean = mean(medv);
	double s = 0.0;
	for (size_t i = 0; i < rm.size(); ++i) {
		s += (rm[i] - rmMean) * (medv[i] - medvMean);
	}
	return s / (static_cast<double>(rm.size()) - 1);
}

double stdev(vector<double> &vec)
{
	double s = 0.0;
	double m = mean(vec);
	for (auto &x : vec) {
		s += pow(x - m, 2.0);
	}
	return sqrt((s / static_cast<double>(vec.size())));
}

double cor(vector<double> &rm, vector<double> &medv)
{
	double rmSd = stdev(rm);
	double medvSd = stdev(medv);
	return covar(rm, medv) / (rmSd * medvSd);
}

int main(int argc, char *argv[])
{
	ifstream inFS;
	string line;
	string rm_in, medv_in;
	const int MAX_LEN = 1000;
	vector<double> rm(MAX_LEN);
	vector<double> medv(MAX_LEN);

	cout << "Opening file Boston.csv." << endl;

	inFS.open("Boston.csv");
	if (!inFS.is_open()) {
		cout << "Could not open file Boston.csv." << endl;
		return 1;
	}

	cout << "Reading line 1" << endl;
	getline(inFS, line);

	cout << "heading: " << line << endl;

	int numObservations = 0;
	while (inFS.good()) {
		getline(inFS, rm_in, ',');
		getline(inFS, medv_in, '\n');

		rm.at(numObservations) = stof(rm_in);
		medv.at(numObservations) = stof(medv_in);

		++numObservations;
	}

	rm.resize(numObservations);
	medv.resize(numObservations);

	cout << "new length " << rm.size() << endl;

	cout << "Closing file Boston.csv." << endl;
	inFS.close();

	cout << "Number of records: " << numObservations << endl;

	cout << "\nStats for rm" << endl;
	print_stats(rm);

	cout << "\nStats for medv" << endl;
	print_stats(medv);

	cout << "\n Covariance = " << covar(rm, medv) << endl;

	cout << "\n Correlation = " << cor(rm, medv) << endl;

	cout << "\nProgram terminated.";

	return 0;
}