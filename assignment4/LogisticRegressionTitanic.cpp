#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
using namespace std;

// Function Declarations
vector<double> logOdds(vector<double>, vector<int>, int);
vector<double> sigmoid(vector<double>, int);

int main(){

    /***************************** Load Data ************************************/
    ifstream inFS;
    string line;
    string entry_in, pclass_in, survived_in, sex_in, age_in;
    const int MAX_LEN = 1500;
    vector<string> entry(MAX_LEN);      // 'column 1' of titanic_project.csv 
    vector<int> pclass(MAX_LEN);        // 'column 2' of ""
    vector<int> survived(MAX_LEN);      // 'column 3' of "" Note: survived is target variable
    vector<int> sex(MAX_LEN);           // 'column 4' of "" Note: sex is predictor variable
    vector<double> age(MAX_LEN);        // 'column 5' of ""

    // Open the file & send message if there's an error
    cout << "Opening file titanic_project.csv." << endl;

    inFS.open("titanic_project.csv");
    if(!inFS){
        cout << "Could not open file titanic_project.csv." << endl;
        return 1; // 1 incidates error
    }

    // Get header line (column information)
    // first line of file: "","pclass","survived","sex","age"
    getline(inFS, line);

    // initialize number of observations
    int numObservations = 0;
    while(inFS.good()){

        // get values of each variable
        getline(inFS, entry_in, ',');
        getline(inFS, pclass_in, ',');
        getline(inFS, survived_in, ',');
        getline(inFS, sex_in, ',');
        getline(inFS, age_in, '\n');

        // push variable values onto vectors
        entry.at(numObservations) = entry_in;
        pclass.at(numObservations) = stoi(pclass_in);
        survived.at(numObservations) = stoi(survived_in);
        sex.at(numObservations) = stoi(sex_in);
        age.at(numObservations) = stof(age_in);

        numObservations++;
    }
    cout << "Total number of observations (rows): " << numObservations << endl << endl;

    // Resize vectors
    entry.resize(numObservations);
    pclass.resize(numObservations);
    survived.resize(numObservations);
    sex.resize(numObservations);
    age.resize(numObservations);

    inFS.close(); // Done with file, so close it

    /************************ Training Phase *************************/

    // Initialization
    int train = 800;
    vector<double> weights = {1.0, 1.0};
    vector<double> prob_vector(train);
    vector<double> error_vector(train);
    vector<int> train_labels(train);

    cout << "Number of rows for training: " << train << endl;

    // Copy over first 800 values of survived
    // vector to train_labels for training
    for(int i = 0; i < train; i++){
        train_labels.at(i) = survived.at(i);
    }

    // Computing for weight coefficients (gradient descent)
    double learning_rate = 0.01;
    const int MAX_ITERATIONS = 500;

    auto trainStart = chrono::high_resolution_clock::now(); // start train time
    int i = 0;
    while(i < MAX_ITERATIONS){

        // Calculate vector of predictions for training data
        // based on current values of weight coefficients
        prob_vector = sigmoid(logOdds(weights, sex, train), train);
        // Calulate vector of errors of the predictions
        for(int j = 0; j< train; j++){
            error_vector.at(j) = train_labels.at(j) - prob_vector.at(j);
        }
        
        // Calculate new coefficient values for weights based on error in prediction
        for(int j = 0; j < train; j++){
            double prob = prob_vector.at(j) * (1 - prob_vector.at(j));
            weights.at(0) = weights.at(0) + learning_rate * error_vector.at(j) * prob * 1.0;
            weights.at(1) = weights.at(1) + learning_rate * error_vector.at(j) * prob * sex.at(j);
        }
        i++;
    }
    auto trainEnd = chrono::high_resolution_clock::now(); // end train time

    // Output Train Time
    cout << "Train time: " << chrono::duration_cast<chrono::nanoseconds>(trainEnd - trainStart).count() << " nanoseconds" << endl;

    // Output coefficients for weights
    cout << "Coefficients: w0 =  " << weights.at(0) << "; w1 =  " << weights.at(1) << endl << endl;

    /**************************** Testing Phase ************************************/

    // Use remaining data for testing (index 800 to max observations)
    int test = numObservations - train;
    vector<int> test_labels(test);
    vector<int> test_predictor(test);

    cout << "Number of rows for testing: " << test << endl;

    // Copy over remaining values of survived
    // vector to tesg_labels for testing
    for(int i = train; i < numObservations; i++){
        test_labels.at(i - train) = survived.at(i);
        test_predictor.at(i - train) = sex.at(i);
    }

    // Predict using generated coefficient weights
    double e = 2.71828;
    double logOdds = 0.0;
    double probability = 0.0;
    vector<int> predictions(test);
    for(int i = 0; i < test; i++){
        logOdds = weights.at(0) + weights.at(1) * test_predictor.at(i);
        probability = 1.0 / (1.0 + pow(e,-logOdds));
        if(probability > 0.5){
            predictions.at(i) = 1;
        }
        else{
            predictions.at(i) = 0;
        }
    }

    /************************ Test Metrics ****************************/
    double accuracy = -1.0, sensitivity = -1.0, specificity = -1.0;

    int truePos = 0, trueNeg = 0, falsePos = 0, falseNeg = 0;
    for(int i = 0; i < test; i++){
        if(predictions.at(i) == test_labels.at(i)){
            if(predictions.at(i) == 1){
                truePos++;
            }
            else{
                trueNeg++;
            }
        }
        else{
            if(predictions.at(i) == 1){
                falsePos++;
            }
            else{
                falseNeg++;
            }
        }
    }
    int trueTotal = truePos + trueNeg;

    accuracy = (double)trueTotal / test;
    sensitivity = (double)truePos / (truePos + falseNeg);
    specificity = (double)trueNeg / (trueNeg + falsePos);

    cout << "Test metrics: Accuracy = " << accuracy << "; sensitivity = " << sensitivity << "; specificity = " << specificity << endl;
        
} // End of main()

// Function for computing and returning a vector of log odds
vector<double> logOdds(vector<double> weights, vector<int> predictor, int train_size){

    // train_size = 800
    vector<double> p(train_size);

    for(int i = 0; i < train_size; i++){
        p.at(i) = weights.at(0) + weights.at(1) * predictor.at(i);
    }

    return p;
}

// Function for computing and returning a vector of
// predictions using the sigmoid formula = 1 / (1 + e^-z) 
vector<double> sigmoid(vector<double> z, int train_size){
    
    vector<double> prediction(train_size);
    
    double e = 2.71828;
    for(int i = 0; i < train_size; i++){
        prediction.at(i) = 1.0 / (1.0 + pow(e,-(z.at(i))));
    }

    return prediction;
}