#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

class NeuralNetwork {
public:
    vector<double> inputLayer;
    vector<double> hiddenLayer;
    vector<double> outputLayer;

    vector<vector<double>> weightsInputHidden;
    vector<vector<double>> weightsHiddenOutput;

    double learningRate = 0.1;

    NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        inputLayer.resize(inputSize);
        hiddenLayer.resize(hiddenSize);
        outputLayer.resize(outputSize);

        weightsInputHidden.resize(inputSize, vector<double>(hiddenSize));
        weightsHiddenOutput.resize(hiddenSize, vector<double>(outputSize));

        srand(time(0));
        for (int i = 0; i < inputSize; i++)
            for (int j = 0; j < hiddenSize; j++)
                weightsInputHidden[i][j] = ((double)rand() / RAND_MAX) - 0.5;

        for (int i = 0; i < hiddenSize; i++)
            for (int j = 0; j < outputSize; j++)
                weightsHiddenOutput[i][j] = ((double)rand() / RAND_MAX) - 0.5;
    }

    void forward(const vector<double>& inputs) {
        for (int i = 0; i < inputLayer.size(); i++)
            inputLayer[i] = inputs[i];

        for (int j = 0; j < hiddenLayer.size(); j++) {
            double sum = 0.0;
            for (int i = 0; i < inputLayer.size(); i++)
                sum += inputLayer[i] * weightsInputHidden[i][j];
            hiddenLayer[j] = sigmoid(sum);
        }

        for (int k = 0; k < outputLayer.size(); k++) {
            double sum = 0.0;
            for (int j = 0; j < hiddenLayer.size(); j++)
                sum += hiddenLayer[j] * weightsHiddenOutput[j][k];
            outputLayer[k] = sigmoid(sum);
        }
    }

    void train(const vector<double>& inputs, const vector<double>& targets) {
        forward(inputs);

        vector<double> outputErrors(outputLayer.size());
        for (int k = 0; k < outputLayer.size(); k++)
            outputErrors[k] = targets[k] - outputLayer[k];

        vector<double> hiddenErrors(hiddenLayer.size());
        for (int j = 0; j < hiddenLayer.size(); j++) {
            double error = 0.0;
            for (int k = 0; k < outputLayer.size(); k++)
                error += outputErrors[k] * weightsHiddenOutput[j][k];
            hiddenErrors[j] = error;
        }

        for (int j = 0; j < hiddenLayer.size(); j++) {
            for (int k = 0; k < outputLayer.size(); k++) {
                double delta = learningRate * outputErrors[k] * sigmoid_derivative(outputLayer[k]) * hiddenLayer[j];
                weightsHiddenOutput[j][k] += delta;
            }
        }

        for (int i = 0; i < inputLayer.size(); i++) {
            for (int j = 0; j < hiddenLayer.size(); j++) {
                double delta = learningRate * hiddenErrors[j] * sigmoid_derivative(hiddenLayer[j]) * inputLayer[i];
                weightsInputHidden[i][j] += delta;
            }
        }
    }
};

int main() {
    NeuralNetwork nn(2, 16, 1);

    vector<vector<double>> trainingInputs = {
        {0,0}, {0,1}, {1,0}, {1,1}
    };
    vector<vector<double>> trainingOutputs = {
        {0}, {1}, {1}, {0}
    };

    for (int epoch = 0; epoch < 35000; epoch++) {
        for (int i = 0; i < trainingInputs.size(); i++) {
            nn.train(trainingInputs[i], trainingOutputs[i]);
        }
    }

    cout << "XOR Test Results:" << endl;
    for (int i = 0; i < trainingInputs.size(); i++) {
        nn.forward(trainingInputs[i]);
        cout << trainingInputs[i][0] << " XOR " << trainingInputs[i][1] << " = "
            << nn.outputLayer[0] << endl;
    }

    return 0;
}
