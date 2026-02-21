#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

// Activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid function
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Adjustable parameters -----------------
double learningRate = 0.1;       // Learning rate
uint8_t hiddenNeuronCount = 4;   // Number of hidden neurons
uint32_t trainingEpochs = 30000; // Training iterations
// ---------------------------------------

class NeuralNetwork {
public:
    vector<double> inputLayer;
    vector<double> hiddenLayer;
    vector<double> outputLayer;

    vector<vector<double>> weightsInputHidden;
    vector<vector<double>> weightsHiddenOutput;

    NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        inputLayer.resize(inputSize);
        hiddenLayer.resize(hiddenSize);
        outputLayer.resize(outputSize);

        weightsInputHidden.resize(inputSize, vector<double>(hiddenSize));
        weightsHiddenOutput.resize(hiddenSize, vector<double>(outputSize));

        srand(time(0));
        // Initialize weights between input and hidden layer
        for (int i = 0; i < inputSize; i++)
            for (int j = 0; j < hiddenSize; j++)
                weightsInputHidden[i][j] = ((double)rand() / RAND_MAX) - 0.5;

        // Initialize weights between hidden and output layer
        for (int i = 0; i < hiddenSize; i++)
            for (int j = 0; j < outputSize; j++)
                weightsHiddenOutput[i][j] = ((double)rand() / RAND_MAX) - 0.5;
    }

    // Forward propagation
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

    // Backpropagation training
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

        // Update weights hidden → output
        for (int j = 0; j < hiddenLayer.size(); j++) {
            for (int k = 0; k < outputLayer.size(); k++) {
                double delta = learningRate * outputErrors[k] * sigmoid_derivative(outputLayer[k]) * hiddenLayer[j];
                weightsHiddenOutput[j][k] += delta;
            }
        }

        // Update weights input → hidden
        for (int i = 0; i < inputLayer.size(); i++) {
            for (int j = 0; j < hiddenLayer.size(); j++) {
                double delta = learningRate * hiddenErrors[j] * sigmoid_derivative(hiddenLayer[j]) * inputLayer[i];
                weightsInputHidden[i][j] += delta;
            }
        }
    }
};

// -------------------- Helper Functions --------------------

// Training function
void trainNetwork(NeuralNetwork& nn, const vector<vector<double>>& inputs,
    const vector<vector<double>>& outputs, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < inputs.size(); i++) {
            nn.train(inputs[i], outputs[i]);
        }
    }
}

// Testing function
void testNetwork(NeuralNetwork& nn, const vector<vector<double>>& inputs,
    const string& gateName) {
    cout << gateName << " Test Results:" << endl;
    for (int i = 0; i < inputs.size(); i++) {
        nn.forward(inputs[i]);
        cout << inputs[i][0] << " " << gateName << " " << inputs[i][1]
            << " = " << nn.outputLayer[0] << endl;
    }
    cout << endl;
}

// -------------------- main --------------------
int main() {
    vector<vector<double>> trainingInputs = {
        {0,0}, {0,1}, {1,0}, {1,1}
    };

    // XOR Gate
    NeuralNetwork nn_xor(2, hiddenNeuronCount, 1);
    vector<vector<double>> trainingOutputsXOR = {
        {0}, {1}, {1}, {0}
    };
    trainNetwork(nn_xor, trainingInputs, trainingOutputsXOR, trainingEpochs);
    testNetwork(nn_xor, trainingInputs, "XOR");

    // NAND Gate
    NeuralNetwork nn_nand(2, hiddenNeuronCount, 1);
    vector<vector<double>> trainingOutputsNAND = {
        {1}, {1}, {1}, {0}
    };
    trainNetwork(nn_nand, trainingInputs, trainingOutputsNAND, trainingEpochs);
    testNetwork(nn_nand, trainingInputs, "NAND");

    return 0;
}
