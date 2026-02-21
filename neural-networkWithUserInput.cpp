#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <stack>
#include <string>

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
uint32_t trainingEpochs = 10000; // Training iterations
// ---------------------------------------

class NeuralNetwork {
public:
    vector<double> inputLayer;
    vector<double> hiddenLayer;
    vector<double> outputLayer;

    vector<vector<double>> weightsInputHidden;
    vector<vector<double>> weightsHiddenOutput;

    // Bias terms
    vector<double> biasHidden;
    vector<double> biasOutput;

    NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        inputLayer.resize(inputSize);
        hiddenLayer.resize(hiddenSize);
        outputLayer.resize(outputSize);

        weightsInputHidden.resize(inputSize, vector<double>(hiddenSize));
        weightsHiddenOutput.resize(hiddenSize, vector<double>(outputSize));

        biasHidden.resize(hiddenSize);
        biasOutput.resize(outputSize);

        srand(time(0));
        // Initialize weights and biases
        for (int i = 0; i < inputSize; i++)
            for (int j = 0; j < hiddenSize; j++)
                weightsInputHidden[i][j] = ((double)rand() / RAND_MAX) - 0.5;

        for (int j = 0; j < hiddenSize; j++)
            biasHidden[j] = ((double)rand() / RAND_MAX) - 0.5;

        for (int j = 0; j < hiddenSize; j++)
            for (int k = 0; k < outputSize; k++)
                weightsHiddenOutput[j][k] = ((double)rand() / RAND_MAX) - 0.5;

        for (int k = 0; k < outputSize; k++)
            biasOutput[k] = ((double)rand() / RAND_MAX) - 0.5;
    }

    // Forward propagation
    void forward(const vector<double>& inputs) {
        for (int i = 0; i < inputLayer.size(); i++)
            inputLayer[i] = inputs[i];

        for (int j = 0; j < hiddenLayer.size(); j++) {
            double sum = biasHidden[j];
            for (int i = 0; i < inputLayer.size(); i++)
                sum += inputLayer[i] * weightsInputHidden[i][j];
            hiddenLayer[j] = sigmoid(sum);
        }

        for (int k = 0; k < outputLayer.size(); k++) {
            double sum = biasOutput[k];
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

        // Update biases output
        for (int k = 0; k < outputLayer.size(); k++) {
            double delta = learningRate * outputErrors[k] * sigmoid_derivative(outputLayer[k]);
            biasOutput[k] += delta;
        }

        // Update weights input → hidden
        for (int i = 0; i < inputLayer.size(); i++) {
            for (int j = 0; j < hiddenLayer.size(); j++) {
                double delta = learningRate * hiddenErrors[j] * sigmoid_derivative(hiddenLayer[j]) * inputLayer[i];
                weightsInputHidden[i][j] += delta;
            }
        }

        // Update biases hidden
        for (int j = 0; j < hiddenLayer.size(); j++) {
            double delta = learningRate * hiddenErrors[j] * sigmoid_derivative(hiddenLayer[j]);
            biasHidden[j] += delta;
        }
    }
};

// -------------------- Helper Functions --------------------

// Training function with error tracking
void trainNetwork(NeuralNetwork& nn, const vector<vector<double>>& inputs,
    const vector<vector<double>>& outputs, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalError = 0.0;
        for (int i = 0; i < inputs.size(); i++) {
            nn.train(inputs[i], outputs[i]);
            nn.forward(inputs[i]);
            // Calculate mean squared error
            double err = 0.0;
            for (int k = 0; k < nn.outputLayer.size(); k++) {
                err += pow(outputs[i][k] - nn.outputLayer[k], 2);
            }
            totalError += err / nn.outputLayer.size();
        }
        if (epoch % 1000 == 0) {
            cout << "Epoch " << epoch << " - Error: " << totalError / inputs.size() << endl;
        }
    }
}

// Evaluate a single XOR using the trained network
int evaluateXOR(NeuralNetwork& nn, int a, int b) {
    vector<double> input = { (double)a, (double)b };
    nn.forward(input);
    double output = nn.outputLayer[0];
    return (output > 0.5) ? 1 : 0; // round to 0 or 1
}

// Parse and evaluate expression like "1xor0(0xor1)1xor1"
int evaluateExpression(NeuralNetwork& nn, const string& expr) {
    stack<int> values;
    stack<string> ops;

    for (size_t i = 0; i < expr.size(); i++) {
        if (expr[i] == '0' || expr[i] == '1') {
            values.push(expr[i] - '0');
        }
        else if (expr.substr(i, 3) == "xor") {
            ops.push("xor");
            i += 2; // skip "xor"
        }
        else if (expr[i] == '(') {
            ops.push("(");
        }
        else if (expr[i] == ')') {
            while (!ops.empty() && ops.top() != "(") {
                string op = ops.top(); ops.pop();
                int b = values.top(); values.pop();
                int a = values.top(); values.pop();
                values.push(evaluateXOR(nn, a, b));
            }
            ops.pop(); // remove "("
        }
    }

    while (!ops.empty()) {
        string op = ops.top(); ops.pop();
        int b = values.top(); values.pop();
        int a = values.top(); values.pop();
        values.push(evaluateXOR(nn, a, b));
    }

    return values.top();
}

// -------------------- main --------------------
int main() {
    vector<vector<double>> trainingInputs = {
        {0,0}, {0,1}, {1,0}, {1,1}
    };

    NeuralNetwork nn_xor(2, hiddenNeuronCount, 1);
    vector<vector<double>> trainingOutputsXOR = {
        {0}, {1}, {1}, {0}
    };
    trainNetwork(nn_xor, trainingInputs, trainingOutputsXOR, trainingEpochs);

    cout << "Neural Network trained on XOR gate.\n";
    cout << "Enter XOR expressions e.g., 1xor(0xor(1xor1)). Ctrl+C to quit.\n";

    string expr;
    while (cin >> expr) {
        int result = evaluateExpression(nn_xor, expr);
        cout << "Result = " << result << endl;
    }

    return 0;
}
