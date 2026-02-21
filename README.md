# Neural Network Logic Gates Project

## Overview
This project implements a simple **feedforward neural network** in **C++** to demonstrate how fundamental logic gates (**XOR** and **NAND**) can be learned through **supervised training**. The code showcases the principles of **forward propagation**, **backpropagation**, and **weight updates** using the **sigmoid activation function**.  

The project is designed as an educational example to illustrate how **neural networks** can approximate **non-linear functions** and be extended to more complex tasks such as **character recognition** or **word generation**.

---

## Features
- **Neural Network Class**  
  - Configurable **input layer**, **hidden layer**, and **output layer**.  
  - **Sigmoid activation function** and its **derivative**.  
  - Random **weight initialization**.  
  - **Forward propagation** and **backpropagation training**.  

- **Training and Testing Functions**  
  - `trainNetwork()` encapsulates the **training loop**.  
  - `testNetwork()` encapsulates the **testing loop**.  
  - Clear separation of concerns for **readability** and **maintainability**.  

- **Logic Gate Implementations**  
  - **XOR gate**: non-linear problem requiring hidden layer neurons.  
  - **NAND gate**: simpler logic gate, also trained and tested.  

- **Configurable Parameters**  
  - **learningRate**: controls the speed of **weight updates**.  
  - **hiddenNeuronCount**: sets the number of neurons in the **hidden layer**.  
  - **trainingEpochs**: defines the number of **training iterations**.  

---

## Code Structure
- **Activation Functions**: `sigmoid()` and `sigmoid_derivative()`  
- **NeuralNetwork Class**: encapsulates **layers**, **weights**, **forward pass**, and **training logic**.  
- **Helper Functions**:  
  - `trainNetwork()` – trains the network with given **inputs** and **outputs**.  
  - `testNetwork()` – evaluates the network and prints **results**.  
- **Main Function**: defines **training data** for **XOR** and **NAND**, trains networks, and prints **test results**.

---

## Example Output
After training, the network produces outputs close to the expected **truth tables**:

**XOR Gate**
`0 XOR 0 = ~0.03
0 XOR 1 = ~0.97
1 XOR 0 = ~0.97
1 XOR 1 = ~0.02`

**NAND Gate**
`0 NAND 0 = ~0.97
0 NAND 1 = ~0.96
1 NAND 0 = ~0.96
1 NAND 1 = ~0.03`

## Purpose
This project serves as:
- A learning tool for understanding neural networks at a low level.
- A demonstration of how networks can learn logic gates.
- A foundation for extending to more complex problems (e.g., character recognition, word generation).

## Future Extensions
- Add support for other logic gates (AND, OR, NOR).
- Experiment with different activation functions (ReLU, tanh).
- Extend input size to handle character arrays or pixel representations.
- Explore recurrent architectures for sequential data (e.g., words).

## License
This project is released under the MIT License. You are free to use, modify, and distribute it with attribution.
