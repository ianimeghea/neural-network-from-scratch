# neural-network-from-scratch
Neural network that classifies handwritten digits, from scratch, in python.

The neural network is coded inside the neural_nertworks.py file, and, after 100 thousand iterations of performing gradient descent, the model parameters are saved in model_params.pkl using pickle. 

The test_model.py chooses a random data sample and tests the accuracy of the network, which comes out to 94.5%.

No libraries expect from numpy are used in this project, as it is meant to be a low-level mathematical representation of a simple neural network.
