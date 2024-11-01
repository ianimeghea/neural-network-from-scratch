import numpy as np
import pickle
import pandas as pd
from neural_network import forward_prop, get_predictions, get_accuracy

with open("model_params.pkl", "rb") as f:
    model_params = pickle.load(f)

W1, b1, W2, b2 = model_params["W1"], model_params["b1"], model_params["W2"], model_params["b2"]

data = pd.read_csv('train.csv').values
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:].T / 255.0  

_, _, _, A2 = forward_prop(W1, b1, W2, b2, X_dev)
predictions = get_predictions(A2)
accuracy = get_accuracy(predictions, Y_dev)
print("Model accuracy on the development set:", accuracy)
