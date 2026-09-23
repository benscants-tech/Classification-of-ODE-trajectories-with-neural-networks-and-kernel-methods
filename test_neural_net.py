import numpy as np
import pandas as pd

steps = 201
num_testing_examples = 100

num_layers = 4
df = pd.read_csv('data/competitive_data.csv')
competitive_data = [df["y1"][-num_testing_examples * steps:], df["y2"][-num_testing_examples * steps:]]
df = pd.read_csv('data/pred_prey_data.csv')
pred_prey_data = [df["y1"][-num_testing_examples * steps:], df["y2"][-num_testing_examples * steps:]]
competitive_data = np.array(competitive_data)
pred_prey_data = np.array(pred_prey_data)
competitive_data = competitive_data.reshape((2, num_testing_examples, steps))
competitive_data = np.concatenate((competitive_data[0, :, :], competitive_data[1, :, :]), axis=1)
pred_prey_data = pred_prey_data.reshape((2, num_testing_examples, steps))
pred_prey_data = np.concatenate((pred_prey_data[0, :, :], pred_prey_data[1, :, :]), axis=1)

data = np.concatenate((competitive_data, pred_prey_data), axis=0)
y_vals = np.concatenate((np.zeros(num_testing_examples), (np.ones(num_testing_examples))))

means = np.load('data/training_data_mean.npy')
std = np.load('data/training_data_mean.npy')

data = (data - means) / std
w = []
b = []
for i in range(num_layers):
    w_current = np.load('data/w_' + str(i) + '.npy')
    b_current = np.load('data/b_' + str(i) + '.npy')
    w.append(w_current)
    b.append(b_current)
count = 0
def sigmoid(z):
    return 1/(1 + np.exp(-z))
for i in range(num_testing_examples * 2):
    z = data[i, :]
    for j in range(num_layers):
        z = sigmoid(w[j].T @ z + b[j])
    print(z)
    print(z.item())
    if (z.item() > 0.5) == y_vals[i]:
        count += 1

print(count)
