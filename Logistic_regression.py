import numpy as np
import pandas as pd


n_training_examples = 400
kernel = np.load('data/kernel.npy')
alpha = 0.1
batch_size = 50

beta = np.zeros(n_training_examples * 2)

y_vals = np.concatenate((np.zeros(n_training_examples), (np.ones(n_training_examples))))
def sigmoid(z):
    return 1/(1+ np.exp(-z))
def update_beta(indices, beta, learning_rate, y):
    beta[indices] += learning_rate * (y[indices] - sigmoid((kernel[indices] @ beta)))
for i in range(10000):
    indices = np.random.choice(np.arange(n_training_examples * 2), size=batch_size, replace=False)
    update_beta(indices, beta, alpha, y_vals)
np.save('data/beta.npy', beta)