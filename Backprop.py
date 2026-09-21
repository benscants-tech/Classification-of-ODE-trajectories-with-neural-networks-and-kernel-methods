import numpy as np
import pandas as pd

competitive_data = pd.read_csv('data/competitive_data.csv')
pred_prey_data = pd.read_csv('data/pred_prey_data.csv')


n_examples = 800

layers = [402, 240, 80, 10, 2]

batch_size = 32
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def forward_prop(weight, bias, X):
    z = []
    a = [X]
    a_current = X
    for idx in range(len(weight)):
        z_current = weight[idx].T @ a_current + bias[idx][:, None]
        z.append(z_current)
        if idx < len(weight) - 1:
            a_current = sigmoid(z_current)
            a.append(a_current)
    return z, a

def back_prop(z, a, weight, y):
    r = len(weight)
    current_z_derivative = sigmoid(z[r-1]) - y
    grad_weight =[]
    grad_bias = []
    for k in range(r)[::-1]:
        if k < r - 1:
            current_z_derivative = sigmoid_prime(z[k]) * current_a_derivative  
        grad_weight.append((a[k] @ current_z_derivative.T)/np.size(y, axis=1))
        grad_bias.append(current_z_derivative.mean(axis=1))
        current_a_derivative = weight[k] @ current_z_derivative
    return grad_weight[::-1], grad_bias[::-1]

def logistic(y, z):
    return np.sum(y * np.log(1 + np.exp(-z[-1])) + (1 - y) * np.log(1 + np.exp(z[-1])))

def training(x, y, b_size, max_steps, alpha):
    w = []
    b = []
    for i in range(len(layers) - 1):
        w.append(np.random.randn(layers[i], layers[i + 1]) / np.sqrt(layers[i]))
        b.append(np.zeros(layers[i + 1]))
    count = 0
    loss_current = 0
    while count < max_steps:
        choices = np.random.choice(b_size, size=n_examples)
        z, a = forward_prop(w, b, x[choices])
        grad_w, grad_b = back_prop(z, a, w, y[choices])
        for i in range(len(w)):
            w[i] -= alpha * grad_w[i]
            b[i] -= alpha * grad_b[i]
        count += 1
        loss_prev, loss_current = loss_current, logistic(y, z)
        if np.abs(loss_current - loss_prev) < 1e-4:
            return loss_current, w, b
    return loss_current, w, b