import numpy as np
import pandas as pd

competitive_data = pd.read_csv('data/competitive_data.csv')
pred_prey_data = pd.read_csv('data/pred_prey_data.csv')


layers = [10, 6, 3]
# functional parameters of backprop are W, b, y, z, and a
weights = []
biases = []

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def forward_prop(weight, bias, x):
    z = []
    a = [x]
    a_current = x
    for idx in range(len(weight)):
        z_current = weight[idx].T @ a_current + bias[idx]
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
        grad_weight.append(np.outer(a[k], current_z_derivative))
        grad_bias.append(current_z_derivative)
        current_a_derivative = weight[k] @ current_z_derivative
    return grad_weight[::-1], grad_bias[::-1]

for i in range(len(layers) - 1):
    weights.append(np.random.randn(layers[i], layers[i + 1]) / np.sqrt(layers[i]))
    biases.append(np.zeros(layers[i + 1],))

x = np.random.randn(10)
y = np.random.choice(np.array([0, 1]), size=3).astype(float)
z, a = forward_prop(weights, biases, x)
gradients = back_prop(z, a, weights, y)

def logistic(y, z):
    return np.sum(y * np.log(1 + np.exp(-z[-1])) + (1 - y) * np.log(1 + np.exp(z[-1])))


grad_w, grad_b = back_prop(z, a, weights, y)
k, i, j = 0, 2, 3
eps = 1e-5
weights[k][i, j] += eps
z, a = forward_prop(weights, biases, x)
lp = logistic(y, z)
weights[k][i, j] -= 2 * eps
z, a = forward_prop(weights, biases, x)
lm = logistic(y, z)

print((lp - lm)/(2 * eps), grad_w[k][i, j])

