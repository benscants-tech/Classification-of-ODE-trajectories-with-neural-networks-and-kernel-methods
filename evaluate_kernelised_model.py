import numpy as np
import pandas as pd



steps = 201
num_testing_examples = 100
num_training_examples = 400
beta = np.load('data/beta.npy')
a = 6
df = pd.read_csv('data/competitive_data.csv')
competitive_testing_data = [df["y1"][-num_testing_examples * steps:], df["y2"][-num_testing_examples * steps:]]
competitive_training_data = [df["y1"][:num_training_examples * steps], df["y2"][:num_training_examples * steps]]
df = pd.read_csv('data/pred_prey_data.csv')
pred_prey_testing_data = [df["y1"][-num_testing_examples * steps:], df["y2"][-num_testing_examples * steps:]]
pred_prey_training_data = [df["y1"][:num_training_examples * steps], df["y2"][:num_training_examples * steps]]

#handling testing data
competitive_testing_data = np.array(competitive_testing_data)
pred_prey_testing_data = np.array(pred_prey_testing_data)
competitive_testing_data = competitive_testing_data.reshape((2, num_testing_examples, steps))
competitive_testing_data = np.concatenate((competitive_testing_data[0, :, :], competitive_testing_data[1, :, :]), axis=1)
pred_prey_testing_data = pred_prey_testing_data.reshape((2, num_testing_examples, steps))
pred_prey_testing_data = np.concatenate((pred_prey_testing_data[0, :, :], pred_prey_testing_data[1, :, :]), axis=1)
testing_data = np.concatenate((competitive_testing_data, pred_prey_testing_data), axis=0)

#handling training data
competitive_training_data = np.array(competitive_training_data)
pred_prey_training_data = np.array(pred_prey_training_data)
competitive_training_data = competitive_training_data.reshape((2, num_training_examples, steps))
competitive_training_data = np.concatenate((competitive_training_data[0, :, :], competitive_training_data[1, :, :]), axis=1)
pred_prey_training_data = pred_prey_training_data.reshape((2, num_training_examples, steps))
pred_prey_training_data = np.concatenate((pred_prey_training_data[0, :, :], pred_prey_training_data[1, :, :]), axis=1)
training_data = np.concatenate((competitive_training_data, pred_prey_training_data), axis=0)

print(np.shape(training_data))
print(np.shape(testing_data))

y_vals = np.concatenate((np.zeros(num_testing_examples), (np.ones(num_testing_examples))))

testing_data = (testing_data - testing_data.mean(axis=0, keepdims=True))/testing_data.std(axis=0, keepdims=True)
training_data = (training_data - training_data.mean(axis=0, keepdims=True)) / training_data.std(axis=0, keepdims=True)
inner_product_matrix = (training_data @ testing_data.T) / (steps * 2)

kernel = np.zeros(shape=np.shape(inner_product_matrix))
for i in (np.arange(a) + 1):
    kernel += inner_product_matrix ** i
def sigmoid(z):
    return 1/(1 + np.exp(-z))
print(sigmoid(kernel.T @ beta))
print(np.sum(y_vals == (sigmoid((kernel.T @ beta)) > 0.5)))
