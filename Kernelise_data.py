import pandas as pd
import numpy as np

steps = 201
num_training_examples = 400
df = pd.read_csv('data/competitive_data.csv')
competitive_data = [df["y1"][:num_training_examples * steps], df["y2"][:num_training_examples * steps]]
df = pd.read_csv('data/pred_prey_data.csv')
pred_prey_data = [df["y1"][:num_training_examples * steps], df["y2"][:num_training_examples * steps]]
competitive_data = np.array(competitive_data)
pred_prey_data = np.array(pred_prey_data)
competitive_data = competitive_data.reshape((2, num_training_examples, steps))
competitive_data = np.concatenate((competitive_data[0, :, :], competitive_data[1, :, :]), axis=1)
pred_prey_data = pred_prey_data.reshape((2, num_training_examples, steps))
pred_prey_data = np.concatenate((pred_prey_data[0, :, :], pred_prey_data[1, :, :]), axis=1)

data = np.concatenate((competitive_data, pred_prey_data), axis=0)


a = 6
inner_product_matrix = (data @ data.T) / (steps * 2)
kernel_matrix = np.zeros(shape=np.shape(inner_product_matrix))
for i in (np.arange(a) + 1):
    kernel_matrix += inner_product_matrix ** i

np.save('data/kernel.npy', kernel_matrix)