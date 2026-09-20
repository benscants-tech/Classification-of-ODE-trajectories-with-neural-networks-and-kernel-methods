from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
t1 = time.time()

pred_prey_params = np.random.uniform(0.5, 1.0, size=(400, 4))
pred_prey_initial_conditions = np.random.uniform(1, 2, size=(400, 2))

competitive_params = np.random.uniform(0.5, 1.0, size = (400, 6))
competitive_initial_conditions = np.random.uniform(1, 2, size=(400, 2))


def pred_prey(input_params, initial_condition):
    def predator_prey_derivative(t, y):
        a, b, c, d = input_params
        y1, y2 = y
        return [a * y1 - b * y1 * y2, - c * y2 + d * y1 * y2]

    sol = solve_ivp(fun = predator_prey_derivative,
        t_span = (0, 10),
        y0 = initial_condition,
        method = "RK45",
        t_eval = np.linspace(0, 10, 201))
    return sol

def competitive(input_params, initial_condition):
    def competitive_derivative(t, y):
        a, b, c, d, e, f = input_params
        y1, y2 = y
        return [y1 * (a - b * y1 - c * y2), y2 * (d - e * y1 - f * y2)]
    sol = solve_ivp(fun = competitive_derivative, 
        t_span = (0, 10),
        y0 = initial_condition,
        method = 'RK45',
        t_eval = np.linspace(0, 10, 201))
    return sol


pred_prey_data = []
sim_no = 0
competitive_data = []
for i in range(400):
    initial = pred_prey_initial_conditions[i, :]
    params = pred_prey_params[i, :]
    sol = pred_prey(params, initial)
    a, b, c, d = params
    for t, y1, y2 in zip(sol.t, sol.y[0], sol.y[1]):
        pred_prey_data.append({
            "sim_no": sim_no, 'a': a, 'b': b, 'c': c, 'd': d,
            'y1_0': initial[0], 'y2_0': initial[1], 't': t, 'y1': y1, 'y2': y2
        })
    sim_no += 1

df = pd.DataFrame(pred_prey_data)
df.to_csv("data/pred_prey_data.csv", index=False)

sim_no = 0
for i in range(400):
    initial = competitive_initial_conditions[i, :]
    params = competitive_params[i, :]
    sol = competitive(params, initial)
    a, b, c, d, e, f = params
    for t, y1, y2 in zip(sol.t, sol.y[0], sol.y[1]):
        competitive_data.append({
            "sim_no": sim_no, 'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f,
            'y1_0': initial[0], 'y2_0': initial[1], 't': t, 'y1': y1, 'y2': y2
        })
    sim_no += 1
df = pd.DataFrame(competitive_data)
df.to_csv("data/competitive_data.csv", index=False)
