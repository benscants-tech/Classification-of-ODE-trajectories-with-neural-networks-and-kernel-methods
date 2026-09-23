# Classification of ODE Trajectories with Neural Networks and Kernel Methods

Can a model tell which system of differential equations produced a trajectory, just by looking at the trajectory?

This project generates solutions to two classic two-species population models and trains two classifiers, written from scratch in NumPy, to distinguish them:

- a **fully connected neural network** trained with hand-written backpropagation
- **kernelised logistic regression** with a polynomial kernel

## The two systems

**Predator–prey (Lotka–Volterra)**, labelled `1`:

$$
\dot{y}_1 = a y_1 - b y_1 y_2, \qquad \dot{y}_2 = -c y_2 + d y_1 y_2
$$

**Competitive Lotka–Volterra**, labelled `0`:

$$
\dot{y}_1 = y_1 (a - b y_1 - c y_2), \qquad \dot{y}_2 = y_2 (d - e y_1 - f y_2)
$$

For each system, 500 trajectories are generated. Parameters are drawn uniformly from $[0.01, 1]$ and initial conditions uniformly from $[1, 2]$. Each system is integrated over $t \in [0, 10]$ with RK45 (`scipy.integrate.solve_ivp`) and sampled at 201 time points.

## Method

**Features.** Each trajectory becomes a single 402-dimensional vector: the 201 values of $y_1$ followed by the 201 values of $y_2$. Each vector is then standardised by its own mean and standard deviation. This means the classifiers see the *shape* of a trajectory rather than its absolute scale.

**Train/test split.** The first 400 trajectories of each system (800 total) are used for training. The last 100 of each (200 total) are held out for testing.

**Neural network** (`Backprop.py`). The architecture is 402 → 240 → 80 → 10 → 1, with sigmoid activations and a logistic loss. It is trained with mini-batch SGD (batch size 64, learning rate 0.1, 10,000 steps). Weights are initialised with scaling $1/\sqrt{n_\text{in}}$.

**Kernelised logistic regression** (`Kernelise_data.py`, `Logistic_regression.py`). The model uses the kernel

$$
K(x, x') = \sum_{k=1}^{6} \left( \frac{\langle x, x' \rangle}{402} \right)^k .
$$

This is a sum of polynomial kernels and is therefore positive semi-definite. The model is trained with the kernelised SGD update

$$
\beta_i \leftarrow \beta_i + \alpha \left( y^{(i)} - \sigma\!\left( \textstyle\sum_j K(x^{(i)}, x^{(j)})\, \beta_j \right) \right).
$$

## Repository structure

| File | Purpose |
| --- | --- |
| `ODE_solutions.py` | Generates trajectories for both systems and saves them to `data/*.csv` |
| `Kernelise_data.py` | Builds the training kernel matrix, saved as `data/kernel.npy` |
| `Logistic_regression.py` | Trains kernelised logistic regression, saving coefficients to `data/beta.npy` |
| `Backprop.py` | Trains the neural network, saving weights to `data/w_*.npy` and `data/b_*.npy` |
| `evaluate_kernelised_model.py` | Tests the kernel model on the held-out trajectories |
| `evaluate_neural_net.py` | Tests the neural network on the held-out trajectories |

## Getting started

Requires Python 3.10+.

```bash
git clone https://github.com/benscants-tech/Classification-of-ODE-trajectories-with-neural-networks-and-kernel-methods.git
cd Classification-of-ODE-trajectories-with-neural-networks-and-kernel-methods
pip install -r requirements.txt
mkdir data
```

Run the scripts in this order:

```bash
python ODE_solutions.py              # generate data
python Kernelise_data.py             # kernel model: build kernel
python Logistic_regression.py        # kernel model: train
python evaluate_kernelised_model.py  # kernel model: evaluate
python Backprop.py                   # neural net: train
python evaluate_neural_net.py        # neural net: evaluate
```

Each evaluation script prints the number of correctly classified test trajectories, out of 200.

## Results

All scripts that use randomness (`ODE_solutions.py`, `Logistic_regression.py`, `Backprop.py`) fix the NumPy seed with `np.random.seed(0)`, so the results below are exactly reproducible.

| Model | Test accuracy (seed 0) |
| --- | --- |
| Kernelised logistic regression | 197 / 200 (98.5%) |
| Neural network | 196 / 200 (98.0%) |

Across unseeded runs, both models typically score between 97% and 100%. A single seeded result is one sample from that range, not a precise measure of either model's accuracy.

## Limitations and possible extensions

- **The task is probably quite easy.** Predator–prey solutions oscillate, while competitive solutions typically settle towards an equilibrium. The classifiers may largely be detecting oscillation. Harder variants could add noise, use shorter or irregular time windows, or include more systems.
- **Scale information is discarded.** Per-trajectory standardisation throws away amplitude information.
- **Results come from a single seed.** Averaging over several seeds would give a more reliable accuracy estimate.
- **Hyperparameters are untuned.** The learning rates, architecture and kernel degree were not chosen with a validation set.
