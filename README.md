# Soliton PINNs

Physics-Informed Neural Networks for solving the Korteweg–de Vries (KdV) and Kadomtsev–Petviashvili (KP) equations for soliton solutions. 

## `models/`

PyTorch implementations of the PINN models for KdV and KP equations.

- **`base.py`** — Base neural network architectures (2D and 3D input networks)
- **`kdv.py`** — KdV PINN class with training, testing, and visualization methods
- **`kp.py`** — KP PINN class with training, testing, and visualization methods

## `notebooks/`

Jupyter notebooks demonstrating how to use the models.

- **`kdv.ipynb`** — Examples for training and testing KdV models (1, 2, and 3 solitons)
- **`kp.ipynb`** — Examples for training and testing KP models (1 and 2 solitons) 
- **`initial_conditions.ipynb`** — Explanation of initial conditions for multi-soliton cases (for KdV)

## `ensembles/`

Experiment scripts for running multiple training runs with different configurations.

### `ensembles/replications/`

Run the same configuration with different random seeds to assess random/statistical variability. Each script saves results (losses and error grids) per seed, plus one full JSON for inspection.

- **`kdv-1-soliton/`** — KdV with 1 soliton
- **`kdv-2-soliton/`** — KdV with 2 solitons
- **`kp-1-soliton/`** — KP with 1 line soliton
- **`kp-x-junction/`** — KP x-junction (two line solitons)
- **`kp-y-junction/`** — KP y-junction (two solitons at different angle compared to x-junction)

### `ensembles/architecture/`

Test different network architectures (depth and width) and saves relevant results. 