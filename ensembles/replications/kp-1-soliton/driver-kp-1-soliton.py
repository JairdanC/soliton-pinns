"""
KP 1-soliton replication study.

Trains and tests the model multiple times with different random seeds.

Saves results (losses and error grids) for each seed, plus one full JSON result for inspection.

Can be interrupted and resumed - progress is tracked in progress.csv.
"""

import sys
from pathlib import Path

# add repo root to Python path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import csv
import gc
import time

import torch
import random

from models.kp import KP

# configuration 
N_HIDDEN_LAYERS = 5
N_NEURONS = 4
K_VECTOR = (0.5,)
P_VECTOR = (2/3,)
T_LIMS = (-10, 10)
N_COLLOCATION = 50000
N_INITIAL = 10000
N_BOUNDARY = 500

NUM_RUNS = 100

# setup paths
results_root = Path(__file__).parent
results_root.mkdir(parents=True, exist_ok=True)
progress_csv = results_root / "progress.csv"

# load previous progress or initialize
if progress_csv.exists():
    # read existing seeds and times
    with open(progress_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        SEEDS = [int(row["seed"]) for row in rows]
        completed_seeds = {row["seed"] for row in rows if row["time_s"]}
else:
    # first run - generate seeds and write CSV with empty times
    SEEDS = random.sample(range(1_000_000), NUM_RUNS)
    with open(progress_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "time_s"])
        writer.writeheader()
        for seed in SEEDS:
            writer.writerow({"seed": seed, "time_s": ""})
    completed_seeds = set()

# only run for seeds that are not completed
pending = [(i, seed) for i, seed in enumerate(SEEDS) if str(seed) not in completed_seeds]

# main loop (train, test, save)
for run, seed in pending:
    start_time = time.time()
    print(f"\nRun {run + 1}/{NUM_RUNS} (seed {seed})")
    
    # create model
    model = KP(
        n_hidden_layers=N_HIDDEN_LAYERS,
        n_neurons_per_layer=N_NEURONS,
        k=K_VECTOR,
        P=P_VECTOR,
        t_lims=T_LIMS,
        seed=seed,
        verbose=False,
    )
    
    # train and test
    model.train(n_collocation=N_COLLOCATION, n_initial=N_INITIAL, n_boundary=N_BOUNDARY)
    model.test()
    
    # save results for this seed
    model.save_experiment_run(results_root)
    
    # save one full result for inspection (first seed only)
    if run == 0:
        model.save_results(results_root / f"model_{seed}.json")
    
    # track timing and update progress
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} s")
    
    # update CSV with timing for this seed
    rows = []
    with open(progress_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["seed"] == str(seed):
                row["time_s"] = f"{elapsed:.2f}"
            rows.append(row)
    
    with open(progress_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "time_s"])
        writer.writeheader()
        writer.writerows(rows)
    
    # memory cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 