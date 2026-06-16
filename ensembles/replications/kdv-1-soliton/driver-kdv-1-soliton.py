"""
KDV 1-soliton replication study.

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
import torch.nn as nn
import random

from models.kdv import KDV_LEGACY

# configuration
INIT_PARAMS = dict(
    num_solitons=1,
    n_hidden_layers=4, 
    n_neurons_per_layer=32, 
    activation=nn.Tanh,
    seed=None,  # overridden per run
    verbose=False,
)

TRAIN_PARAMS = dict(
    adam_epochs=1000,
    lbfgs_epochs=3000,
    verbose_step=100,
    n_collocation=50000, 
    n_initial=30000,  
    n_boundary=10000,  
    w_ic=5.0,    
    w_bc=1.0,    
    w_pde=15.0,   
    adam_lr=1e-3,   
    lbfgs_lr=1.0,    
    lbfgs_history_size=100,    
    adaptive_sampling=False,   
    lbfgs_version='old'
)

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
    init_params = INIT_PARAMS.copy()
    init_params["seed"] = seed
    model = KDV_LEGACY(init_params)
    
    # train and test
    model.train(TRAIN_PARAMS)
    model.test()
    
    # save results for this seed
    model.save_experiment_run(results_root)
    
    # save one full result for inspection (first seed only)
    if run == 0:
        model.save_model_result(results_root / f"model_{seed}.json")
    
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