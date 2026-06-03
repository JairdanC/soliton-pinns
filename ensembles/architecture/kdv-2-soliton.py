"""
KDV 2-soliton architecture sweep.

Tests network architectures (depth x width) with multiple random seeds for each architecture.

Saves metrics (final loss, mean error, max error) to CSV.

Can be interrupted and resumed - completed runs are skipped.
"""

import sys
from pathlib import Path

# add repo root to Python path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import csv
import gc
import random
import time

import torch
import torch.nn as nn

from models.kdv import KDV_LEGACY  

# architecture grid to sweep
N_HIDDEN_LAYERS_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
N_NEURONS_PER_LAYER_LIST = [2, 4, 8, 16, 32, 48, 62]
N_SEEDS = 10
SEEDS = [42 + i for i in range(N_SEEDS)]

# fixed training configuration
TRAIN_PARAMS = dict(
    adam_epochs=1000,
    lbfgs_epochs=100000,
    verbose_step=100,
    n_collocation=100000,
    n_initial=10000,
    n_boundary=10000,
    w_ic=10.0,
    w_bc=1.0,
    w_pde=100.0,
    adam_lr=1e-3,
    lbfgs_lr=2.0,
    lbfgs_history_size=295,
    adaptive_sampling=False,
)

# setup paths
script_dir = Path(__file__).resolve().parent
results_root = script_dir.parent / "Results"
results_root.mkdir(parents=True, exist_ok=True)
results_csv = results_root / "results-kdv-2-soliton-architecture.csv"

# load completed runs
completed = set()
if results_csv.exists():
    with results_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fl = row.get("final_loss", "nan").lower()
                if fl == "nan" or fl == "":
                    continue
                completed.add((int(row["n_hidden_layers"]), int(row["n_neurons"]), int(row["seed"])))
            except (KeyError, ValueError):
                continue

# build list of pending runs
pending = [
    (layers, neurons, seed)
    for layers in N_HIDDEN_LAYERS_LIST
    for neurons in N_NEURONS_PER_LAYER_LIST
    for seed in SEEDS
    if (layers, neurons, seed) not in completed
]

print(f"Total combinations: {len(N_HIDDEN_LAYERS_LIST) * len(N_NEURONS_PER_LAYER_LIST) * N_SEEDS}")
print(f"Pending: {len(pending)}")

# randomize run order
random.shuffle(pending)

# open CSV for appending
write_header = not results_csv.exists()
csv_file = results_csv.open("a", newline="")
writer = csv.DictWriter(
    csv_file,
    fieldnames=["n_hidden_layers", "n_neurons", "seed", "final_loss", "mean_error", "max_error", "time_seconds"]
)
if write_header:
    writer.writeheader()
    csv_file.flush()

# main loop
for idx, (n_layers, n_neurons, seed) in enumerate(pending, 1):
    start_time = time.time()
    print(f"\n[{idx}/{len(pending)}] layers={n_layers}, neurons={n_neurons}, seed={seed}")
    
    final_loss = float("nan")
    mean_error = float("nan")
    max_error = float("nan")
    
    try:
        # create model
        init_params = dict(
            num_solitons=2,
            n_hidden_layers=n_layers,
            n_neurons_per_layer=n_neurons,
            activation=nn.Tanh,
            seed=seed,
            verbose=False,
        )
        model = KDV_LEGACY(init_params)
        
        # train and test
        model.train(TRAIN_PARAMS)
        model.test()
        
        # extract metrics
        final_loss = float(getattr(model, "final_loss", float("nan")))
        mean_error = float(getattr(model, "mae", float("nan")))
        max_error = float(getattr(model, "max_error", float("nan")))
        
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
    
    # save results
    elapsed = time.time() - start_time
    writer.writerow(dict(
        n_hidden_layers=n_layers,
        n_neurons=n_neurons,
        seed=seed,
        final_loss=final_loss,
        mean_error=mean_error,
        max_error=max_error,
        time_seconds=f"{elapsed:.2f}",
    ))
    csv_file.flush()
    
    # memory cleanup
    try:
        del model
    except NameError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

csv_file.close()
print(f"\nDone - results saved to {results_csv}")