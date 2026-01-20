"""
KP y-junction architecture sweep.

Tests network architectures (depth x width) with multiple random seeds for each architecture.

Saves metrics (final loss, mean error, max error) to CSV.

Can be interrupted and resumed - completed runs are skipped.
"""

import sys
from pathlib import Path

# add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import csv
import gc
import random
import time

import torch

from models.kp import KP

# architecture grid to sweep
N_HIDDEN_LAYERS_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
N_NEURONS_PER_LAYER_LIST = [2, 4, 8, 16, 32, 48, 62]
N_SEEDS = 10
SEEDS = [42 + i for i in range(N_SEEDS)]

# fixed configuration
K = (0.5, 1.0)
P = (3 / 4, 1 / 4)
T_LIMS = (-20, 20)

TRAIN_PARAMS = dict(
    adam_epochs=1000,
    lbfgs_epochs=100000,
    verbose_step=100,
    n_collocation=50000,
    n_initial=10000,
    n_boundary=250,
)

# setup paths
script_dir = Path(__file__).resolve().parent
results_root = script_dir.parent / "Results"
results_root.mkdir(parents=True, exist_ok=True)
results_csv = results_root / "results-kp-y-junction-architecture.csv"

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
    fieldnames=["n_hidden_layers", "n_neurons", "seed", "final_loss", "mean_error", "max_error", "time_seconds"],
)
if write_header:
    writer.writeheader()
    csv_file.flush()

for idx, (n_layers, n_neurons, seed) in enumerate(pending, 1):
    start_time = time.time()
    print(f"\n[{idx}/{len(pending)}] layers={n_layers}, neurons={n_neurons}, seed={seed}")

    final_loss = float("nan")
    mean_error = float("nan")
    max_error = float("nan")

    try:
        # clear GPU cache before each run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = KP(
            n_hidden_layers=n_layers,
            n_neurons_per_layer=n_neurons,
            k=K,
            P=P,
            t_lims=T_LIMS,
            seed=seed,
            verbose=False,
        )

        model.train(**TRAIN_PARAMS)
        model.test()

        if hasattr(model, "losses") and "total" in model.losses and model.losses["total"]:
            final_loss = float(model.losses["total"][-1])

        mean_error = float(getattr(model, "mae", float("nan")))

        if hasattr(model, "error"):
            max_error = float(torch.max(model.error).item())

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)

    elapsed = time.time() - start_time
    writer.writerow(
        dict(
            n_hidden_layers=n_layers,
            n_neurons=n_neurons,
            seed=seed,
            final_loss=final_loss,
            mean_error=mean_error,
            max_error=max_error,
            time_seconds=f"{elapsed:.2f}",
        )
    )
    csv_file.flush()

    try:
        del model
    except NameError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

csv_file.close()
print(f"\nDone - results saved to {results_csv}")
