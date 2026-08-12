import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import gc
import pickle

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path.cwd().parent.parent))

from models import KDV

cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
testing_seeds = [42, 72, 83]

def conservation_test(path: str | Path,
                      momentum_int,
                      momentum_weight,
                      energy_int,
                      energy_weight,
                      hamilt_int,
                      hamilt_weight,
                      seeds = testing_seeds):
        
    """
    It is very important that you put in the correctly formatted arrays for each run, this is not a sweep python function
    with integrated error handling (though it may change), it is a one all function to make calls easier
    """

    path_obj = Path(path)
    
    # --- RESUME LOGIC ADDED HERE ---
    if path_obj.exists():
        print(f"Found existing file at {path_obj}. Loading previous results...")
        with open(path_obj, 'rb') as f:
            experiment_results = pickle.load(f)
        
        # Start index is the number of already completed runs
        start_index = len(experiment_results)
        print(f"Resuming experiment from index {start_index} out of {momentum_int.size}...\n")
        
        if start_index >= momentum_int.size:
            print("All runs are already complete!")
            return
    else:
        experiment_results = []
        start_index = 0
        print("Starting 3-Soliton Experiment Experiment from scratch...\n")
    # -------------------------------

    if not isinstance(momentum_int, np.ndarray):
        raise ValueError('passed non-np.ndarray')
    
    # Loop now begins at start_index instead of 0
    for i in range(start_index, momentum_int.size):

        results = {
            'time': [],
            'mae': [],
            'max_error': []
        }
    
        for current_seed in testing_seeds:

            INIT_PARAMS = dict(
                num_solitons             = 3,
                n_hidden_layers          = 5, 
                n_neurons_per_layer      = 106, 
                activation               = nn.Tanh,
                seed                     = current_seed, 
                verbose                  = False,
            )
            
            TRAIN_PARAMS = dict(
                adam_epochs              = 1500,
                verbose_step             = 100,
                n_collocation            = 175000, 
                n_initial                = 40000,  
                n_boundary               = 15000,
                n_momentum               = momentum_int[i], # only the t component, the x domain resolution will be the same as n_initial (nx * nt) = (n_initial * n_momentum)
                n_energy                 = energy_int[i],   # only the t component, the x domain resolution will be the same as n_initial (nx * nt) = (n_initial * n_energy)
                n_hamiltonian            = hamilt_int[i],
                adam_lr                  = 1e-3,   
                lbfgs_lr                 = 2.0,    
                lbfgs_history_size       = 295, 
                lbfgs_version            = 'test', #test is 'old' and anything else will default to a modified version of 'new' from legacy
                adaptive_sampling        = False,   
                logging                  = False, #new parameter, stops loss logging bottleneck for quick training (no loss history)
                verbose                  = False,
            )
            TRAIN_WEIGHTS = dict[str, float]( #seperated out from the train params
                w_ic                     = 10.0,    
                w_bc                     = 1.0,     
                w_pde                    = 400.0,
                w_momentum               = float(momentum_weight[i]),
                w_energy                 = float(energy_weight[i]),
                w_hamiltonian            = float(hamilt_weight[i]),
            )
        
            model = KDV(INIT_PARAMS)
            training_stats, _ = model.fit(TRAIN_PARAMS, TRAIN_WEIGHTS)
            error_stats = model.test(nx = 1000, nt=1000, error_type='absolute-normalized')

            print(f" -> Index Run {i} | Seed: {current_seed:<4} | Time: {training_stats.time:6.2f} s | MAE: {error_stats.mae:.6e}")

            results['time'].append(training_stats.time)
            results['mae'].append(error_stats.mae)
            results['max_error'].append(error_stats.max_error)
            
            # Explicitly free up GPU memory
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        experiment_results.append({
            'n_integrals' : dict(momentum = momentum_int[i], energy = energy_int[i], hamiltonian = hamilt_int[i]),
            'weights' : dict(momentum = momentum_weight[i], energy = energy_weight[i], hamiltonian = hamilt_weight[i]),
            'time_mean': float(np.mean(results['time'])),
            'time_std': float(np.std(results['time'])),
            'mae_mean': float(np.mean(results['mae'])),
            'mae_std': float(np.std(results['mae'])),
            'raw_data': results
        })
        
        # Save a backup to disk
        # Overwriting is safe here because experiment_results contains all previously loaded items + new items
        with open(path_obj, 'wb') as f:
            pickle.dump(experiment_results, f)
        
        print(f"=== Finished index = {i} | Avg MAE: {experiment_results[-1]['mae_mean']:.6e} ===\n")
            
energy_weight_flat = np.asarray([0.0, 1.0, 2.5, 5.0])
hamilt_weight_flat = np.asarray([0.0, 1.0, 2.5, 5.0])
energy_weight_mesh, hamilt_weight_mesh = np.meshgrid(energy_weight_flat, hamilt_weight_flat, indexing='xy')
energy_weight = energy_weight_mesh.flatten()
hamilt_weight = hamilt_weight_mesh.flatten()
momentum_int = np.zeros_like(energy_weight, dtype=int)
momentum_weight = np.zeros_like(energy_weight, dtype=float)
energy_int = np.ones_like(energy_weight, dtype=int) * 30
hamilt_int = np.ones_like(energy_weight, dtype=int) * 30


path = '3_soliton_energy_hamilt_weight_test.pkl'
seeds = testing_seeds

conservation_test(path, momentum_int, momentum_weight, energy_int, energy_weight, hamilt_int, hamilt_weight, seeds)