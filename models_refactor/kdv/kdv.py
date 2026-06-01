"""
This file is the base KDV physics informed neural network class, it call the other 
"""

import torch
import torch.nn as nn
import numpy as np
import random
import gc
import typing

from dataclasses import dataclass

from network import MLP
import kdv_trainer as trainer

from kdv_analysis import *
from kdv_tester import *
from kdv_types import *

class KDV(nn.Module):
    def __init__(self, init_params) -> None:

        #defaults
        defaults = dict(
            num_solitons=1,
            n_hidden_layers=3,
            n_neurons_per_layer=32,
            activation=nn.Tanh,
            seed=None,
            verbose=True,
            use_layernorm=False, 
        )

        # Merge user params with defaults for the characteristic parameters of the neural network
        self.char_params = {**defaults, **init_params}

        super(KDV, self).__init__() #calls the constructor of the parent class (PyTorch function)

        if self.char_params['seed'] is not None:
            self.seed = int(self.char_params['seed'])
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
            # enforce deterministic behaviour in cuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # set device to GPU (if available) otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.char_params['verbose']:
            print(f"Using device: {self.device}")

        self.neural_net = MLP(self.char_params['n_hidden_layers'], self.char_params['n_neurons_per_layer'], self.char_params['activation'], self.char_params['use_layernorm'], input=2, output=1)
        self.neural_net.to(self.device)

        match self.char_params['num_solitons']:
            case 1:
                x_lims = torch.tensor([-30, 30], device=self.device)
                t_lims = torch.tensor([-15, 15], device=self.device)
                k = 0.9  # wavenumber
                phi = 0.0  # phase parameter
                k_vector = torch.tensor([k], device=self.device)
                phi_vector = torch.tensor([phi], device=self.device)                
            case 2:
                x_lims = torch.tensor([-35, 50], device=self.device)
                t_lims = torch.tensor([-20, 35], device=self.device)
                k1 = torch.tensor(4/4, device=self.device).sqrt() 
                k2 = torch.tensor(1.2/4, device=self.device).sqrt()
                phi1 = 0.0
                phi2 = 0.0
                k_vector = torch.tensor([k1, k2], device=self.device)
                phi_vector = torch.tensor([phi1, phi2], device=self.device)
            case 3:
                k1 = torch.tensor(1.0, device=self.device).sqrt()
                k2 = torch.tensor(0.8, device=self.device).sqrt()
                k3 = torch.tensor(0.5, device=self.device).sqrt()
                x_lims = torch.tensor([-35, 65], device=self.device)
                t_lims = torch.tensor([-25, 50], device=self.device)
                phi1 = 0.0
                phi2 = 0.0
                phi3 = 0.0
                k_vector = torch.tensor([k1, k2, k3], device=self.device)
                phi_vector = torch.tensor([phi1, phi2, phi3], device=self.device)
            case _:
                raise ValueError("n_soliton only implemented for N = 1, 2, 3 solitons") 
            
        self.soliton_params = {
                'x_lims': x_lims,
                't_lims': t_lims,
                'k_vec': k_vector,
                'phi_vec': phi_vector
            }
            
        #here is where testing domain is called in the original code seems too early
        return

    #wrapper function to call to module
    def train(self, train_params: dict[str, typing.Any], train_weights: dict[str: float]):
        training_stats = trainer.train(self.neural_net, self.soliton_params, train_params, train_weights, self.device)
        return training_stats
        
    def test(self, nx: int = 1000, nt: int = 1000, error_type='absolute-normalized') -> ErrorStats:
        domain = setup_testing_domain(self.soliton_params['x_lims'], self.soliton_params['t_lims'], nx, nt)
        solutions = self.compute_solutions(domain)
        test(solutions.predicted, solutions.exact, error_type, self.char_params['verbose'])
        return True
    
    def compute_solutions(self, domain: TestingDomain, test_batch: int = 20000
                          ) -> Solutions:

        with torch.inference_mode():
            B = test_batch
            n_points = domain.x_test.numel()

            X_gpu = domain.x_test.to(self.device)
            T_gpu = domain.t_test.to(self.device)
            X_flat = X_gpu.reshape(-1, 1)
            T_flat = T_gpu.reshape(-1, 1)

            pred_chunks = []
            for i in range(0, n_points, B):
                pred_chunks.append(self.neural_net(X_flat[i:i+B], T_flat[i:i+B]))
            U_pred = torch.cat(pred_chunks).reshape(domain.x_test.shape).cpu()
        
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        U_exact = n_soliton(domain.x_test, domain.t_test, 
                            self.soliton_params['k_vec'], self.soliton_params['phi_vec'])
        U_linear = linear_combination(domain.x_test, domain.t_test,
                                      self.soliton_params['k_vec'], self.soliton_params['phi_vec'])

        solution = Solutions(U_exact, U_linear, U_pred)
        return solution