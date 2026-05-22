"""
This file is the base KDV physics informed neural network class, it call the other 
"""

import torch
import torch.nn as nn

from dataclasses import dataclass

import numpy as np
import random
from network import MLP

#Define domain dataclass, used to hold the points in domain
@dataclass
class TrainingDomain:
    x_coll: torch.Tensor
    t_coll: torch.Tensor
    x_ic: torch.Tensor
    t_ic: torch.Tensor
    u_ic: torch.Tensor
    x_bc: torch.Tensor
    t_bc: torch.Tensor
    u_bc: torch.Tensor

@dataclass
class TestingDomain:


@dataclass
class Solutions:
    exact: torch.Tensor
    linear: torch.Tensor
    predicted: torch.Tensor

@dataclass
class ErrorStats:
    mae: float
    max_error: float
    error: torch.Tensor


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
        if self.verbose:
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
                k1 = torch.sqrt(4/4) 
                k2 = torch.sqrt(1.2/4) 
                phi1 = 0.0
                phi2 = 0.0
                k_vector = torch.tensor([k1, k2], device=self.device)
                phi_vector = torch.tensor([phi1, phi2], device=self.device)
            case 3:
                k1 = torch.sqrt(1.0)
                k2 = torch.sqrt(0.8)
                k3 = torch.sqrt(0.5)
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

    def train(self):
        return True
        
    def test(self) -> ErrorStats:
        return True
    
    def compute_solutions(self) -> Solutions:
        return {'finish': torch.tensor([0])}

        
