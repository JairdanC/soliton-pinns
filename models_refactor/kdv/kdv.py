"""
This file is the base KDV physics informed neural network class, it call the other 
"""

import torch
import torch.nn as nn

import numpy as np
import random
from network import MLP

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
                x_lims = (-30, 30)
                t_lims = (-15, 15)
                k = 0.9  # wavenumber
                phi = 0  # phase parameter
                k_vector = torch.tensor([k])
                phi_vector = torch.tensor([phi])                
            case 2:
                x_lims = (-35, 50)
                t_lims = (-20, 35)
                k1 = np.sqrt(4/4) 
                k2 = np.sqrt(1.2/4) 
                phi1 = 0
                phi2 = 0
                k_vector = np.array([k1, k2])
                phi_vector = np.array([phi1, phi2])
            case 3:
                k1 = np.sqrt(1)
                k2 = np.sqrt(0.8)
                k3 = np.sqrt(0.5)
                x_lims = (-35, 65)
                t_lims = (-25, 50)
                phi1 = 0
                phi2 = 0
                phi3 = 0
                k_vector = np.array([k1, k2, k3])
                phi_vector = np.array([phi1, phi2, phi3])
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
        
    def test(self):
        return True
    
    def compute_solutions(self) -> dict[str, torch.Tensor]:
        return {'finish': torch.tensor([0])}

        
