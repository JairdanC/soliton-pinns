"""
The base KdV PINN solver class, calls to init, fit, solutions and plotting
"""

#Libraries
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import random
#Types
import typing
from matplotlib.figure import Figure
#Scripts
from . import trainer
from . import visualizer
#Methods
from ..network import MLP
from .methods import n_soliton, linear_combination
from .tester import setup_testing_domain, test
from .types import *

class KDV(nn.Module):
    def __init__(self, init_params
                 ) -> None:
        """
        The initialization of a PINN using a MLP architecture to solve the KdV equation
        """

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

        #Enforce deterministic seeding
        if self.char_params['seed'] is not None:
            self.seed = int(self.char_params['seed'])
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
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
    def fit(self, train_params: dict[str, typing.Any], 
              train_weights: dict[str, float]
              ) -> tuple[TrainingStats, TrainingDomain]:
        """
        The training (named fit as to not interfer with PyTorch superclass) of the neural network calling to the trainer module for helper functions, returns a
        tuple of the training statistics and the domain over which it was trained.
        """

        super(KDV, self).train(True)
        self.adam_epochs = train_params['adam_epochs'] #stashed for use in plotting
        training_stats, domain = trainer.train(self.neural_net, self.soliton_params, train_params, train_weights, self.device)
        super(KDV, self).train(False)
        return training_stats, domain
    
        
    def test(self, nx: int = 1000, 
             nt: int = 1000, 
             error_type='absolute-normalized'
             ) -> ErrorStats:
        """
        Test the current state of the neural network by computing exact solutions and 
        comparing them against model inference along a nx * nt grid, return the
        error statistics as a ErrorStats dataclass 
        """
        
        domain = setup_testing_domain(self.soliton_params['x_lims'], self.soliton_params['t_lims'], nx, nt)
        solutions = self.compute_solutions(domain)
        error_stats = test(solutions.predicted, solutions.exact, error_type, self.char_params['verbose'])
        return error_stats
    
    
    def compute_solutions(self, domain: TestingDomain, 
                          test_batch: int = 20000
                          ) -> Solutions:
        """
        Compute the exact, linear and prediction solutions and return them as
        a Solutions dataclass
        """

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
    
    
    #Wrapper to call without specific solutions
    def plot_profiles(self, t_values: list[int], 
                      which: tuple[str, ...] = ('predicted', ),
                      nx: int = 1000,
                      nt: int = 1000
                      ) -> Figure:
        """
        Auto-run version of the profiler in which the current model recomputes the solutions and then those
        profiles are passed to the plotter, overload will be added where a domain and solutions (already ran)
        can be plotted
        """
        
        domain = setup_testing_domain(self.soliton_params['x_lims'], self.soliton_params['t_lims'], nx, nt)
        solutions = self.compute_solutions(domain)
        plot = visualizer.plot_profiles(t_values, domain, solutions, which)
        return plot
    
    
    #Wrapper call
    def plot_losses(self, training_stats: TrainingStats,
                    components: list[str] = ['total', 'pde', 'boundary', 'initial', 'momentum', 'energy'], 
                    ) -> Figure:
        """
        Plot the losses given in the training stats returned from a training run
        """

        return visualizer.plot_losses(components, training_stats.losses, self.adam_epochs)
    

    #Wrapper call without specific solutions
    def plot_spacetime(self, nx: int = 1000,
                       nt: int = 1000,
                       scatter_which: tuple[str, ...] | None = None,
                       training_domain: TrainingDomain | None = None
                       ) -> Figure:
        """
        Auto-run version of the spacetime in which the current model recomputes the solutions and then those
        solutions are passed to the plotter, overload will be added where a domain and solution (already ran)
        can be plotted
        """
        
        domain = setup_testing_domain(self.soliton_params['x_lims'], self.soliton_params['t_lims'], nx, nt)
        solutions = self.compute_solutions(domain)
        if scatter_which is not None and training_domain is not None:
            scatter_coords = {}
            for key in scatter_which:
                match key:
                    case 'boundary':
                        coords = torch.stack((training_domain.x_bc, training_domain.t_bc), 0)
                    case 'pde':
                        coords = torch.stack((training_domain.x_coll, training_domain.t_coll), 0)
                    case 'initial':
                        coords = torch.stack((training_domain.x_ic, training_domain.t_ic), 0)
                    case _:
                        raise ValueError(f'Each key in scatter_which must be pde, initial or boundary.')
                
                scatter_coords[key] = coords
            
            return visualizer.plot_spacetime(domain, solutions.predicted, scatter_coords=scatter_coords)
        
        elif scatter_which is not None or training_domain is not None:
            raise ValueError('To scatter plot overlay the spacetime plot you must include both the ' \
            'keys of which conditions you want to scatter (pde, initial or boundary) AND the training domain')
        
        else:
            return visualizer.plot_spacetime(domain, solutions.predicted)
        
        
    def plot_heatmap(self, nx: int = 1000,
                     nt: int = 1000,
                     error_type: str = 'absolute-normalized'
                     ) -> Figure:
        """
        Auto-run version of the heatmap in which the current model is re-evaluated and then those
        errors are passed to the plotter, overload will be added where a domain and error (already ran)
        can be plotted
        """

        domain = setup_testing_domain(self.soliton_params['x_lims'], self.soliton_params['t_lims'], nx, nt)
        error_stats = self.test(nx, nt, error_type)
        fig = visualizer.plot_heatmap(error_stats.error, domain)
        return fig
