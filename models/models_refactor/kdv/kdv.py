"""
This file is the base KDV physics informed neural network class, it call the other 
"""

import torch
import torch.nn as nn

class KDV(nn.Module):
    def __init__(self, init_params):
        # Set defaults
        defaults = dict(
            num_solitons=1,
            n_hidden_layers=3,
            n_neurons_per_layer=32,
            activation=nn.Tanh,
            seed=None,
            verbose=True,
            use_layernorm=False, 
        )
        # Merge user params with defaults
        params = {**defaults, **init_params}
        self.init_params = params.copy()

        super(KDV, self).__init__() # call constructor of parent class
        
        self.num_solitons = params['num_solitons'] 
        self.verbose = params['verbose'] 
        
        # deterministic seeding across CPU, GPU and numpy
        if params['seed'] is not None:
            self.seed = int(params['seed'])
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
        
        # create network and move to device
        self.net = PINN(params['n_hidden_layers'], params['n_neurons_per_layer'], params['activation'], params['use_layernorm'])
        self.net.to(self.device)
        
        # Set domain limits and parameters for different number of solitons
        if self.num_solitons == 1:
            self.x_lims = (-30, 30)
            self.t_lims = (-15, 15)

            k = 0.9  # wavenumber
            phi = 0  # phase parameter

            self.k_vector = np.array([k])
            self.phi_vector = np.array([phi])

        elif self.num_solitons == 2:
            self.x_lims = (-35, 50)
            self.t_lims = (-20, 35)

            k1 = np.sqrt(4/4) 
            k2 = np.sqrt(1.2/4) 
            phi1 = 0
            phi2 = 0

            self.k_vector = np.array([k1, k2])
            self.phi_vector = np.array([phi1, phi2])

        elif self.num_solitons == 3:
            k1 = np.sqrt(1)
            k2 = np.sqrt(0.8)
            k3 = np.sqrt(0.5)

            self.x_lims = (-35, 65)
            self.t_lims = (-25, 50)

            phi1 = 0
            phi2 = 0
            phi3 = 0

            self.k_vector = np.array([k1, k2, k3])
            self.phi_vector = np.array([phi1, phi2, phi3])
        
        # setup training and testing domain points
        self.setup_testing_domain()
        
        # setup figure size 
        self.figsize = (10, 6)

        # default batch size for streaming network evaluation during testing
        # keep the full test grid on CPU to save GPU RAM, send it in chunks only when needed
        self.test_batch_size = params.get('test_batch_size', 20_000)
        # ---- memory probe after initialization ----
        self._log_gpu_memory("after __init__")