"""
This file contains the functions used in training the given neural network
"""
import torch
import torch.nn as nn
import time
import typing

from kdv_loss import *
from utils import *
from network import *

#Not fixed yet
def setup_training_domain(n_collocation, n_initial, n_boundary):
        """
        Setup the domain points for training. Also generates initial conditions for the corresponding number of solitons.
        """
        # unpack domain limits
        x0 = self.x_lims[0]
        x1 = self.x_lims[1]
        
        t0 = self.t_lims[0]
        t1 = self.t_lims[1]

        # 1. Collocation points (random in the domain)
        x_collocation = torch.rand(n_collocation, 1) * (x1 - x0) + x0
        t_collocation = torch.rand(n_collocation, 1) * (t1 - t0) + t0

        # 2. Initial condition points (t=t0)
        x_initial = torch.linspace(x0, x1, n_initial).reshape(-1, 1)
        t_initial = torch.ones_like(x_initial) * t0
        u_initial = self.linear_combination(x_initial, t_initial)

        # 3. Boundary condition points - uniform grid
        t_boundary_left = torch.linspace(t0, t1, n_boundary//2).reshape(-1, 1)
        x_boundary_left = torch.ones_like(t_boundary_left) * x0
        t_boundary_right = torch.linspace(t0, t1, n_boundary//2).reshape(-1, 1)
        x_boundary_right = torch.ones_like(t_boundary_right) * x1
        x_boundary = torch.cat([x_boundary_left, x_boundary_right], dim=0)
        t_boundary = torch.cat([t_boundary_left, t_boundary_right], dim=0)
        u_boundary = torch.zeros_like(x_boundary)

        # Move all tensors to the device
        self.x_collocation = x_collocation.to(self.device)
        self.t_collocation = t_collocation.to(self.device)
        self.x_initial = x_initial.to(self.device)
        self.t_initial = t_initial.to(self.device)
        self.u_initial = u_initial.to(self.device)
        self.x_boundary = x_boundary.to(self.device)
        self.t_boundary = t_boundary.to(self.device)
        self.u_boundary = u_boundary.to(self.device)

        # ---- memory probe after setup_training_domain ----
        self._log_gpu_memory("after setup_training_domain")

        if self.verbose:
            print(f"""Training domain setup complete: 
                    - {n_collocation} collocation points
                    - {n_initial} initial points
                    - {n_boundary} boundary points""")
            print(f"Using {self.num_solitons}-soliton initial condition.")
            print(f"Solving over the domain: t: {self.t_lims}, x: {self.x_lims}")

def train(neural_net: MLP, train_params) -> dict[str, typing.Any]:
    #set defaults
    defaults = dict(
        adam_epochs=1000,
        lbfgs_epochs=50000,
        verbose_step=100,
        n_collocation=30000,
        n_initial=30000,
        n_boundary=30000,
        w_ic=1.0,
        w_bc=1.0,
        w_pde=1.0,
        adam_lr=0.001,
        lbfgs_lr=1.0,
        lbfgs_history_size=100,
        adaptive_sampling=False,
        lbfgs_version='old', # 'old' or 'new'
        verbose= True
    )
    params = {**defaults, **train_params} #unpack parameters

    #start wall-clock timer
    start_time = time.time() #export this function to utils later

    setup_training_domain(
        n_collocation=params['n_collocation'],
        n_initial=params['n_initial'],
        n_boundary=params['n_boundary']
    )

    print_weighted_loss_components(label='start') #not fixed yet

    losses = init_loss_list()
    loss_weights = init_loss_weights()

    #Adam Optimizer
    if params['verbose']: print('Starting Adam optimization...')
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=params['adam_lr'])
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
    log_gpu_memory("train start")

    for epoch in range(params['adam_epochs']):
        optimizer.zero_grad(set_to_none=True)
        loss_comps = loss_components()
        total_loss = total_loss(loss_weights, loss_comps)
        total_loss.backward()
        optimizer.step()

        update_loss_list()

        if params['verbose'] and (epoch % params['verbose_step'] == 0 or epoch == params['adam_epochs'] - 1):
            print(f"Adam - Epoch {epoch}/{params['adam_epochs']}, Total Loss: {total_loss.item():.6e}")

    log_gpu_memory("after Adam")

    #adaptive sampling would go here
    

    #L-BFGS optimization
    if params['verbose']: print("\nStarting L-BFGS optimization...")
    #potential delete later
    if params['lbfgs_version'] == 'old':
        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss_comps = loss_components()
            total_loss = total_loss(loss_weights, loss_comps)
            total_loss.backward()

            update_loss_list()

            if params['verbose'] and len(losses['total']) % params['verbose_step'] == 0:
                print(f"L-BFGS - Iteration {len(losses['total']) - params['adam_epochs']}, Total Loss: {total_loss.item():.6e}")
        
            return total_loss
        
        optimizer = torch.optim.LBFGS(neural_net.net.parameters(),
                                    lr= params['lbfgs_lr'], 
                                    max_iter=params['lbfgs_epochs'],
                                    max_eval=params['lbfgs_epochs']*2,
                                    tolerance_grad=1e-9,
                                    tolerance_change=1e-16,
                                    history_size=params['lbfgs_history_size'],
                                    line_search_fn="strong_wolfe"
                                    )
        
        optimizer.step(closure)

        if params['verbose']:
            print(f"L-BFGS complete, Final Loss: {losses['total'][-1]:.6e}")

    elif params['lbfgs_version'] == 'new':
        optimizer = torch.optim.LBFGS(
                neural_net.parameters(),
                lr=params['lbfgs_lr'],
                max_iter=1,                  #one accepted iteration per step()
                max_eval=100,
                tolerance_grad=1e-9,
                tolerance_change=1e-16,
                history_size=params['lbfgs_history_size'],
                line_search_fn="strong_wolfe",
            )
        
        last_vals = {}

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss_comps = loss_components()
            total_loss = total_loss(loss_weights, loss_comps)
            total_loss.backward()

            update_loss_dict()

            return total_loss
        
        for i in range(params['lbfgs_epochs']):
            loss = optimizer.step(closure)
            update_loss_list()

            if params['verbose'] and (i % params['verbose_step'] == 0 or i == params['lbfgs_epochs'] - 1):
                print(f"L-BFGS - Iteration {i+1}/{params['lbfgs_epochs']}, Total Loss: {float(loss):.6e}")

    else: 
        raise ValueError("lbfgs_version only implemented for" \
        "\'old\' and \'new\', must be defined as such")

    log_gpu_memory("after L-BFGS")

    training_stats = {
        'losses': losses,
        'training time': time.time() - start_time
    }

    if params['verbose']: print(f"Training completed in {training_stats['training time']:.2f} s")
    
    print_weighted_loss_components(label='end')

    return training_stats






            
         


        
