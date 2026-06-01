"""
This file contains the functions used in training the given neural network including the setup_training_domain_
"""
import torch
import torch.nn as nn
import time
import typing

from kdv_types import TrainingDomain
from kdv_loss import *
from utils import *
from network import *
from kdv_analysis import linear_combination


def setup_training_domain(
        n_collocation: int,
        n_initial: int,
        n_boundary: int,
        soliton_params: dict[str, torch.Tensor]
        ) -> TrainingDomain:
    """
    Setup the training domain and return it as a custom datatype (found in kdv_types), generating tensors of 
    coordinates and exact soliton solutions where applicable (ic, bc) on model device
    """
    
    device = soliton_params['x_lims'].device
    
    #domain limits
    x0 = soliton_params['x_lims'][0]
    x1 = soliton_params['x_lims'][1]
    t0 = soliton_params['t_lims'][0]
    t1 = soliton_params['t_lims'][1]


    #collocation points
    x_collocation = torch.rand(n_collocation, 1, device=device) * (x1 - x0) + x0
    t_collocation = torch.rand(n_collocation, 1, device=device) * (t1 - t0) + t0

    #ic points
    x_initial = torch.linspace(x0, x1, n_initial, device=device).reshape(-1, 1)
    t_initial = torch.ones_like(x_initial, device=device) * t0
    u_initial = linear_combination(x_initial, t_initial, soliton_params['k_vec'], soliton_params['phi_vec'])

    #bc points
    t_boundary_left = torch.linspace(t0, t1, n_boundary//2, device=device).reshape(-1, 1)
    x_boundary_left = torch.ones_like(t_boundary_left, device=device) * x0
    t_boundary_right = torch.linspace(t0, t1, n_boundary//2, device=device).reshape(-1, 1)
    x_boundary_right = torch.ones_like(t_boundary_right, device=device) * x1
    x_boundary = torch.cat([x_boundary_left, x_boundary_right], dim=0)
    t_boundary = torch.cat([t_boundary_left, t_boundary_right], dim=0)
    u_boundary = torch.zeros_like(x_boundary, device=device)

    domain = TrainingDomain(
        x_collocation,
        t_collocation,
        x_initial,
        t_initial,
        u_initial,
        x_boundary,
        t_boundary,
        u_boundary)

    return domain

def train(
        neural_net: MLP,
        soliton_params: dict[str, torch.Tensor], 
        train_params: dict[str, typing.Any],
        train_weights: dict[str, float], 
        device: torch.DeviceLikeType,
        ) -> dict[str, typing.Any]:
    """
    Calls to setup_training_domain to create its own training domain, then uses a Adam -> L-BFGS optimization scheme
    returning a dict of training data
    """

    #set defaults
    defaults = {
        'adam_epochs': 1000,
        'lbfgs_epochs': 50000,
        'verbose_step': 100,
        'n_collocation': 30000,
        'n_initial': 30000,
        'n_boundary': 30000,
        'adam_lr': 0.001,
        'lbfgs_lr': 1.0,
        'lbfgs_history_size': 100,
        'adaptive_sampling': False,
        'lbfgs_version': 'old', # 'old' or 'new'
        'verbose': True,
        'logging': True
    }
    params = {**defaults, **train_params} #unpack parameters
    losses = init_loss_list() #returns the loss dict
    loss_weights = init_loss_weights(device, train_weights) #returns a torch tensor for compute

    #start wall-clock timer
    start_time = time.time() #export this function to utils later

    domain = setup_training_domain(
        params['n_collocation'],
        params['n_initial'],
        params['n_boundary'],
        soliton_params
    )

    if params['verbose']:
        loss_comps = loss_components(neural_net, domain)
        print_weighted_loss_components(loss_weights, loss_comps, tag='start') 

    #Adam Optimizer
    if params['verbose']: print('Starting Adam optimization...')
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=params['adam_lr'])
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
    if params['verbose']: log_gpu_memory("train start")

    for epoch in range(params['adam_epochs']):
        optimizer.zero_grad(set_to_none=True)
        loss_comps = loss_components(neural_net, domain)
        total_loss = compute_total_loss(loss_weights, loss_comps)
        total_loss.backward()
        optimizer.step()

        if params['logging']:
            update_loss_list(losses, total_loss, loss_comps)

        if params['verbose'] and (epoch % params['verbose_step'] == 0 or epoch == params['adam_epochs'] - 1):
            print(f"Adam - Epoch {epoch}/{params['adam_epochs']}, Total Loss: {total_loss.item():.6e}")

    if params['verbose']: log_gpu_memory("after Adam")

    #adaptive sampling would go here
    

    #L-BFGS optimization
    if params['verbose']: print("\nStarting L-BFGS optimization...")
    
    #This version allows for auto-end per epoch and leaves it up to pytorch without user interference, use when test new features
    if params['lbfgs_version'] == 'test':

        optimizer = torch.optim.LBFGS(neural_net.parameters(),
                                    lr= params['lbfgs_lr'], 
                                    max_iter=params['lbfgs_epochs'],
                                    max_eval=params['lbfgs_epochs']*2,
                                    tolerance_grad=1e-9,
                                    tolerance_change=1e-16,
                                    history_size=params['lbfgs_history_size'],
                                    line_search_fn="strong_wolfe"
                                    )
        
        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss_comps = loss_components(neural_net, domain)
            total_loss = compute_total_loss(loss_weights, loss_comps)
            total_loss.backward()

            update_loss_list(losses, total_loss, loss_comps)

            if params['verbose'] and len(losses['total']) % params['verbose_step'] == 0:
                print(f"L-BFGS - Iteration {len(losses['total']) - params['adam_epochs']}, Total Loss: {total_loss.item():.6e}")
        
            return total_loss
        
        
        optimizer.step(closure)

        if params['verbose']:
            print(f"L-BFGS complete, Final Loss: {losses['total'][-1]:.6e}")
        
    else: #Non-testing version, must know the optimal number of epochs for best use
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

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss_comps = loss_components(neural_net, domain)
            total_loss = compute_total_loss(loss_weights, loss_comps)
            total_loss.backward()
            return total_loss

        for i in range(params['lbfgs_epochs']):
            optimizer.step(closure)

            if params['logging']:
                loss_comps = loss_components(neural_net, domain)
                total_loss = compute_total_loss(loss_weights, loss_comps)
                update_loss_list(losses, total_loss, loss_comps)
            
            if params['verbose'] and (i % params['verbose_step'] == 0 or i == params['lbfgs_epochs'] - 1):
                    if not params['logging']:
                        loss_comps = loss_components(neural_net, domain)
                        total_loss = compute_total_loss(loss_weights, loss_comps)
                    print(f"L-BFGS - Iteration {i+1}/{params['lbfgs_epochs']}, Total Loss: {total_loss.item():.6e}")

    if params['verbose']: log_gpu_memory("after L-BFGS")

    training_stats = {
        'losses': losses,
        'training time': time.time() - start_time
    }

    if params['verbose']: print(f"Training completed in {training_stats['training time']:.2f} s")
    
    if params['verbose']: 
        loss_comps = loss_components(neural_net, domain)
        print_weighted_loss_components(loss_weights, loss_comps, tag='end')

    return training_stats

  
