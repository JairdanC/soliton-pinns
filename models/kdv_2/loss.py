"""
This file contains the methods used in computing the loss of the neural network during the training of a PINN
for the KdV equation
"""
#Libraries
import torch
import torch.nn as nn
#Types
from .types import TrainingDomain
from ..network import MLP
#Methods
from .methods import momentum_integral, energy_integral

def compute_pde_residual(neural_net: MLP, 
                     x: torch.Tensor, 
                     t: torch.Tensor
                     ) -> torch.Tensor:
    """
    Compute PDE residual for the KdV equation: u_t + 6u*u_x + u_xxx = 0
    """

    # copies of x and t that require gradients
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    # forward pass = u(x,t)
    u = neural_net(x, t)

    # calculate derivatives needed for KdV
    # first-order derivatives
    u_grad = torch.autograd.grad(
        outputs=u, 
        inputs=[t, x], 
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )
    u_t = u_grad[0]  # temporal derivative
    u_x = u_grad[1]  # first spatial derivative

    # second-order spatial derivative (u_xx)
    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    # third-order spatial derivative (u_xxx)
    u_xxx = torch.autograd.grad(
        outputs=u_xx,
        inputs=x,
        grad_outputs=torch.ones_like(u_xx),
        create_graph=True
    )[0]

    # KdV equation residual
    residual = u_t + 6.0 * u * u_x + u_xxx

    return residual

def compute_momentum_int_residual(neural_net: MLP,
                                  u_momentum: torch.Tensor,
                                  x_momentum: torch.Tensor,
                                  t_momentum: torch.Tensor
                                  ) -> torch.Tensor:
    
    x_flat = x_momentum.flatten()
    t_flat = t_momentum.flatten()
    x_grid, t_grid = torch.meshgrid(x_flat, t_flat, indexing='ij')
    x_net = x_grid.reshape(-1, 1)
    t_net = t_grid.reshape(-1, 1)
    u_pred_flat = neural_net(x_net, t_net)
    u_pred = u_pred_flat.reshape(x_grid.shape)

    momentum_pred = momentum_integral(u_pred, x_momentum)

    residual = momentum_pred - u_momentum

    return residual


def compute_energy_int_residual(neural_net: MLP,
                                u_energy: torch.Tensor,
                                x_energy: torch.Tensor,
                                t_energy: torch.Tensor
                                ) -> torch.Tensor:

    x_flat = x_energy.flatten()
    t_flat = t_energy.flatten()
    x_grid, t_grid = torch.meshgrid(x_flat, t_flat, indexing='ij')
    x_net = x_grid.reshape(-1, 1)
    t_net = t_grid.reshape(-1, 1)
    u_pred_flat = neural_net(x_net, t_net)
    u_pred = u_pred_flat.reshape(x_grid.shape)

    energy_pred = energy_integral(u_pred, x_energy)

    residual = energy_pred - u_energy

    return residual
    

def compute_initial_loss(neural_net: MLP, 
                         u_ic: torch.Tensor, 
                         x_ic: torch.Tensor, 
                         t_ic: torch.Tensor
                         ) -> torch.Tensor:
    """
    Compute the initial loss for the KdV equation. (ICs)
    """

    u_pred_initial = neural_net(x_ic, t_ic)
    initial_loss = torch.mean((u_pred_initial - u_ic)**2)
    return initial_loss

def compute_boundary_loss(neural_net: MLP, 
                          u_bc: torch.Tensor, 
                          x_bc: torch.Tensor, 
                          t_bc: torch.Tensor
                          ) -> torch.Tensor:
    """
    Compute the boundary loss for the KdV equation. (BCs)
    """

    u_pred_boundary = neural_net(x_bc, t_bc)
    boundary_loss = torch.mean((u_pred_boundary - u_bc)**2)
    return boundary_loss

def init_loss_list() -> dict[str, list[float]]:
    """
    Initializes and returns a loss list to hold the logged training losses
    """

    losses = {
        'total': [],
        'initial': [],
        'boundary': [],
        'pde': [],
        'momentum': [],
        'energy': []
    }
    return losses

def init_loss_weights(device, 
                      init_weights: dict[str, float]
                      ) -> torch.Tensor:
    """
    Initializes the loss weights per component
    """

    defaults = {
        'w_ic': 1.0,
        'w_bc': 1.0,
        'w_pde': 1.0,
        'w_momentum': 1.0,
        'w_energy': 1.0,
        
    }
    if init_weights is not None:
        dict_weights = defaults | init_weights #overwrites any existing key with the user defined value
        weights = torch.tensor(list(dict_weights.values()), device=device)
        return weights
    else: return torch.tensor(list(defaults.values()), device=device)

def loss_components(neural_net: MLP,
                    domain: TrainingDomain
                    ) -> torch.Tensor:
    """
    Calculates the loss per-component and returns it as a stacked torch tensor  
    """

    ic = compute_initial_loss(neural_net, domain.u_ic, domain.x_ic, domain.t_ic)
    bc = compute_boundary_loss(neural_net, domain.u_bc, domain.x_bc, domain.t_bc)
    pde = torch.mean(compute_pde_residual(neural_net, domain.x_coll, domain.t_coll)**2)

    if domain.t_momentum is not None and domain.t_momentum.numel() > 0: 
        momentum = torch.mean(compute_momentum_int_residual(neural_net, domain.u_momentum, domain.x_ic, domain.t_momentum)**2)
    else:
        momentum = torch.zeros_like(ic)
    if domain.t_energy is not None and domain.t_energy.numel() > 0: 
        energy = torch.mean(compute_energy_int_residual(neural_net, domain.u_energy, domain.x_ic, domain.t_energy)**2)
    else:
        energy = torch.zeros_like(ic)

    

    components = torch.stack([ic, bc, pde, momentum, energy])
    return components
    

def update_loss_list(losses: dict[str, list[float]], 
                     total_loss: torch.Tensor, 
                     loss_comps: torch.Tensor
                     ) -> None:
    """
    Updates the loss list formatted as a dict passed as a parameter,
    used for logging
    """

    losses['total'].append(float(total_loss))
    losses['initial'].append(float(loss_comps[0]))
    losses['boundary'].append(float(loss_comps[1]))
    losses['pde'].append(float(loss_comps[2]))
    losses['momentum'].append(float(loss_comps[3]))
    losses['energy'].append(float(loss_comps[4]))

def update_last_vals(last_vals: dict[str, float], 
                     total_loss: torch.Tensor, 
                     loss_comps: torch.Tensor
                     ) -> None:
    """
    Update the lost dict used for state tracking
    """

    last_vals['total'] = float(total_loss)
    last_vals['initial'] = float(loss_comps[0])
    last_vals['boundary'] = float(loss_comps[1])
    last_vals['pde'] = float(loss_comps[2])
    last_vals['momentum'] = float(loss_comps[3])
    last_vals['energy'] = float(loss_comps[4])

def append_last_vals(losses: dict[str, list[float]],
                     last_vals: dict[str, float]
                     ) -> None:
    """
    Update the loss list using a state dict
    """
    
    losses['total'].append(last_vals['total'])
    losses['initial'].append(last_vals['initial'])
    losses['boundary'].append(last_vals['boundary'])
    losses['pde'].append(last_vals['pde'])
    losses['momentum'].append(last_vals['momentum'])
    losses['energy'].append(last_vals['energy'])