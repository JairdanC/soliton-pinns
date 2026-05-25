"""
This file contains the methods used in computing the loss of the neural network during the training of a PINN
for the KdV equation
"""

import torch
import torch.nn as nn

from kdv import TrainingDomain
from network import MLP

#not finished yet
def compute_pde_loss(neural_net: MLP, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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

    loss = torch.mean(residual**2)

    return loss

def compute_initial_loss(neural_net: MLP, u_ic: torch.Tensor, x_ic: torch.Tensor, t_ic: torch.Tensor) -> torch.Tensor:
    """
    Compute the initial loss for the KdV equation. (ICs)
    """
    u_pred_initial = neural_net(x_ic, t_ic)
    initial_loss = torch.mean((u_pred_initial - u_ic)**2)
    return initial_loss

def compute_boundary_loss(neural_net: MLP, u_bc: torch.Tensor, x_bc: torch.Tensor, t_bc: torch.Tensor) -> torch.Tensor:
    """
    Compute the boundary loss for the KdV equation. (BCs)
    """
    u_pred_boundary = neural_net(x_bc, t_bc)
    boundary_loss = torch.mean((u_pred_boundary - u_bc)**2)
    return boundary_loss

def init_loss_list() -> dict[str, list[float]]:
    losses = {
        'total': [],
        'initial': [],
        'boundary': [],
        'pde': []
    }
    return losses

def init_loss_weights(device, init_weights: dict[str, float] = None) -> torch.Tensor:
    defaults = {
        'w_ic': 1.0,
        'w_bc': 1.0,
        'w_pde': 1.0
    }
    if init_weights is not None:
        dict_weights = defaults | init_weights #overwrites any existing key with the user defined value
        weights = torch.tensor(list(dict_weights.values()), device=device)
        return weights
    else: return torch.tensor(list(defaults.values()), device=device)

#touch up once done with the KDV class
def loss_components(neural_net: MLP, domain: TrainingDomain) -> torch.Tensor:
    ic = compute_initial_loss(neural_net, domain.u_ic, domain.x_ic, domain.t_ic)
    bc = compute_boundary_loss(neural_net, domain.u_bc, domain.x_bc, domain.t_bc)
    pde = compute_pde_loss(neural_net, domain.x_coll, domain.t_coll)

    components = torch.tensor([ic, bc, pde], device=domain.x_coll.device)
    return components

def compute_total_loss(weights: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    total = torch.dot(weights, components)
    return total

def update_loss_list(losses: dict[str, list[float]], total_loss: torch.Tensor, loss_comps: dict[str, torch.Tensor]) -> None:
    losses['total'].append(float(total_loss))
    losses['initial'].append(float(loss_comps['ic']))
    losses['boundary'].append(float(loss_comps['bc']))
    losses['pde'].append(float(loss_comps['pde']))

def update_last_vals(last_vals: dict[str, float], total_loss: torch.Tensor, loss_comps: dict[str, torch.Tensor]) -> None:
    last_vals['total'] = float(total_loss)
    last_vals['initial'] = float(loss_comps['ic'])
    last_vals['boundary'] = float(loss_comps['bc'])
    last_vals['pde'] = float(loss_comps['pde'])

def append_last_vals(losses: dict[str, list[float]], last_vals: dict[str, float]) -> None:
    losses['total'].append(last_vals['total'])
    losses['initial'].append(last_vals['initial'])
    losses['boundary'].append(last_vals['boundary'])
    losses['pde'].append(last_vals['pde'])