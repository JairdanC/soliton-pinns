"""
This file contains the methods used in computing the loss of the neural network during the training of a PINN
for the KdV equation
"""

import torch
import torch.nn as nn

#not finished yet
def compute_pde_loss(neuralNet, x, t):
    """
    Compute PDE residual for the KdV equation: u_t + 6u*u_x + u_xxx = 0
    """
    # copies of x and t that require gradients
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    # forward pass = u(x,t)
    u = neuralNet(x, t)

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

def compute_initial_loss(neuralNet, u_ic, x_ic, t_ic):
    """
    Compute the initial loss for the KdV equation. (ICs)
    """
    u_pred_initial = neuralNet(x_ic, t_ic)
    initial_loss = torch.mean((u_pred_initial - u_ic)**2)
    return initial_loss

def compute_boundary_loss(neuralNet, u_bc, x_bc, t_bc):
    """
    Compute the boundary loss for the KdV equation. (BCs)
    """
    u_pred_boundary = neuralNet(x_bc, t_bc)
    boundary_loss = torch.mean((u_pred_boundary - u_bc)**2)
    return boundary_loss

def init_loss_list():
    losses = {
        'total': [],
        'initial': [],
        'boundary': [],
        'pde': []
    }
    return losses

def init_loss_weights(**init_weights: dict[str, float]):
    default_weights = {
        'w_ic': 1.0,
        'w_bc': 1.0,
        'w_pde': 1.0
    }
    init_weights.update(default_weights)
    weights = torch.tensor(list(init_weights.values()))
    return weights

#touch up once done with the KDV class
def loss_components(neuralNet, x, t, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc):
    ic = compute_initial_loss(neuralNet, u_ic, x_ic, t_ic)
    bc = compute_boundary_loss(neuralNet, u_bc, x_bc, t_bc)
    pde = compute_pde_loss(neuralNet, x, t)

    components = torch.tensor([ic, bc, pde])

    return components

def total_loss(weights, components):
    total = torch.dot(weights, components)
    return total