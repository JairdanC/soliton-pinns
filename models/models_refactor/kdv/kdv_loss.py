"""
This file contains the methods used in computing the loss of the neural network during the training of a PINN
for the KdV equation
"""

import torch
import torch.nn as nn

def compute_pde_loss(self, x, t):
    """
    Compute PDE residual for the KdV equation: u_t + 6u*u_x + u_xxx = 0
    """
    # copies of x and t that require gradients
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    # forward pass = u(x,t)
    u = self.net(x, t)

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

def compute_initial_loss(neuralNet):
    """
    Compute the initial loss for the KdV equation. (ICs)
    """
    u_pred_initial = self.net(self.x_initial, self.t_initial)
    initial_loss = torch.mean((u_pred_initial - self.u_initial)**2)
    return initial_loss

def compute_boundary_loss(neuralNet: nn.Module, x, t):

    """
    Compute the boundary loss for the KdV equation. (BCs)
    """
    u_pred_boundary = neuralNet.net(x, t)
    boundary_loss = torch.mean((u_pred_boundary - neuralNet.system['u_boundary'])**2)
    return boundary_loss

def init_loss_list():
    losses = {
        'total': [],
        'initial': [],
        'boundary': [],
        'pde': []
    }
    return losses

def init_loss_weights(**init_weights):
    default_weights = {
        'w_ic': 1.0,
        'w_bc': 1.0,
        'w_pde': 1.0
    }
    init_weights.update(default_weights)
    weights = torch.tensor(list(init_weights.values))
    return weights

def loss_components():
    ic = compute_initial_loss()
    bc = compute_boundary_loss()
    pde = compute_pde_loss()

    components = torch.tensor(list(ic, bc, pde))

    return components

def total_loss(weights, components):
    total = torch.dot(weights, components)
    return total