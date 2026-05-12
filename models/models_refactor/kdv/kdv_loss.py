"""
This file contains the methods used in computing the loss of the neural network during the training of a PINN
for the KdV equation
"""

import torch

def compute_pde_residual(self, x, t):
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

    return residual

def compute_initial_loss(self):
    """
    Compute the initial loss for the KdV equation. (ICs)
    """
    u_pred_initial = self.net(self.x_initial, self.t_initial)
    initial_loss = torch.mean((u_pred_initial - self.u_initial)**2)
    return initial_loss

def compute_boundary_loss(self):

    """
    Compute the boundary loss for the KdV equation. (BCs)
    """
    u_pred_boundary = self.net(self.x_boundary, self.t_boundary)
    boundary_loss = torch.mean((u_pred_boundary - self.u_boundary)**2)
    return boundary_loss

def _loss_components(self):
        ic  = self.compute_initial_loss()
        bc  = self.compute_boundary_loss()
        resid = self.compute_pde_residual(self.x_collocation, self.t_collocation)
        pde = torch.mean(resid**2)
        return dict(ic=ic, bc=bc, pde=pde)