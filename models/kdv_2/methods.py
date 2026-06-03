"""
This file contains the analytical methods used for obtaining the exact solutions of a KdV soliton,
used across training a PINN on the KdV equation for computing error and initial conditions and visualization
"""

import torch

@torch.compile(fullgraph=True)
def n_soliton(x: torch.Tensor, 
              t: torch.Tensor, 
              k_vec: torch.Tensor,
              delta_vec: torch.Tensor
              ) -> torch.Tensor:
    """
    Uses Hirota form to evaluate an exact solution to n solitons with the 
    passed soliton parameters
    """

    # promote to float64 tensors on the same device
    x_d = x.to(torch.float64)
    t_d = t.to(torch.float64)
    k = torch.as_tensor(k_vec, dtype=torch.float64, device=x.device)
    d = torch.as_tensor(delta_vec, dtype=torch.float64, device=x.device)

    n = k.numel()

    # single soliton solution
    if n == 1:
        k1  = k
        d1  = d
        eta1 = k1 * (x_d - k1**2 * t_d) + d1
        f    = 1.0 + torch.exp(eta1)
        fx   = k1 * torch.exp(eta1)
        fxx  = k1**2 * torch.exp(eta1)

        u = 2.0 * (f * fxx - fx**2) / f**2
        return u.to(x.dtype)
    
    # two soliton solution
    elif n == 2:
        k1, k2 = k
        d1, d2 = d

        eta1 = k1 * (x_d - k1**2 * t_d) + d1
        eta2 = k2 * (x_d - k2**2 * t_d) + d2

        A12 = ((k1 - k2) / (k1 + k2))**2

        exp1 = torch.exp(eta1)
        exp2 = torch.exp(eta2)
        exp12 = torch.exp(eta1 + eta2)

        f   = 1.0 + exp1 + exp2 + A12 * exp12
        fx  = k1 * exp1 + k2 * exp2 + A12 * (k1 + k2) * exp12
        fxx = k1**2 * exp1 + k2**2 * exp2 + A12 * (k1 + k2)**2 * exp12

        u = 2.0 * (f * fxx - fx**2) / f**2

        return u.to(x.dtype)

    # three soliton solution
    elif n == 3:
        k1, k2, k3 = k
        d1, d2, d3 = d

        eta1 = k1 * (x_d - k1**2 * t_d) + d1
        eta2 = k2 * (x_d - k2**2 * t_d) + d2
        eta3 = k3 * (x_d - k3**2 * t_d) + d3

        A12 = ((k1 - k2) / (k1 + k2))**2
        A13 = ((k1 - k3) / (k1 + k3))**2
        A23 = ((k2 - k3) / (k2 + k3))**2

        exp1   = torch.exp(eta1)
        exp2   = torch.exp(eta2)
        exp3   = torch.exp(eta3)
        exp12  = torch.exp(eta1 + eta2)
        exp13  = torch.exp(eta1 + eta3)
        exp23  = torch.exp(eta2 + eta3)
        exp123 = torch.exp(eta1 + eta2 + eta3)

        f = (
            1.0
            + exp1 + exp2 + exp3
            + A12 * exp12 + A13 * exp13 + A23 * exp23
            + A12 * A13 * A23 * exp123
        )

        fx = (
            k1 * exp1 + k2 * exp2 + k3 * exp3
            + A12 * (k1 + k2) * exp12
            + A13 * (k1 + k3) * exp13
            + A23 * (k2 + k3) * exp23
            + A12 * A13 * A23 * (k1 + k2 + k3) * exp123
        )

        fxx = (
            k1**2 * exp1 + k2**2 * exp2 + k3**2 * exp3
            + A12 * (k1 + k2)**2 * exp12
            + A13 * (k1 + k3)**2 * exp13
            + A23 * (k2 + k3)**2 * exp23
            + A12 * A13 * A23 * (k1 + k2 + k3)**2 * exp123
        )

        u = 2.0 * (f * fxx - fx**2) / f**2

        return u.to(x.dtype)
    else:
        raise ValueError("n_soliton implemented only for N = 1, 2 or 3 solitons")

@torch.compile(fullgraph=True)
def phase_shifts(k_vec: torch.Tensor) -> torch.Tensor:
    """
    Compute the phase shifts for the linear combination of single-soliton solutions.
    """


    def aij(ki, kj):
        return 2*torch.log((ki - kj) / (ki + kj))
    
    device = k_vec.device
    n = len(k_vec)

    if n == 1: 
        return torch.tensor([0.0], device=device)
    elif n == 2:
        k1, k2 = k_vec

        return torch.tensor([0.0, aij(k1, k2)], device=device)
    elif n == 3:
        k1, k2, k3 = k_vec

        a12 = aij(k1, k2)
        a13 = aij(k1, k3)
        a23 = aij(k2, k3)
        return torch.tensor([0.0, a12, a13 + a23], device=device)
    else:
        raise ValueError('k_vec length is not equal to 1, 2, or 3, an invalid number of solitons phases')
    
@torch.compile(fullgraph=True)
def linear_combination(x: torch.Tensor,
                       t: torch.Tensor, 
                       k_vec: torch.Tensor, 
                       phi_vec: torch.Tensor
                       ) -> torch.Tensor:
    """
    Computes the linear solution of n soliton by creating a linear combination of single
    solitons, excluding the nonlinearity
    """

    shifts = phase_shifts(k_vec=k_vec)
    u = torch.zeros_like(x)

    for k_i, phi_i, delta_i in zip(k_vec, phi_vec, shifts):
        k_1d = k_i.unsqueeze(0)
        phase_1d = (phi_i + delta_i).unsqueeze(0)
        temp = n_soliton(x, t, k_1d, phase_1d)
        u = u + temp 
        
    return u

