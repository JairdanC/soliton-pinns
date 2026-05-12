"""
This file contains the analytical methods used for obtaining the exact solutions of a KdV soliton,
used across training a PINN on the KdV equation for computing error and initial conditions and visualization
"""

import torch

def n_soliton(x, t, k_vec, delta_vec):
        """
        Exact KdV N-soliton solution in Hirota form.

        Parameters
        ----------
        x, t       : torch.Tensor (any broadcast-compatible shapes)
        k_vec      : 1-D sequence/array of floats (len N)
        delta_vec  : 1-D sequence/array of floats (len N)

        Returns
        -------
        torch.Tensor  u(x,t)  (same shape & device as `x`)
        """

        # promote to float64 tensors on the same device
        x_d = x.to(torch.float64)
        t_d = t.to(torch.float64)
        k = torch.as_tensor(k_vec, dtype=torch.float64, device=x.device)
        d = torch.as_tensor(delta_vec, dtype=torch.float64, device=x.device)

        # number of solitons
        n = len(k)

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
        
def phase_shifts(self):
        """
        Compute the phase shifts for the linear combination of single-soliton solutions.

        Returns
        -------
        list[float]  phase shifts (len N)
        """

        def aij(ki, kj):
            ki = float(ki); kj = float(kj)
            return 2*np.log((ki - kj) / (ki + kj))

        if self.num_solitons == 1:
            return [0.0]
        
        elif self.num_solitons == 2:
            k1, k2 = self.k_vector

            return [0, aij(k1, k2)]
        
        elif self.num_solitons == 3:
            k1, k2, k3 = self.k_vector

            a12 = aij(k1, k2)
            a13 = aij(k1, k3)
            a23 = aij(k2, k3)

            return [0.0, a12, a13 + a23]
        
        else:
            raise ValueError("phase_shifts implemented only for N = 1, 2 or 3 solitons.")

def linear_combination(self, x, t):
    """Return the linear superposition of single-soliton solutions as a torch tensor."""

    # compute the phase shifts
    shifts = self.phase_shifts() 

    # initialize the linear combination
    u = torch.zeros_like(x)

    # add the single-soliton solutions with their corresponding phase shifts
    for k_i, phi_i, delta_i in zip(self.k_vector, self.phi_vector, shifts):

        # this returns the single-soliton solution for the given k_i, phi_i and delta_i
        u += self.n_soliton(x, t, [k_i], [phi_i + delta_i]) 

    return u

def compute_solutions(self, recompute: bool = False, to_numpy: bool = True):
    """Evaluate and store all relevant solutions (exact, pinn-predicted, and linear combination) on the test grid.

    Creates/uses `self.solutions` (torch tensors) and `self.solutions_np`
    (NumPy arrays) with the keys `predicted`, `exact`, `linear`.

    Parameters
    ----------
    recompute : bool, default False
        Force recomputation even if cached data exist.
    to_numpy : bool, default True
        Also create NumPy views of the solutions for quick plotting.
    """

    # Skip expensive work if the solutions are already computed
    if (not recompute) and hasattr(self, 'solutions'):
        return self.solutions

    # (1) PINN prediction – stream in manageable chunks to the GPU
    with torch.inference_mode():
        # batch size for the test grid
        B = getattr(self, "test_batch_size", 20_000)

        # number of points in the test grid
        n_pts = self.X_flat_test.shape[0]

        # list to store the predictions
        pred_chunks = []

        # loop through the test grid in chunks
        for i in range(0, n_pts, B):
            # slice CPU test grid tensors, move slice to GPU, run net, bring result back to CPU
            x_batch = self.X_flat_test[i:i+B].to(self.device, non_blocking=True)
            t_batch = self.T_flat_test[i:i+B].to(self.device, non_blocking=True)

            # run the network on the chunk and store the predictions
            pred_chunks.append(self.net(x_batch, t_batch).cpu())

            # free GPU chunk ASAP
            del x_batch, t_batch

        # concatenate the predictions
        U_pred_flat = torch.cat(pred_chunks, dim=0)

        # reshape the predictions to the shape of the test grid
        U_pred = U_pred_flat.reshape(self.X_test.shape)

        # release GPU cache now that heavy lifting is done
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # (2) Exact analytical solution 
    U_exact = self.n_soliton(self.X_test, self.T_test)

    # (3) Linear combination of single-soliton solutions
    U_linear = self.linear_combination(self.X_test, self.T_test)

    # store solutions as dictionaries
    self.solutions = {
        'predicted': U_pred,
        'exact': U_exact,
        'linear': U_linear,
    }
    self.solutions_np = {k: v.detach().cpu().numpy() for k, v in self.solutions.items()}
    
    # torch versions (used for error computation in test())
    self.U_pred = U_pred
    self.U_exact = U_exact
    self.U_lin_comb = U_linear
    
    # numpy versions (used for plotting and saving)
    self.U_pred_np = self.solutions_np['predicted']
    self.U_exact_np = self.solutions_np['exact']
    self.U_lin_comb_np = self.solutions_np['linear']
    self.X_np = self.X_test.cpu().numpy()
    self.T_np = self.T_test.cpu().numpy()

    # ---- memory probe after computing the solutions ----
    self._log_gpu_memory("inside compute_solutions")

    return