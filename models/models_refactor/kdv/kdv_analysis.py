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