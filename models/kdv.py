import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm

from .base import PINN

import json
import os
from pathlib import Path
import time
import random  

class KDV_LEGACY(nn.Module):
    def __init__(self, init_params):
        # Set defaults
        defaults = dict(
            num_solitons=1,
            n_hidden_layers=3,
            n_neurons_per_layer=32,
            activation=nn.Tanh,
            seed=None,
            verbose=True,
            use_layernorm=False, 
        )
        # Merge user params with defaults
        params = {**defaults, **init_params}
        self.init_params = params.copy()

        super(KDV_LEGACY, self).__init__() # call constructor of parent class
        
        self.num_solitons = params['num_solitons'] 
        self.verbose = params['verbose'] 
        
        # deterministic seeding across CPU, GPU and numpy
        if params['seed'] is not None:
            self.seed = int(params['seed'])
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
            # enforce deterministic behaviour in cuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # set device to GPU (if available) otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose:
            print(f"Using device: {self.device}")
        
        # create network and move to device
        self.net = PINN(params['n_hidden_layers'], params['n_neurons_per_layer'], params['activation'], params['use_layernorm'])
        self.net.to(self.device)
        
        # Set domain limits and parameters for different number of solitons
        if self.num_solitons == 1:
            self.x_lims = (-30, 30)
            self.t_lims = (-15, 15)

            k = 0.9  # wavenumber
            phi = 0  # phase parameter

            self.k_vector = np.array([k])
            self.phi_vector = np.array([phi])

        elif self.num_solitons == 2:
            self.x_lims = (-35, 50)
            self.t_lims = (-20, 35)

            k1 = np.sqrt(4/4) 
            k2 = np.sqrt(1.2/4) 
            phi1 = 0
            phi2 = 0

            self.k_vector = np.array([k1, k2])
            self.phi_vector = np.array([phi1, phi2])

        elif self.num_solitons == 3:
            k1 = np.sqrt(1)
            k2 = np.sqrt(0.8)
            k3 = np.sqrt(0.5)

            self.x_lims = (-35, 65)
            self.t_lims = (-25, 50)

            phi1 = 0
            phi2 = 0
            phi3 = 0

            self.k_vector = np.array([k1, k2, k3])
            self.phi_vector = np.array([phi1, phi2, phi3])
        
        # setup training and testing domain points
        self.setup_testing_domain()
        
        # setup figure size 
        self.figsize = (10, 6)

        # default batch size for streaming network evaluation during testing
        # keep the full test grid on CPU to save GPU RAM, send it in chunks only when needed
        self.test_batch_size = params.get('test_batch_size', 20_000)
        # ---- memory probe after initialization ----
        self._log_gpu_memory("after __init__")

    def setup_training_domain(self, n_collocation, n_initial, n_boundary):
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

    def setup_testing_domain(self, nx=1000, nt=1000):
        """
        Create a regular grid for testing and visualization.
        
        Sets up a uniform meshgrid covering the entire domain
        for evaluation and visualization of the PINN solution. The grid
        is stored as instance variables for later use.
        """
        # unpack domain limits 
        x0 = self.x_lims[0]
        x1 = self.x_lims[1]
        
        t0 = self.t_lims[0]
        t1 = self.t_lims[1]

        # define points in each dimension *on CPU*  (we will stream them to GPU later when needed, to avoid OOM)
        x = torch.linspace(x0, x1, nx)
        t = torch.linspace(t0, t1, nt)

        self.x_test = x.cpu().numpy()
        self.t_test = t.cpu().numpy()

        # meshgrid is also kept on CPU (for visualization)
        X, T = torch.meshgrid(x, t, indexing='ij')
        self.X_test = X
        self.T_test = T
        self.X_flat_test = X.reshape(-1, 1)
        self.T_flat_test = T.reshape(-1, 1)

        # Set flag indicating test domain has been created
        self.test_domain_created = True

        if self.verbose:
            print(f"Testing domain created with {nx}x{nt} grid points.")

        return

    # LOSS FUNCTIONS 
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
    
    ## TRAINING 
    def train(self, train_params):
        # Set defaults
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
            lbfgs_version='old' # 'old' or 'new'
        )
        params = {**defaults, **train_params}
        self.train_params = params.copy()

        # start wall-clock timer
        start_time = time.time()

        self.setup_training_domain(
            n_collocation=params['n_collocation'],
            n_initial=params['n_initial'],
            n_boundary=params['n_boundary']
        )
        
        # Print weighted loss components at the start
        self.print_weighted_loss_components(label="start")

        # initialize losses list 
        losses = {
            'total': [],
            'initial': [],
            'boundary': [],
            'pde': []
        }
        
        # Phase 1: Adam optimization
        if self.verbose:
            print("Starting Adam optimization...")
        
        optimizer = torch.optim.Adam(self.net.parameters(), lr=params['adam_lr'])
        
        # reset peak stats and log baseline before the heavy optimization loops
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        self._log_gpu_memory("train start")
        
        for epoch in range(params['adam_epochs']):
            optimizer.zero_grad(set_to_none=True)

            # Compute loss components
            comps = self._loss_components()
            
            total_loss = (
                params['w_ic'] * comps['ic'] +
                params['w_bc'] * comps['bc'] +
                params['w_pde'] * comps['pde']
            )
            
            # backpropagation and optimization
            total_loss.backward()
            optimizer.step()
            # store losses
            losses['total'].append(total_loss.item())
            losses['initial'].append(comps['ic'].item())
            losses['boundary'].append(comps['bc'].item())
            losses['pde'].append(comps['pde'].item())
            # print progress
            if self.verbose and (epoch % params['verbose_step'] == 0 or epoch == params['adam_epochs'] - 1):
                print(f"Adam - Epoch {epoch}/{params['adam_epochs']}, Total Loss: {total_loss.item():.6e}")
        
        # Store the number of Adam epochs as an instance variable
        self.adam_epochs = params['adam_epochs']

        # ---- memory probe after Adam optimization ----
        self._log_gpu_memory("after Adam")

        # Adaptive Sampling Step (if enabled) 
        if params.get('adaptive_sampling', False):
            if self.verbose:
                print("\nPerforming adaptive sampling...")
            n_new = params['n_collocation']

            # Create a dense grid (sqrt(n_new)*sqrt(n_new) points)
            x0, x1 = self.x_lims
            t0, t1 = self.t_lims
            n_grid = int(np.sqrt(10 * n_new))  # oversample for better selection
            x_dense = torch.linspace(x0, x1, n_grid)
            t_dense = torch.linspace(t0, t1, n_grid)
            X_dense, T_dense = torch.meshgrid(x_dense, t_dense, indexing='ij')
            X_flat = X_dense.reshape(-1, 1).to(self.device)
            T_flat = T_dense.reshape(-1, 1).to(self.device)

            # Compute residuals in batches to avoid OOM
            batch_size = 1000
            n_points = X_flat.shape[0]
            residuals_list = []
            for i in range(0, n_points, batch_size):
                x_batch = X_flat[i:i+batch_size]
                t_batch = T_flat[i:i+batch_size]
                res_batch = torch.abs(self.compute_pde_residual(x_batch, t_batch)).detach().cpu()
                residuals_list.append(res_batch)
            residuals = torch.cat(residuals_list).numpy().flatten()

            # Select top n_new points
            idx = np.argpartition(-residuals, n_new)[:n_new]
            x_new = X_flat[idx]
            t_new = T_flat[idx]

            # Concatenate to collocation set
            self.x_collocation = torch.cat([self.x_collocation, x_new], dim=0)
            self.t_collocation = torch.cat([self.t_collocation, t_new], dim=0)
            
            if self.verbose:
                print(f"Added {n_new} adaptive collocation points at high-residual locations.")

        # Phase 2: L-BFGS optimization
        if params['lbfgs_version'] == 'old':
            if self.verbose:
                print("\nStarting L-BFGS optimization...")
            
            # L-BFGS requires a closure that reevaluates the model and returns the loss
            def closure():
                optimizer.zero_grad(set_to_none=True)

                comps = self._loss_components()
                
                total_loss = (
                    params['w_ic'] * comps['ic'] +
                    params['w_bc'] * comps['bc'] +
                    params['w_pde'] * comps['pde']
                )
                total_loss.backward()
                
                # Store current loss values
                losses['total'].append(total_loss.item())
                losses['initial'].append(comps['ic'].item())
                losses['boundary'].append(comps['bc'].item())
                losses['pde'].append(comps['pde'].item())
                
                # print progress if verbose
                if self.verbose and len(losses['total']) % params['verbose_step'] == 0:
                    print(f"L-BFGS - Iteration {len(losses['total']) - params['adam_epochs']}, Total Loss: {total_loss.item():.6e}")
                    
                return total_loss
            
            # initialize L-BFGS optimizer
            optimizer = torch.optim.LBFGS(self.net.parameters(),
                                        lr = params['lbfgs_lr'], 
                                        max_iter=params['lbfgs_epochs'],
                                        max_eval=params['lbfgs_epochs']*2,
                                        tolerance_grad=1e-9,
                                        tolerance_change=1e-16,
                                        history_size=params['lbfgs_history_size'],
                                        line_search_fn="strong_wolfe")
            
            # run the optimizer
            optimizer.step(closure)
            
            if self.verbose:
                print(f"L-BFGS complete, Final Loss: {losses['total'][-1]:.6e}")
        elif params['lbfgs_version'] == 'new':
            optimizer = torch.optim.LBFGS(
                self.net.parameters(),
                lr=params['lbfgs_lr'],
                max_iter=1,                  # one accepted iteration per step()
                max_eval=100,
                tolerance_grad=1e-9,
                tolerance_change=1e-16,
                history_size=params['lbfgs_history_size'],
                line_search_fn="strong_wolfe",
            )

            last_vals = {}  # will hold floats from the LAST closure call of each step

            def closure():
                optimizer.zero_grad(set_to_none=True)
                comps = self._loss_components()
                
                total = (
                    params['w_ic'] * comps['ic'] +
                    params['w_bc'] * comps['bc'] +
                    params['w_pde'] * comps['pde']
                )
                total.backward()
                # stash floats for logging once step() returns
                last_vals['ic'] = float(comps['ic'])
                last_vals['bc'] = float(comps['bc'])
                last_vals['pde'] = float(comps['pde'])
                last_vals['total'] = float(total)
                return total

            for it in range(params['lbfgs_epochs']):
                loss = optimizer.step(closure)              # loss from final accepted closure call
                # one log per accepted iteration
                losses['total'].append(float(loss))         # or last_vals['total']
                losses['initial'].append(last_vals['ic'])
                losses['boundary'].append(last_vals['bc'])
                losses['pde'].append(last_vals['pde'])

                if self.verbose and (it % params['verbose_step'] == 0 or it == params['lbfgs_epochs'] - 1):
                    print(f"L-BFGS - Iteration {it+1}/{params['lbfgs_epochs']}, Total Loss: {float(loss):.6e}")

        # save losses as instance variable
        self.losses = losses
        
        # store final loss
        self.final_loss = losses['total'][-1] if losses['total'] else float('nan')
        
        # stop timer and store training duration (in seconds)
        self.training_time = time.time() - start_time

        if self.verbose:
            print(f"Training completed in {self.training_time:.2f} s")

        # ---- memory probe after L-BFGS optimization ----
        self._log_gpu_memory("after LBFGS")

        # after training is complete, print weighted loss components again
        self.print_weighted_loss_components(label="end")

        return
    
    ## TESTING
    def n_soliton(self, x, t, k_vec=None, delta_vec=None):
        """
        Exact KdV N-soliton solution in Hirota form.

        Parameters
        ----------
        x, t       : torch.Tensor (any broadcast-compatible shapes)
        k_vec      : 1-D sequence/array of floats (len N), default -> self.k_vector
        delta_vec  : 1-D sequence/array of floats (len N), default -> self.phi_vector

        Returns
        -------
        torch.Tensor  u(x,t)  (same shape & device as `x`)
        """
        # inputs & dtype
        if k_vec is None:
            k_vec = self.k_vector
        if delta_vec is None:
            delta_vec = self.phi_vector

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

    def test(self, plot_heatmap = False, error_type: str = 'absolute-normalized'):
        """
        Compute error metrics between the predicted and analytical solutions.
        Also plots the time-averaged error over (x, y) as a heatmap (log color scale).
        """
        # compute test solutions
        self.compute_solutions()
        
        # Compute error (array of shape [nx, nt])
        if error_type == 'absolute':
            self.error = torch.abs(self.U_pred - self.U_exact)
        elif error_type == 'absolute-normalized':
            # compute the max value of the exact solution
            max_exact = torch.max(torch.abs(self.U_exact))
            self.error = torch.abs(self.U_pred - self.U_exact) / max_exact
        else:
            raise ValueError(f"Invalid error type: {error_type}")
        
        # Compute mean error
        mae = torch.mean(self.error).item()
        self.mae = mae
        
        # Compute maximum error (L-infinity norm)
        max_error = torch.max(self.error).item()
        self.max_error = max_error
        
        # Print a summary of the error metrics
        if self.verbose:
            print(f"{error_type} error metrics:")
            print(f"Mean: {mae:.6e}")
            print(f"Maximum: {max_error:.6e}")
        
        # Plot error heatmap if requested
        if plot_heatmap:
            self.error_np = self.error.cpu().numpy() # Store error as numpy array for plotting
            plt.figure(figsize=(10, 6))
            
            contour = plt.pcolormesh(self.T_np[0, :], self.X_np[:, 0], self.error_np, 
                                cmap='hot', norm=LogNorm())
            plt.colorbar(contour, label='Error')
            plt.xlabel('Time (t)')
            plt.ylabel('Position (x)')
            plt.tight_layout()
        
        return 

     
    ## PLOTTING 
    
    # profiles
    def extract_profiles(self, t_values, which: str = 'predicted'):
        """Return 1-D profiles of a chosen solution at specified times.

        Parameters
        ----------
        t_values : float or sequence of floats
            Times at which to sample the solution.
        which : {'predicted', 'exact', 'linear'}
            Which solution field to use.

        Returns
        -------
        x : 1-D NumPy array (spatial grid)
        profiles : list[1-D NumPy array]
            Each entry corresponds to `t_values[i]`.
        """

        # Ensure solutions are available
        self.compute_solutions()

        if which not in ('predicted', 'exact', 'linear'):
            raise ValueError("`which` must be one of 'predicted', 'exact', or 'linear'.")

        # Accept scalar or iterable
        if np.isscalar(t_values):
            t_values = [float(t_values)]
        else:
            t_values = list(map(float, t_values))

        # Locate nearest indices along the uniform time grid (axis 1)
        t_axis = self.T_np[0, :]
        indices = [int(np.argmin(np.abs(t_axis - t))) for t in t_values]

        sol_field = self.solutions_np[which]
        profiles = [sol_field[:, idx] for idx in indices]

        return self.X_np[:, 0], profiles

    def plot_profiles(self, t_values, solutions=("predicted",)):
        """Plot profiles of chosen solutions at the requested times.

        Parameters
        ----------
        t_values : float or list of floats
            Times at which to plot the profiles.
        solutions : tuple/list of solution keys
            Combination of "predicted", "exact", "linear" to plot.
        """

        # Ensure that the solutions are computed
        self.compute_solutions()

        # Validate that the times lie within the computed domain
        t_axis = self.T_np[0, :]
        t_min, t_max = t_axis.min(), t_axis.max()
        for t in t_values:
            if not (t_min <= t <= t_max):
                raise ValueError(f"Requested time {t} outside computed domain [{t_min}, {t_max}].")

        # Start figure
        plt.figure(figsize=(15, 4))

        for sol_key in solutions:
            x, profiles = self.extract_profiles(t_values, which=sol_key)
            for t_val, prof in zip(t_values, profiles):
                plt.plot(
                    x,
                    prof,
                    label=f"{sol_key}  t={t_val}",
                )

        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        return

    # losses
    def plot_losses(self, component=None, show_optimizer_switch=True):
        """
        Plot the losses over epochs to visualize training progress.
        
        Parameters:
        -----------
        losses: dict
            Dictionary with loss components
        component: str, list, or None
            If None, plots total loss (default behavior)
            If 'all', plots all loss components
            If a list of strings, plots all specified components
        show_optimizer_switch: bool
            Whether to show a vertical line indicating the switch from Adam to LBFGS optimizer
        """
        plt.figure(figsize=self.figsize)
        
        # Convert single component to list for uniform processing
        if component is None:
            component = ['total']
        elif component == 'all':
            component = list(self.losses.keys())
        elif isinstance(component, str):
            component = [component]
        
        # Plot each requested component
        for comp in component:
            if comp in self.losses:
                plt.plot(self.losses[comp], label=f'{comp} loss')
            else:
                raise ValueError(f"Unknown loss component '{comp}'")
        
        # Add vertical line for optimizer switch if requested
        if show_optimizer_switch and hasattr(self, 'adam_epochs') and self.adam_epochs > 0:
            # Add a vertical line at the Adam-to-LBFGS transition
            plt.axvline(x=self.adam_epochs, color='r', linestyle='--', alpha=0.7)
            
            # Add text annotation
            plt.text(self.adam_epochs + 5, 0.2, 'Adam → L-BFGS', 
                     rotation=90, verticalalignment='center', transform=plt.gca().get_xaxis_transform())
        
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
            
        plt.tight_layout()
        return

    # spacetime
    def plot_spacetime(self, show_data=False):
        """
        Plot the solution as a color map over space and time.
        
        Parameters:
        -----------
        show_data: bool
            Whether to show the training data points (collocation, initial, boundary)
        """
    
        self.compute_solutions()
        plt.figure(figsize=(10, 6))
        
        # Plot the solution as a color map
        contour = plt.pcolormesh(self.T_np[0, :], self.X_np[:, 0], self.U_pred_np, 
                                cmap='plasma', shading='auto')
        plt.colorbar(contour, label='u(x,t)')
        
        # Plot the training data points if requested
        if show_data:
            # Convert collocation points to numpy
            x_coll_np = self.x_collocation.cpu().numpy()
            t_coll_np = self.t_collocation.cpu().numpy()
            
            # Plot with different markers and sizes for clarity
            plt.scatter(t_coll_np, x_coll_np, marker='.', color='black', s=0.3, alpha=0.5, 
                    label='Collocation points')
            plt.scatter(self.t_initial.cpu().numpy(), self.x_initial.cpu().numpy(),
                    marker='x', color='white', s=3, label='Initial condition')
            plt.scatter(self.t_boundary.cpu().numpy(), self.x_boundary.cpu().numpy(),
                    marker='o', color='red', s=1, label='Boundary condition')
            plt.legend(loc='upper right', fontsize='small')
        
        plt.xlabel('Time (t)')
        plt.ylabel('Position (x)')
        plt.tight_layout()
        
        return
    
    # SAVING RESULTS
    def save_model_result(self, filename):
        """
        Save model results to a JSON file. Used when all results are needed, as opposed to save_experiment_run().
        
        Parameters:
        -----------
        filename : str
            Path to save the JSON file
        """

        
        # Ensure predictions exist
        self.compute_solutions()

        # helper function to convert config dict to JSON-serializable format
        def sanitize_config(cfg):
            """Convert config dict to JSON format (handles nn.Tanh -> 'Tanh')."""
            config_json = {}
            for key, value in cfg.items():
                # convert class types (e.g., nn.Tanh) to string names
                if isinstance(value, type):
                    config_json[key] = value.__name__
                else:
                    config_json[key] = value
            return config_json
        
        # compute some metrics 
        abs_error = torch.abs(self.U_pred - self.U_exact)
        mean_absolute_error = torch.mean(abs_error).item()
        
        # Prepare the results dictionary
        results = {
            "domain": {
                "x": self.X_np[:, 0].tolist(),  # First column = x coordinates
                "t": self.T_np[0, :].tolist()   # First row = t coordinates
            },
            "solution": {
                "u_pred": self.U_pred_np.tolist(),
                "u_lin_comb": self.U_lin_comb_np.tolist()
            },
            "losses": {}, 
            "metrics": {
                "training_time": float(getattr(self, "training_time", float('nan'))),
                "mean_absolute_error": mean_absolute_error
            },
            "config": {
                "init": sanitize_config(getattr(self, "init_params", {})),
                "train": sanitize_config(getattr(self, "train_params", {}))
            }
        }
        
        # Add exact solution if available
        results["solution"]["u_exact"] = self.U_exact_np.tolist()
        
        # Add losses if available
        if hasattr(self, 'losses'):
            # Convert any NumPy or PyTorch values to regular Python types
            losses_dict = {}
            for key, values in self.losses.items():
                losses_dict[key] = [float(val) for val in values]
            results["losses"] = losses_dict
        
        # Check if file already exists
        if os.path.exists(filename):
            raise FileExistsError(f"File {filename} already exists. Please choose a different filename.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(results, f)
        
        print(f"Results saved to {filename}")
        
        return

    def save_experiment_run(self, root_path, to_save: tuple = ("loss", "error")):
        """
        Save selected results from the current model. Used only when certain results are needed, rather than everything from the full model, as opposed to save_model_result(). 

        Parameters
        ----------
        root_path : str | Path
            Results folder.
        to_save : tuple[str]
            Which results to save: 'loss' and/or 'error'.
        """
        root = Path(root_path)
        root.mkdir(parents=True, exist_ok=True)

        # 1. LOSS HISTORY
        if "loss" in to_save:
            if not hasattr(self, "losses"):
                raise AttributeError("losses missing - call train() first")
            (root / "losses").mkdir(exist_ok=True)
            np.savez_compressed(
                root / "losses" / f"loss_{self.seed}.npz",
                **{k: np.asarray(v) for k, v in self.losses.items()}
            )

        # 2. POINT-WISE ERROR GRID
        if "error" in to_save:
            if not hasattr(self, "error"):
                raise AttributeError("error missing - call test() first")
            (root / "errors").mkdir(exist_ok=True)
            error_np = self.error.detach().cpu().numpy()
            np.savez_compressed(
                root / "errors" / f"error_{self.seed}.npz",
                error=error_np
            )

        return


    # Utility: GPU-memory logger
    def _log_gpu_memory(self, tag: str = ""):
        """Print current and peak GPU memory usage (MB) with a tag."""
        if self.verbose:
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
                peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                print(f"[gpu mem] {tag:<25} alloc {alloc:7.1f} MB  reserved {reserved:7.1f} MB  peak {peak:7.1f} MB")
            else:
                print(f"[gpu mem] {tag:<25} cuda not available")

    def quick_diagnostics(self):
        with torch.no_grad():
            resid  = self.compute_pde_residual(self.x_collocation, self.t_collocation)
            print("RMS(pde_residual) =", resid.pow(2).mean().sqrt().item())
            print("RMS(ic_residual)  =", (self.net(self.x_initial, self.t_initial)-self.u_initial)
                                            .pow(2).mean().sqrt().item())

    def print_weighted_loss_components(self, label=None):
        """
        Print the weighted loss components (IC, BC, PDE) using current weights.
        Optionally add a label for context (e.g., 'start', 'end').
        """
        w_ic = self.train_params.get('w_ic', 1.0)
        w_bc = self.train_params.get('w_bc', 1.0)
        w_pde = self.train_params.get('w_pde', 1.0)
        
        with torch.no_grad():
            ic = self.compute_initial_loss().item()
            bc = self.compute_boundary_loss().item()
            
        # For PDE, need grad, so do NOT use torch.no_grad()
        x = self.x_collocation.clone().detach().requires_grad_(True)
        t = self.t_collocation.clone().detach().requires_grad_(True)
        resid = self.compute_pde_residual(x, t)
        pde = torch.mean(resid**2).item()
        
        weighted_ic = w_ic * ic
        weighted_bc = w_bc * bc
        weighted_pde = w_pde * pde
        tag = f" [{label}]" if label else ""
        print(f"Weighted losses{tag}: IC={weighted_ic:.3e}, BC={weighted_bc:.3e}, PDE={weighted_pde:.3e}")

    def _loss_components(self):
        ic  = self.compute_initial_loss()
        bc  = self.compute_boundary_loss()
        resid = self.compute_pde_residual(self.x_collocation, self.t_collocation)
        pde = torch.mean(resid**2)
        return dict(ic=ic, bc=bc, pde=pde)