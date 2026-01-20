import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .base import PINN3D

import json
import os

import copy
import random  

from pathlib import Path

class KP(nn.Module):

    def __init__(self, n_hidden_layers=3, n_neurons_per_layer=32, k=None, P=None, t_lims=(-5, 5), verbose=True, seed=None):
        super(KP, self).__init__() # call constructor of parent class

        # Validate k and P
        if k is None or P is None:
            raise ValueError("Both k and P must be provided and must be sequences of the same length.")
        if not (hasattr(k, '__len__') and hasattr(P, '__len__')):
            raise TypeError("k and P must be sequences (e.g., list, tuple, or array).")
        if len(k) != len(P):
            raise ValueError(f"k and P must have the same length. Got len(k)={len(k)}, len(P)={len(P)}.")

        # Store as tuples to ensure immutability
        self.k = tuple(k)
        self.P = tuple(P)
        self.n = len(self.k)  # number of solitons, always matches k and P
        self.verbose = verbose

        # deterministic seeding across CPU, GPU and numpy
        if seed is not None:
            self.seed = int(seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # domain limits
        self.t_lims = t_lims
        self.x_lims = (-30, 30)
        self.y_lims = (-30, 30)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose:
            print(f"Using device: {self.device}")
            print(f" Solving KP-II equation with {self.n} solitons.")

        # Create the PINN3D model and move to device
        self.net = PINN3D(n_hidden_layers, n_neurons_per_layer, activation=nn.Tanh)
        self.net.to(self.device)

        self.setup_testing_domain()   # Will generate test grid for visualization

    # TRAINING 
    def setup_training_domain(self, n_collocation, n_initial, n_boundary):
        """
        Setup the domain points for training: collocation, initial, and boundary points. Computes and stores the corresponding exact solution values at the domain points for initial and boundary conditions.
        """
        # Unpack domain limits
        x0, x1 = self.x_lims
        y0, y1 = self.y_lims
        t0, t1 = self.t_lims
        device = self.device

        # 1. Collocation points (random in the domain)
        x_collocation = torch.rand(n_collocation, 1, device=device) * (x1 - x0) + x0
        y_collocation = torch.rand(n_collocation, 1, device=device) * (y1 - y0) + y0
        t_collocation = torch.rand(n_collocation, 1, device=device) * (t1 - t0) + t0

        # 2. Initial condition points, uniform grid in (x, y) for t=t0
        n_side = int(n_initial ** 0.5)
        x_initial = torch.linspace(x0, x1, n_side, device=device)
        y_initial = torch.linspace(y0, y1, n_side, device=device)
        X_init, Y_init = torch.meshgrid(x_initial, y_initial, indexing='xy')
        t_initial = torch.ones_like(X_init) * t0
        u_initial = self.exact_solution(X_init, Y_init, t_initial)
        x_initial = X_init.reshape(-1, 1)
        y_initial = Y_init.reshape(-1, 1)
        t_initial = t_initial.reshape(-1, 1)
        u_initial = u_initial.reshape(-1, 1)

        # 3. Boundary condition points (spatial boundary sampled over time)
        # We enforce Dirichlet data u(x,y,t) = u_exact(x,y,t) on the perimeter:
        #   x=x0, x=x1, y=y0, y=y1 for t in [t0, t1].
        n_b_side = n_boundary // 4
        t_b = torch.linspace(t0, t1, n_b_side, device=device)
        y_b = torch.linspace(y0, y1, n_b_side, device=device)
        x_b = torch.linspace(x0, x1, n_b_side, device=device)

        # Edge 1: x = x0, (y, t) grid
        Xb1 = torch.full((n_b_side, 1), x0, device=device)
        Yb1, Tb1 = torch.meshgrid(y_b, t_b, indexing='xy')
        Xb1 = Xb1.expand_as(Yb1)
        # Edge 2: x = x1, (y, t) grid
        Xb2 = torch.full((n_b_side, 1), x1, device=device)
        Yb2, Tb2 = torch.meshgrid(y_b, t_b, indexing='xy')
        Xb2 = Xb2.expand_as(Yb2)
        # Edge 3: y = y0, (x, t) grid
        Yb3 = torch.full((n_b_side, 1), y0, device=device)
        Xb3, Tb3 = torch.meshgrid(x_b, t_b, indexing='xy')
        Yb3 = Yb3.expand_as(Xb3)
        # Edge 4: y = y1, (x, t) grid
        Yb4 = torch.full((n_b_side, 1), y1, device=device)
        Xb4, Tb4 = torch.meshgrid(x_b, t_b, indexing='xy')
        Yb4 = Yb4.expand_as(Xb4)

        # Flatten and stack boundary coordinates into (N, 1) tensors
        x_boundary = torch.cat([
            Xb1.reshape(-1, 1), Xb2.reshape(-1, 1), Xb3.reshape(-1, 1), Xb4.reshape(-1, 1)
        ], dim=0)
        y_boundary = torch.cat([
            Yb1.reshape(-1, 1), Yb2.reshape(-1, 1), Yb3.reshape(-1, 1), Yb4.reshape(-1, 1)
        ], dim=0)
        t_boundary = torch.cat([
            Tb1.reshape(-1, 1), Tb2.reshape(-1, 1), Tb3.reshape(-1, 1), Tb4.reshape(-1, 1)
        ], dim=0)

        # compute the exact solution at the boundary points
        u_boundary = self.exact_solution(x_boundary, y_boundary, t_boundary)

        # store the domain points and exact solutions at initial and boundary points as class attributes
        self.x_collocation = x_collocation
        self.y_collocation = y_collocation
        self.t_collocation = t_collocation
        self.x_initial = x_initial
        self.y_initial = y_initial
        self.t_initial = t_initial
        self.u_initial = u_initial
        self.x_boundary = x_boundary
        self.y_boundary = y_boundary
        self.t_boundary = t_boundary
        self.u_boundary = u_boundary

        # print summary
        if self.verbose:
            print(f"Training domain setup complete:")
            print(f"  Collocation points: {n_collocation}")
            print(f"  Initial points: {x_initial.shape[0]}")
            print(f"  Boundary points: {x_boundary.shape[0]}")

    def compute_pde_residual(self, x, y, t):
        """
        Compute the KP-II equation residual at given points (x, y, t):
        (u_t + 6uu_x + u_xxx)_x + 3u_yy = 0
        """
        # Ensure gradients
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)

        # Forward pass
        u = self.net(x, y, t)

        # First derivatives
        grads = torch.autograd.grad(
            outputs=u,
            inputs=[x, y, t],
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )
        u_x, u_y, u_t = grads[0], grads[1], grads[2]

        # Second derivatives
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0]
        u_yy = torch.autograd.grad(
            outputs=u_y,
            inputs=y,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True,
            retain_graph=True
        )[0]

        # Third derivative
        u_xxx = torch.autograd.grad(
            outputs=u_xx,
            inputs=x,
            grad_outputs=torch.ones_like(u_xx),
            create_graph=True,
            retain_graph=True
        )[0]

        # KP-II residual
        F = u_t + 6.0 * u * u_x + u_xxx
        F_x = torch.autograd.grad(
            outputs=F,
            inputs=x,
            grad_outputs=torch.ones_like(F),
            create_graph=True,
            retain_graph=True
        )[0]
        residual = F_x + 3.0 * u_yy
        return residual

    def loss_fn(self):
        """
        Compute the total loss function: combining data and PDE losses. Returns both the total loss and individual components.
        """
        # Data loss (initial and boundary)
        u_pred_initial = self.net(self.x_initial, self.y_initial, self.t_initial)
        u_pred_boundary = self.net(self.x_boundary, self.y_boundary, self.t_boundary)
        initial_loss = torch.mean((u_pred_initial - self.u_initial) ** 2)
        boundary_loss = torch.mean((u_pred_boundary - self.u_boundary) ** 2)
        data_loss = initial_loss + boundary_loss

        # PDE residual loss
        residual = self.compute_pde_residual(self.x_collocation, self.y_collocation, self.t_collocation)
        pde_loss = torch.mean(residual ** 2)

        # Loss weights
        data_loss_weight = 1.0
        pde_loss_weight = 10.0

        # Total loss
        total_loss = data_loss_weight * data_loss + pde_loss_weight * pde_loss

        return total_loss, initial_loss, boundary_loss, pde_loss

    def train(self, adam_epochs=1000, lbfgs_epochs=50000, verbose_step=100, n_collocation=50000, n_initial=10000, n_boundary=500):
        """
        Train the PINN using Adam followed by L-BFGS. Store and return loss history.
        """
        self.setup_training_domain(n_collocation=n_collocation, n_initial=n_initial, n_boundary=n_boundary)

        # Initialize loss history
        losses = { 
            'total': [],
            'initial': [],
            'boundary': [],
            'pde': []
        }

        # Phase 1: Adam optimization
        optimizer = torch.optim.Adam(self.net.parameters())
        for epoch in range(adam_epochs):
            optimizer.zero_grad()
            total_loss, initial_loss, boundary_loss, pde_loss = self.loss_fn()
            total_loss.backward()
            optimizer.step()
            losses['total'].append(total_loss.item())
            losses['initial'].append(initial_loss.item())
            losses['boundary'].append(boundary_loss.item())
            losses['pde'].append(pde_loss.item())
            if self.verbose and (epoch % verbose_step == 0 or epoch == adam_epochs - 1):
                print(f"Adam - Epoch {epoch}/{adam_epochs}, Total Loss: {total_loss.item():.6e}")

        self.adam_epochs = adam_epochs

        if self.verbose:
            print("\nStarting L-BFGS optimization...")

        # Phase 2: L-BFGS optimization
        def closure():
            optimizer.zero_grad()
            total_loss, initial_loss, boundary_loss, pde_loss = self.loss_fn()
            total_loss.backward()
            losses['total'].append(total_loss.item())
            losses['initial'].append(initial_loss.item())
            losses['boundary'].append(boundary_loss.item())
            losses['pde'].append(pde_loss.item())

            if self.verbose and len(losses['total']) % verbose_step == 0:
                print(f"L-BFGS - Iteration {len(losses['total']) - adam_epochs}, Total Loss: {total_loss.item():.6e}")

            return total_loss

        optimizer = torch.optim.LBFGS(self.net.parameters(),
                                      lr=2.0,
                                      max_iter=lbfgs_epochs,
                                      max_eval=lbfgs_epochs*2,
                                      tolerance_grad=1e-9,
                                      tolerance_change=1e-16,
                                      history_size=200,
                                      line_search_fn="strong_wolfe")
        optimizer.step(closure)

        if self.verbose:
            print(f"L-BFGS complete, Final Loss: {losses['total'][-1]:.6e}")
        self.losses = losses

        return

    # ANALYTICAL SOLUTION
    def exact_solution(self, x, y, t):
        """
        Compute the exact KP solution for n=1 or n=2 line solitons.
        Inputs x, y, t should be torch tensors.Computation is done in float64 and on the same device as x.

        Returns a tensor in the same dtype as x.
        """
        k = self.k
        P = self.P
        n = self.n

        # Promote to float64 (issues with exponentials in single precision) and ensure all tensors are on the same device
        x_d = x.to(torch.float64)
        y_d = y.to(torch.float64)
        t_d = t.to(torch.float64)

        if n == 1:
            k1 = k[0]
            P1 = P[0]
            eta = k1 * (x_d + P1 * y_d - (k1 ** 2 + 3 * P1 ** 2) * t_d)
            F = 1.0 + torch.exp(eta)
            Fx = k1 * torch.exp(eta)
            Fxx = k1 ** 2 * torch.exp(eta)
            u = 2.0 * (F * Fxx - Fx ** 2) / F ** 2
            return u.to(x.dtype)

        elif n == 2:
            k1, k2 = k
            P1, P2 = P
            eta1 = k1 * (x_d + P1 * y_d - (k1 ** 2 + 3 * P1 ** 2) * t_d)
            eta2 = k2 * (x_d + P2 * y_d - (k2 ** 2 + 3 * P2 ** 2) * t_d)
            A12 = ((k1 - k2) ** 2 - (P1 - P2) ** 2) / ((k1 + k2) ** 2 - (P1 - P2) ** 2)
            exp1 = torch.exp(eta1)
            exp2 = torch.exp(eta2)
            exp12 = torch.exp(eta1 + eta2)
            F = 1.0 + exp1 + exp2 + A12 * exp12
            Fx = k1 * exp1 + k2 * exp2 + A12 * (k1 + k2) * exp12
            Fxx = k1 ** 2 * exp1 + k2 ** 2 * exp2 + A12 * (k1 + k2) ** 2 * exp12
            u = 2.0 * (F * Fxx - Fx ** 2) / F ** 2
            return u.to(x.dtype)
        else:
            raise NotImplementedError("KP_solution for n > 2 is not implemented yet")

    # VISUALIZATION
    def plot_heatmap(self, t_value, nx=200, ny=200, which='exact', vmin=None, vmax=None):
        """
        Plot a heatmap of the solution at a given time t_value.
        which: 'exact' (default) or 'predicted'
        """
        x = torch.linspace(self.x_lims[0], self.x_lims[1], nx, device=self.device)
        y = torch.linspace(self.y_lims[0], self.y_lims[1], ny, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        t = torch.full_like(X, t_value)
        if which == 'exact':
            U = self.exact_solution(X, Y, t)
        elif which == 'predicted':
            with torch.no_grad():
                U = self.net(X.reshape(-1, 1), Y.reshape(-1, 1), t.reshape(-1, 1))
            U = U.reshape(X.shape)
        else:
            raise NotImplementedError("which must be 'exact' or 'predicted'")
        U_np = U.cpu().numpy()
        X_np = X.cpu().numpy()
        Y_np = Y.cpu().numpy()
        plt.figure(figsize=(7, 5))
        plt.pcolormesh(X_np, Y_np, U_np, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(label='u(x, y, t)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.show()

    def plot_losses(self, losses=None, component=None, show_optimizer_switch=True):
        """
        Plot the losses over epochs to visualize training progress.
        
        Parameters:
        -----------
        losses: dict or None
            Dictionary with loss components. If None, uses self.losses.
        component: str, list, or None
            If None, plots total loss (default behavior)
            If 'all', plots all loss components
            If a list of strings, plots all specified components
        show_optimizer_switch: bool
            Whether to show a vertical line indicating the switch from Adam to LBFGS optimizer
        """
        if losses is None:
            if not hasattr(self, 'losses'):
                raise ValueError("No loss history available. Train the model or provide a losses dict.")
            losses = self.losses
        
        figsize = getattr(self, 'figsize', (10, 6))
        plt.figure(figsize=figsize)
        
        # Convert single component to list for uniform processing
        if component is None:
            component = ['total']
        elif component == 'all':
            component = list(losses.keys())
        elif isinstance(component, str):
            component = [component]
        
        # Plot each requested component
        for comp in component:
            if comp in losses:
                plt.plot(losses[comp], label=f'{comp} loss')
            else:
                raise ValueError(f"Unknown loss component '{comp}'")
        
        # Add vertical line for optimizer switch if requested
        if show_optimizer_switch and hasattr(self, 'adam_epochs') and self.adam_epochs > 0:
            plt.axvline(x=self.adam_epochs, color='r', linestyle='--', alpha=0.7)
        
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.tight_layout()
        return

    # TESTING
    def predict(self, t_value=0.0, nx=200, ny=200):
        """
        Evaluate the trained PINN on a regular (x, y) grid at a given time t_value.
        Returns X, Y, U as numpy arrays.
        """
        x = torch.linspace(self.x_lims[0], self.x_lims[1], nx, device=self.device)
        y = torch.linspace(self.y_lims[0], self.y_lims[1], ny, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        t = torch.full_like(X, t_value)
        with torch.no_grad():
            U_pred = self.net(X.reshape(-1, 1), Y.reshape(-1, 1), t.reshape(-1, 1))
        U_pred = U_pred.reshape(X.shape)
        return X.cpu().numpy(), Y.cpu().numpy(), U_pred.cpu().numpy()

    def setup_testing_domain(self, nx=300, ny=300, nt=100):
        """
        Create a regular grid for testing and visualization.
        Sets up a uniform meshgrid covering the entire domain for evaluation and visualization of the PINN solution.
        Stores both meshgrids and flattened arrays as class attributes.
        """
        x0, x1 = self.x_lims
        y0, y1 = self.y_lims
        t0, t1 = self.t_lims
        device = self.device

        # 1D grids
        x = torch.linspace(x0, x1, nx, device=device)
        y = torch.linspace(y0, y1, ny, device=device)
        t = torch.linspace(t0, t1, nt, device=device)

        self.x_test = x.cpu().numpy()
        self.y_test = y.cpu().numpy()
        self.t_test = t.cpu().numpy()

        # Meshgrid (x, y, t)
        X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
        self.X_test = X
        self.Y_test = Y
        self.T_test = T

        # Flattened for batch evaluation
        self.X_flat_test = X.reshape(-1, 1)
        self.Y_flat_test = Y.reshape(-1, 1)
        self.T_flat_test = T.reshape(-1, 1)

        # Set flag indicating test domain has been created
        self.test_domain_created = True

    def compute_solutions(self, recompute: bool = False, to_numpy: bool = True):
        """
        Evaluate and cache both the PINN-predicted and exact solutions on the test grid. Creates/uses self.solutions (torch tensors) and self.solutions_np (NumPy arrays). 
        
        Returns a dictionary with keys 'predicted' and 'exact'.
        """
        # Skip if already computed and not recomputing
        if (not recompute) and hasattr(self, 'solutions'):
            return self.solutions

        # Ensure test domain is set up
        if not hasattr(self, 'X_flat_test'):
            raise RuntimeError("Testing domain not set up. Call setup_testing_domain() first.")

        # (1) PINN prediction
        with torch.no_grad():
            U_pred_flat = self.net(self.X_flat_test, self.Y_flat_test, self.T_flat_test)
            U_pred = U_pred_flat.reshape(self.X_test.shape)

        # (2) Exact analytical solution
        U_exact = self.exact_solution(self.X_test, self.Y_test, self.T_test)

        # store solutions as dictionary
        self.solutions = {
            'predicted': U_pred,
            'exact': U_exact
        }

        # torch versions (used for error computation in test())
        self.U_pred = U_pred
        self.U_exact = U_exact

        # numpy versions (used for plotting and saving)
        if to_numpy:
            self.solutions_np = {k: v.detach().cpu().numpy() for k, v in self.solutions.items()}
            self.U_pred_np = self.solutions_np['predicted']
            self.U_exact_np = self.solutions_np['exact']
            self.X_np = self.X_test.cpu().numpy()
            self.Y_np = self.Y_test.cpu().numpy()
            self.T_np = self.T_test.cpu().numpy()
        else:
            self.U_pred_np = None
            self.U_exact_np = None

        return

    # SAVING RESULTS
    def save_results(self, filename):
        """
        Save model results to a JSON file for use in Julia or other tools.
        Includes domain grids, predicted and exact solutions, and loss history.
        """
        # Ensure predictions exist
        self.compute_solutions()

        # Prepare the results dictionary
        results = {
            "domain": {
                "x": self.x_test.tolist(),
                "y": self.y_test.tolist(),
                "t": self.t_test.tolist()
            },
            "solution": {
                "u_pred": self.U_pred_np.tolist(),
                "u_exact": self.U_exact_np.tolist()
            },
            "losses": {}
        }

        # Add losses if available
        if hasattr(self, 'losses'):
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

        if self.verbose:
            print(f"Results saved to {filename}")

    def test(self, error_type: str = 'absolute-normalized', plot_heatmap: bool = False):
        """
        Compute error metrics between the predicted and analytical solutions over the test domain.
        Also plots the time-averaged error over (x, y) as a heatmap (log color scale).
        """
        # Ensure solutions are available
        self.compute_solutions()
        
        # Compute error (array of shape [nx, ny, nt])
        if error_type == 'absolute':
            self.error = torch.abs(self.U_pred - self.U_exact)
        elif error_type == 'absolute-normalized':
            max_exact = torch.max(torch.abs(self.U_exact))
            self.error = torch.abs(self.U_pred - self.U_exact) / max_exact
        else:
            raise ValueError(f"Invalid error type: {error_type}")
        
        # Compute mean error
        mae = torch.mean(self.error).item()
        self.mae = mae
        
        # Compute maximum error
        max_error = torch.max(self.error).item()
        self.max_error = max_error
        
        # print error summary
        if self.verbose:
            print(f"{error_type} error metrics:")
            print(f"Mean: {mae:.6e}")
            print(f"Maximum: {max_error:.6e}")
        
        # Plot heatmap of mean error in (x, y) with log color scale
        if plot_heatmap:
            
            # Compute mean error along the time axis (average over t for each (x, y))
            error_np = self.error.cpu().numpy()
            mean_error_xy = error_np.mean(axis=2)  # shape: (nx, ny)

            # Plot heatmap of mean error in (x, y) with log color scale
            plt.figure(figsize=(7, 5))
            plt.pcolormesh(self.X_np[:, :, 0], self.Y_np[:, :, 0], mean_error_xy, shading='auto', cmap='hot', norm=mcolors.LogNorm())
            plt.colorbar(label='Mean |u_pred - u_exact| (avg over t, log scale)')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.tight_layout()
            plt.show()
        
        return

    def save_experiment_run(self, root_path, to_save: tuple = ("loss", "error")):
        """
        Save selected results from the current model. Used only when certain results are needed, rather than everything from the full model, as opposed to save_results(). 

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