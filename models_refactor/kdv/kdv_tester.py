import torch

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