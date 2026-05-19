

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
