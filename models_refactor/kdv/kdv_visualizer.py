import matplotlib.pyplot as plt

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