"""
This file contains the functions used in training the given neural network
"""

import torch

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