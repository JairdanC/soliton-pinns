import torch 

def log_gpu_memory(tag: str = '', device: torch.DeviceLikeType = None
                   ) -> None:
        
    if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print(f"[gpu mem] {tag:<25} alloc {alloc:7.1f} MB  reserved {reserved:7.1f} MB  peak {peak:7.1f} MB")
    else:
        print(f"[gpu mem] {tag:<25} cuda not available")

    return

def print_weighted_loss_components(loss_weights: torch.Tensor, loss_comps: torch.Tensor, tag: str = ''
                                   ) -> None:
    
    weighted_comps = loss_weights * loss_comps
    print(f'Weighted losses [{tag}]: IC={weighted_comps[0]:.3e} | BC={weighted_comps[1]:.3e} | PDE={weighted_comps[2]:.3e}')
    return