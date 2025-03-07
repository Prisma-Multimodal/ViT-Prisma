from types import SimpleNamespace
from typing import Optional

import torch
import tqdm

def weighted_average(points, weights):
    """
    Returns the weighted average of the points.
    
    Args:
        points: Tensor of shape (n, d) where n is the number of points and d is the dimension
        weights: Tensor of shape (n,) containing the weights for each point
    """
    # Log shapes for debugging
    print(f"In weighted_average - Points shape: {points.shape}, Weights shape: {weights.shape}")
    
    # Ensure weights is a 1D tensor
    if weights.dim() > 1:
        weights = weights.reshape(-1)
    
    # Make sure weights has the right length
    if len(weights) != points.shape[0]:
        raise ValueError(
            f"Number of weights ({len(weights)}) must match number of points ({points.shape[0]})"
        )
    
    # Safe broadcasting by explicitly reshaping weights
    # From (n,) to (n, 1) to broadcast with (n, d)
    weights_reshaped = weights.reshape(-1, 1)
    
    # Compute weighted sum: (n, d) * (n, 1) -> (n, d), then sum over first dimension
    weighted_sum = (points * weights_reshaped).sum(dim=0)
    
    # Return weighted average
    return weighted_sum / weights.sum()

@torch.no_grad()
def geometric_median_objective(
    median: torch.Tensor, points: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:

    norms = torch.linalg.norm(points - median.view(1, -1), dim=1)  # type: ignore

    return (norms * weights).sum()

def compute_geometric_median(points, weights=None, eps=1e-6, maxiter=100, ftol=1e-20, do_log=False):
    """
    Compute the geometric median using the Weiszfeld algorithm.
    
    Args:
        points: Tensor of shape (n, ...) where n is the number of points
        weights: Optional tensor of shape (n,) containing the weights
        eps: Smallest allowed value for denominator
        maxiter: Maximum number of iterations
        ftol: Convergence tolerance
        do_log: Whether to record objective values
    """
    from types import SimpleNamespace
    import tqdm
    
    # Print shapes for debugging
    print(f"In compute_geometric_median - Points shape: {points.shape}")
    
    # Handle multi-dimensional points (beyond 2D)
    original_shape = None
    if points.dim() > 2:
        original_shape = points.shape[1:]  # Save dimensions beyond the first
        # Reshape to (n_samples, n_features)
        points = points.reshape(points.shape[0], -1)
        print(f"Reshaped points to: {points.shape}")
    
    # Now points should be 2D with shape (n_samples, n_features)
    n_points, n_features = points.shape
    print(f"Number of points: {n_points}, Number of features: {n_features}")
    
    with torch.no_grad():
        # Initialize weights if not provided
        if weights is None:
            weights = torch.ones(n_points, device=points.device)
        else:
            # Ensure weights has correct shape
            weights = weights.reshape(-1)
            if len(weights) != n_points:
                raise ValueError(f"Expected {n_points} weights, got {len(weights)}")
        
        # Start with the mean as the initial guess for the median
        try:
            median = (points * weights.reshape(-1, 1)).sum(dim=0) / weights.sum()
            print(f"Initial median shape: {median.shape}")
        except Exception as e:
            print(f"Error in initial median calculation: {str(e)}")
            raise
        
        # Function to compute objective value
        def objective_func(m):
            # Compute distances from each point to m
            diffs = points - m.unsqueeze(0)  # Shape: (n, d)
            norms = torch.linalg.norm(diffs, dim=1)  # Shape: (n,)
            return (weights * norms).sum()
        
        objective_value = objective_func(median)
        
        if do_log:
            logs = [objective_value]
        else:
            logs = None
        
        # Weiszfeld iterations
        early_termination = False
        pbar = tqdm.tqdm(range(maxiter))
        
        for _ in pbar:
            prev_obj_value = objective_value
            
            # Compute distances from each point to current median estimate
            diffs = points - median.unsqueeze(0)  # Shape: (n, d)
            norms = torch.linalg.norm(diffs, dim=1)  # Shape: (n,)
            
            # Compute new weights
            new_weights = weights / torch.clamp(norms, min=eps)
            
            # Compute new median estimate
            median = (points * new_weights.reshape(-1, 1)).sum(dim=0) / new_weights.sum()
            
            # Compute new objective value
            objective_value = objective_func(median)
            
            if logs is not None:
                logs.append(objective_value)
            
            # Check for convergence
            if abs(prev_obj_value - objective_value) <= ftol * objective_value:
                early_termination = True
                break
            
            pbar.set_description(f"Objective value: {objective_value:.4f}")
    
    # Final computation with gradient tracking enabled
    diffs = points - median.unsqueeze(0)
    norms = torch.linalg.norm(diffs, dim=1)
    new_weights = weights / torch.clamp(norms, min=eps)
    median = (points * new_weights.reshape(-1, 1)).sum(dim=0) / new_weights.sum()
    
    # Reshape median back to original dimensions if needed
    if original_shape is not None:
        median = median.reshape(original_shape)
        print(f"Reshaped median back to: {median.shape}")
    
    return SimpleNamespace(
        median=median,
        new_weights=new_weights,
        termination=(
            "function value converged within tolerance"
            if early_termination
            else "maximum iterations reached"
        ),
        logs=logs,
    )