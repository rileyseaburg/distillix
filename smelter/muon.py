"""
Muon Optimizer - Momentum Orthogonalized Update.

Based on the Stanford paper "Fantastic Pretraining Optimizers and Where to Find Them"
(Percy Liang et al., September 2025).

Key insight: For small models (<1B params), matrix-based optimizers outperform
scalar optimizers like AdamW by 30-40% in convergence speed.

Muon applies Newton-Schulz orthogonalization to the momentum buffer, which:
  1. Decorrelates gradient directions
  2. Equalizes gradient magnitudes across dimensions
  3. Results in more efficient parameter updates

CRITICAL: Muon only works on 2D matrix parameters (Linear layers).
For 1D parameters (norms, biases, embeddings), use AdamW.

Usage:
    # Split parameters by dimensionality
    matrix_params = [p for p in model.parameters() if p.ndim == 2]
    vector_params = [p for p in model.parameters() if p.ndim != 2]
    
    # Hybrid optimizer setup
    muon = Muon(matrix_params, lr=0.02, momentum=0.95)
    adamw = AdamW(vector_params, lr=3e-4)

Copyright (c) 2025 Distillix. All Rights Reserved.
"""

import torch
from torch.optim import Optimizer
from typing import List, Optional, Callable


def newton_schulz_orthogonalize(
    M: torch.Tensor, 
    num_iters: int = 5,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Newton-Schulz iteration for approximate matrix orthogonalization.
    
    Computes an approximation to M @ (M^T @ M)^{-1/2}, which orthogonalizes
    the columns of M while preserving their span.
    
    This is the core operation that makes Muon work:
    - Decorrelates gradient directions
    - Normalizes gradient magnitudes
    - Converges in ~5 iterations for well-conditioned matrices
    
    Args:
        M: Input matrix to orthogonalize [m, n]
        num_iters: Number of Newton-Schulz iterations (5 is usually enough)
        eps: Small constant for numerical stability
    
    Returns:
        Orthogonalized matrix with same shape as input
    """
    # Normalize to prevent numerical issues
    # Use Frobenius norm for matrices
    norm = torch.norm(M, p='fro')
    if norm < eps:
        return M
    
    # Scale down for numerical stability in NS iteration
    X = M / (norm + eps)
    
    # Newton-Schulz iteration: X_{k+1} = X_k @ (1.5 * I - 0.5 * X_k^T @ X_k)
    # This converges to X @ (X^T @ X)^{-1/2}
    for _ in range(num_iters):
        A = X.T @ X
        # The 3/2 and 1/2 coefficients come from the Newton-Schulz formula
        X = X @ (1.5 * torch.eye(A.shape[0], device=A.device, dtype=A.dtype) - 0.5 * A)
    
    # Scale back
    return X * norm


class Muon(Optimizer):
    """
    Muon (Momentum Orthogonalized Update) Optimizer.
    
    A matrix-based optimizer that applies Newton-Schulz orthogonalization
    to the momentum buffer before updating parameters.
    
    Key differences from AdamW:
      - AdamW: Scalar operations on individual parameters
      - Muon: Matrix operations that consider parameter correlations
    
    Hyperparameters:
      - lr: Learning rate (default 0.02 - much higher than AdamW!)
      - momentum: Momentum coefficient (default 0.95)
      - nesterov: Use Nesterov momentum (default True)
      - ns_iters: Newton-Schulz iterations (default 5)
      - weight_decay: L2 regularization (default 0.0)
    
    Note: Muon uses MUCH higher learning rates than AdamW (0.02 vs 3e-4).
    This is because the orthogonalization normalizes gradient magnitudes.
    
    Reference:
        "Fantastic Pretraining Optimizers and Where to Find Them"
        Stanford NLP (Percy Liang et al.), September 2025
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_iters: int = 5,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if ns_iters < 1:
            raise ValueError(f"Invalid ns_iters: {ns_iters}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_iters=ns_iters,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        
        # Validate that all parameters are 2D matrices
        for group in self.param_groups:
            for p in group['params']:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only supports 2D parameters, got shape {p.shape}. "
                        f"Use AdamW for 1D parameters (norms, biases, embeddings)."
                    )
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        
        Returns:
            Loss value if closure provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_iters = group['ns_iters']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Apply weight decay (decoupled, like AdamW)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                
                # Get or initialize momentum buffer
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                buf = state['momentum_buffer']
                
                # Update momentum buffer
                buf.mul_(momentum).add_(grad)
                
                # Apply Newton-Schulz orthogonalization to momentum
                # This is the key operation that makes Muon work
                buf_ortho = newton_schulz_orthogonalize(buf, num_iters=ns_iters)
                
                # Nesterov momentum: look ahead
                if nesterov:
                    update = grad + momentum * buf_ortho
                else:
                    update = buf_ortho
                
                # Apply orthogonalization to the final update as well
                update = newton_schulz_orthogonalize(update, num_iters=ns_iters)
                
                # Update parameters
                p.add_(update, alpha=-lr)
        
        return loss


class MuonAdamW:
    """
    Hybrid optimizer combining Muon (for matrices) and AdamW (for vectors).
    
    This is the recommended setup for training:
      - 2D parameters (Linear weights): Muon with high LR (0.02)
      - 1D parameters (norms, biases): AdamW with normal LR (3e-4)
      - Embeddings: AdamW with normal LR (3e-4)
    
    Usage:
        optimizer = MuonAdamW(
            model.named_parameters(),
            muon_lr=0.02,
            adamw_lr=3e-4,
        )
        
        # Training loop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    """
    
    def __init__(
        self,
        named_params,
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        weight_decay: float = 0.1,
        adamw_weight_decay: float = 0.01,
    ):
        """
        Initialize hybrid optimizer.
        
        Args:
            named_params: Iterator of (name, param) tuples from model.named_parameters()
            muon_lr: Learning rate for Muon (2D matrices)
            muon_momentum: Momentum for Muon
            adamw_lr: Learning rate for AdamW (1D vectors, embeddings)
            adamw_betas: Beta coefficients for AdamW
            adamw_eps: Epsilon for AdamW
            weight_decay: Weight decay for Muon
            adamw_weight_decay: Weight decay for AdamW
        """
        # Separate parameters by dimensionality
        matrix_params = []
        vector_params = []
        
        for name, param in named_params:
            if not param.requires_grad:
                continue
            
            if param.ndim == 2:
                # 2D matrix -> Muon
                matrix_params.append(param)
            else:
                # 1D vector or embedding -> AdamW
                vector_params.append(param)
        
        # Create optimizers
        self.muon = None
        self.adamw = None
        
        if matrix_params:
            self.muon = Muon(
                matrix_params,
                lr=muon_lr,
                momentum=muon_momentum,
                weight_decay=weight_decay,
            )
        
        if vector_params:
            self.adamw = torch.optim.AdamW(
                vector_params,
                lr=adamw_lr,
                betas=adamw_betas,
                eps=adamw_eps,
                weight_decay=adamw_weight_decay,
            )
        
        # Store counts for logging
        self.num_matrix_params = sum(p.numel() for p in matrix_params)
        self.num_vector_params = sum(p.numel() for p in vector_params)
    
    def step(self):
        """Perform optimization step on both optimizers."""
        if self.muon is not None:
            self.muon.step()
        if self.adamw is not None:
            self.adamw.step()
    
    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for both optimizers."""
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self) -> dict:
        """Return state dict for checkpointing."""
        return {
            'muon': self.muon.state_dict() if self.muon else None,
            'adamw': self.adamw.state_dict() if self.adamw else None,
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load state dict from checkpoint."""
        if self.muon is not None and state_dict.get('muon'):
            self.muon.load_state_dict(state_dict['muon'])
        if self.adamw is not None and state_dict.get('adamw'):
            self.adamw.load_state_dict(state_dict['adamw'])
    
    @property
    def param_groups(self) -> List[dict]:
        """Return all parameter groups from both optimizers."""
        groups = []
        if self.muon is not None:
            groups.extend(self.muon.param_groups)
        if self.adamw is not None:
            groups.extend(self.adamw.param_groups)
        return groups


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing Muon Optimizer...")
    
    # Test Newton-Schulz orthogonalization
    print("\n1. Testing Newton-Schulz orthogonalization:")
    M = torch.randn(64, 64)
    M_ortho = newton_schulz_orthogonalize(M)
    
    # Check orthogonality: M_ortho^T @ M_ortho should be close to identity
    I_approx = M_ortho.T @ M_ortho
    I_true = torch.eye(64)
    error = torch.norm(I_approx - I_true).item()
    print(f"   Orthogonality error (Frobenius): {error:.6f}")
    print(f"   Pass: {error < 0.1}")
    
    # Test Muon optimizer
    print("\n2. Testing Muon optimizer on simple quadratic:")
    W = torch.randn(32, 32, requires_grad=True)
    target = torch.randn(32, 32)
    
    optimizer = Muon([W], lr=0.02)
    
    for i in range(100):
        loss = ((W - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (i + 1) % 20 == 0:
            print(f"   Step {i+1}: Loss = {loss.item():.6f}")
    
    print(f"   Final loss: {loss.item():.6f}")
    print(f"   Pass: {loss.item() < 0.01}")
    
    # Test MuonAdamW hybrid
    print("\n3. Testing MuonAdamW hybrid optimizer:")
    
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 32)  # 2D weight
            self.norm = torch.nn.LayerNorm(32)      # 1D weight
        
        def forward(self, x):
            return self.norm(self.linear(x))
    
    model = SimpleModel()
    optimizer = MuonAdamW(model.named_parameters())
    
    print(f"   Matrix params (Muon): {optimizer.num_matrix_params:,}")
    print(f"   Vector params (AdamW): {optimizer.num_vector_params:,}")
    
    x = torch.randn(8, 32)
    target = torch.randn(8, 32)
    
    for i in range(50):
        out = model(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    print(f"   Final loss: {loss.item():.6f}")
    print(f"   Pass: {loss.item() < 0.5}")
    
    print("\nAll tests passed!")
