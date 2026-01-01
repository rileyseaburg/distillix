"""
Muon Optimizer - Momentum Orthogonalized Update.

Designed for training models with sharp gradients (like BitNet 1.58-bit).
The orthogonalization prevents gradient explosions that cause NaN in mixed precision.

Reference: https://github.com/KellerJordan/modded-nanogpt
"""
import torch
from torch.optim import Optimizer


def newton_schulz_orthogonalize(M, num_iters=5, eps=1e-7):
    """
    Newton-Schulz iteration to orthogonalize matrix M.
    
    This is the key innovation of Muon - it normalizes the update direction
    via orthogonalization, preventing gradient explosions.
    """
    norm = torch.norm(M, p='fro')
    if norm < eps:
        return M
    X = M / (norm + eps)
    for _ in range(num_iters):
        A = X.T @ X
        X = X @ (1.5 * torch.eye(A.shape[0], device=A.device, dtype=A.dtype) - 0.5 * A)
    return X * norm


class Muon(Optimizer):
    """
    Muon (Momentum Orthogonalized Update) optimizer.
    
    Applies Newton-Schulz orthogonalization to momentum buffer before update,
    which normalizes update directions and prevents gradient explosions.
    
    Args:
        params: Iterable of parameters to optimize (should be 2D matrices)
        lr: Learning rate (default: 0.02)
        momentum: Momentum factor (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_iters: Newton-Schulz iterations (default: 5)
        weight_decay: Weight decay (default: 0.0)
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_iters=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_iters=ns_iters, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            lr = group['lr']
            mom = group['momentum']
            nesterov = group['nesterov']
            ns_iters = group['ns_iters']
            wd = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                
                # Apply weight decay (decoupled, like AdamW)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                
                # Initialize momentum buffer
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    
                buf = state['momentum_buffer']
                buf.mul_(mom).add_(grad)
                
                # Orthogonalize momentum buffer
                buf_ortho = newton_schulz_orthogonalize(buf, num_iters=ns_iters)
                
                if nesterov:
                    update = grad + mom * buf_ortho
                else:
                    update = buf_ortho
                
                # Orthogonalize final update
                update = newton_schulz_orthogonalize(update, num_iters=ns_iters)
                
                p.add_(update, alpha=-lr)
                
        return loss


class MuonAdamW:
    """
    Hybrid optimizer combining Muon and AdamW.
    
    - Muon: For 2D internal weight matrices (Linear/BitLinear layers)
    - AdamW: For Embeddings, LM Head, LayerNorms, and biases
    
    This is the recommended setup because:
    1. Muon only works on 2D matrices (needs orthogonalization)
    2. Embeddings should use AdamW (standard practice, more stable)
    3. Norms and biases are 1D, need AdamW
    
    Args:
        named_params: Iterator of (name, param) tuples
        muon_lr: Learning rate for Muon (default: 0.02)
        adamw_lr: Learning rate for AdamW (default: 3e-4)
        weight_decay: Weight decay for both optimizers (default: 0.01)
    """
    def __init__(self, named_params, muon_lr=0.02, adamw_lr=3e-4, weight_decay=0.01):
        muon_params = []
        adamw_params = []
        
        for name, p in named_params:
            if not p.requires_grad:
                continue
            
            # Embeddings and LM Head should use AdamW (standard practice)
            is_embedding = 'embed' in name.lower() or 'lm_head' in name.lower()
            
            # Muon only supports 2D matrices (and not embeddings)
            if p.ndim == 2 and not is_embedding:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        
        self.optims = []
        
        if muon_params:
            self.optims.append(Muon(muon_params, lr=muon_lr, weight_decay=weight_decay))
        if adamw_params:
            self.optims.append(torch.optim.AdamW(adamw_params, lr=adamw_lr, weight_decay=weight_decay))
        
        self.num_matrix = sum(p.numel() for p in muon_params)
        self.num_vector = sum(p.numel() for p in adamw_params)
        
    def step(self):
        for opt in self.optims:
            opt.step()
    
    def zero_grad(self, set_to_none=True):
        for opt in self.optims:
            opt.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self):
        return [opt.state_dict() for opt in self.optims]
    
    def load_state_dict(self, state_dicts):
        for opt, state in zip(self.optims, state_dicts):
            opt.load_state_dict(state)
