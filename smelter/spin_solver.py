"""
BitNet Spin Solver - A "Quantum-Discrete" Optimizer for Ternary Networks.

The Problem:
  Standard optimizers (Adam, SGD) live in ℝⁿ (continuous Euclidean space).
  BitNet lives on {-1, 0, +1}ⁿ (corners of a hypercube).
  Adam wanders into the INTERIOR of the hypercube where quantized values are 0,
  and gets trapped. This causes "weight collapse" - the death of BitNet models.

The Physics:
  A BitNet is a Spin Glass - a lattice of magnetic spins trying to minimize energy.
  - Spins want to be ±1 (ferromagnetic), not 0 (paramagnetic)
  - Weight decay acts as an external field pulling all spins to 0
  - High temperature (noise) causes random flipping → average = 0
  
The Solution:
  1. Discrete Geometry: Move by SIGNS (directions on hypercube), not magnitudes
  2. Hysteresis: Spins have "memory" - must overcome threshold to flip
  3. Dead Zone Tunneling: "It is illegal to be zero" - force escape from center

Mathematical Framework:
  Instead of the convex L2 "bowl" that pulls weights to 0:
    E_L2 = λ||w||²  →  gradient always points to origin
    
  We use a "double-well" (sombrero) potential that REPELS from 0:
    E_well = λ(|w| - target)²  →  stable points at ±target
    
  This creates:
    - Repulsion when |w| < target (pushes away from 0)
    - Attraction when |w| > target (pulls toward ±target)

Copyright (c) 2025 Distillix. All Rights Reserved.
"""

import torch
from torch.optim import Optimizer
from typing import Optional, Callable, List, Tuple


class BitNetSpinSolver(Optimizer):
    """
    Hysteretic Sign Solver for BitNet ternary weights.
    
    Treats weights as spins on a hypercube. Uses gradient DIRECTION (not magnitude)
    to accumulate "pressure", then flips bits when threshold is crossed.
    
    Key features:
      1. Sign-based updates (discrete geometry)
      2. Latent accumulator with hysteresis (quantum memory)
      3. Dead-zone tunneling (anti-collapse)
    
    Args:
        params: Model parameters
        lr: Learning rate (controls pressure accumulation speed)
        hysteresis_threshold: Magnitude threshold for "dead zone" detection
        tunnel_strength: How hard to push weights out of dead zone
        use_momentum: Whether to use momentum on the sign gradients
        momentum: Momentum coefficient if use_momentum=True
        
    Example:
        >>> optimizer = BitNetSpinSolver(model.parameters(), lr=1e-3, hysteresis_threshold=0.05)
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step()
        ...     optimizer.zero_grad()
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        hysteresis_threshold: float = 0.05,
        tunnel_strength: float = 2.0,
        use_momentum: bool = True,
        momentum: float = 0.9,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if hysteresis_threshold < 0.0:
            raise ValueError(f"Invalid hysteresis threshold: {hysteresis_threshold}")
            
        defaults = dict(
            lr=lr,
            hysteresis_threshold=hysteresis_threshold,
            tunnel_strength=tunnel_strength,
            use_momentum=use_momentum,
            momentum=momentum,
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        """
        Perform a single optimization step.
        
        The algorithm:
          1. Extract gradient SIGN (discrete direction on hypercube)
          2. Update latent weights using sign-based momentum
          3. Detect "dead zone" weights (too close to 0)
          4. Apply tunneling force to escape dead zone
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            thresh = group['hysteresis_threshold']
            tunnel = group['tunnel_strength']
            use_mom = group['use_momentum']
            mom = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    if use_mom:
                        state['momentum_buffer'] = torch.zeros_like(p)
                
                # === STEP 1: Discrete Geometry ===
                # Only care about gradient DIRECTION, not magnitude
                # This prevents spiky gradients from destabilizing training
                grad_sign = grad.sign()
                
                # === STEP 2: Momentum on Signs ===
                if use_mom:
                    buf = state['momentum_buffer']
                    buf.mul_(mom).add_(grad_sign, alpha=1 - mom)
                    # Use momentum buffer for update direction
                    update_direction = buf.sign()
                else:
                    update_direction = grad_sign
                
                # === STEP 3: Update Latent Weights ===
                # Negative because we minimize loss
                p.add_(update_direction, alpha=-lr)
                
                # === STEP 4: Dead Zone Detection & Tunneling ===
                # If weight is in the "danger zone" close to 0, push it out
                dead_zone = p.abs() < thresh
                
                if dead_zone.any():
                    # Determine escape direction:
                    # - If gradient is pushing positive, escape to +thresh
                    # - If gradient is pushing negative, escape to -thresh
                    # - If gradient is 0, use random thermal jitter
                    
                    escape_direction = -update_direction[dead_zone]
                    
                    # Handle zero gradients with random spin (thermal noise)
                    zero_grad_mask = escape_direction == 0
                    if zero_grad_mask.any():
                        random_spin = torch.randint_like(
                            escape_direction[zero_grad_mask], 0, 2
                        ) * 2 - 1
                        escape_direction[zero_grad_mask] = random_spin.float()
                    
                    # Apply tunneling force
                    # tunnel_strength > 1 means we push harder than normal updates
                    p[dead_zone] += escape_direction * (lr * tunnel)
        
        return loss


class PolarizedOptimizer:
    """
    Wrapper that adds "polarization" post-step to any optimizer.
    
    This is a simpler approach than BitNetSpinSolver: use your favorite optimizer
    (Adam, Muon, etc.) but add a post-step hook that prevents weight collapse.
    
    The "Double-Well Potential":
      Standard L2 regularization creates a "bowl" pulling weights to 0.
      We replace it with a "W-shaped" potential that REPELS from 0.
      
      E_new = λ(|w| - target)²
      
      This creates stable equilibria at w = ±target, not w = 0.
    
    Args:
        optimizer: Base optimizer (Adam, SGD, Muon, etc.)
        target_scale: Target weight magnitude (from healthy checkpoints: ~0.005-0.01)
        polarization_strength: How hard to push weights toward ±target
        apply_to_patterns: List of parameter name patterns to apply polarization to
        
    Example:
        >>> base_opt = AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
        >>> optimizer = PolarizedOptimizer(base_opt, target_scale=0.01)
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step()  # Calls base optimizer, then applies polarization
        ...     optimizer.zero_grad()
    """
    
    # Default patterns for BitNet MLP layers (these are what collapse)
    DEFAULT_PATTERNS = ['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    def __init__(
        self,
        optimizer: Optimizer,
        model: torch.nn.Module,
        target_scale: float = 0.01,
        polarization_strength: float = 0.1,
        apply_to_patterns: Optional[List[str]] = None,
    ):
        self.optimizer = optimizer
        self.model = model
        self.target_scale = target_scale
        self.polarization_strength = polarization_strength
        self.patterns = apply_to_patterns or self.DEFAULT_PATTERNS
        
        # Cache parameter names for fast lookup
        self._polarized_params: List[Tuple[str, torch.nn.Parameter]] = []
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in self.patterns):
                self._polarized_params.append((name, param))
    
    def step(self, closure: Optional[Callable] = None):
        """Execute optimizer step, then apply polarization."""
        # Call base optimizer
        loss = self.optimizer.step(closure)
        
        # Apply polarization post-step
        self._apply_polarization()
        
        return loss
    
    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients."""
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    @torch.no_grad()
    def _apply_polarization(self):
        """
        Apply double-well potential force to prevent weight collapse.
        
        The force:
          F = -∂E/∂w = -2λ(|w| - target) * sign(w)
          
        When |w| < target: Force pushes AWAY from 0 (resurrection)
        When |w| > target: Force pushes TOWARD ±target (regularization)
        """
        target = self.target_scale
        strength = self.polarization_strength
        
        for name, param in self._polarized_params:
            w = param.data
            
            # Compute distance from target magnitude
            # Positive when |w| > target, negative when |w| < target
            magnitude_error = w.abs() - target
            
            # Double-well force: push toward ±target
            # F = -sign(magnitude_error) * sign(w) * strength
            # Simplifies to: -magnitude_error.sign() * w.sign() * strength
            
            # But we want different behavior:
            # - If below target: PUSH AWAY from 0 (add sign(w) * strength)
            # - If above target: PULL TOWARD target (subtract excess * sign(w))
            
            # Dead zone: |w| < target * 0.5 (definitely collapsing)
            # Danger zone: target * 0.5 < |w| < target (needs help)
            # Safe zone: |w| >= target (healthy)
            
            dead_zone = w.abs() < (target * 0.5)
            danger_zone = (w.abs() >= target * 0.5) & (w.abs() < target)
            
            # In dead zone: strong push away from 0
            if dead_zone.any():
                # Use sign of weight, but if weight is exactly 0, use random
                push_direction = w[dead_zone].sign()
                zero_mask = push_direction == 0
                if zero_mask.any():
                    push_direction[zero_mask] = (
                        torch.randint_like(push_direction[zero_mask], 0, 2) * 2 - 1
                    ).float()
                
                # Strong resurrection force
                param.data[dead_zone] += push_direction * strength * 2.0
            
            # In danger zone: moderate push away from 0
            if danger_zone.any():
                push_direction = w[danger_zone].sign()
                # How far below target?
                deficit = target - w[danger_zone].abs()
                # Push proportional to deficit
                param.data[danger_zone] += push_direction * deficit * strength
    
    def state_dict(self):
        """Return optimizer state dict."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict)
    
    @property
    def param_groups(self):
        """Access base optimizer param groups."""
        return self.optimizer.param_groups


# =============================================================================
# Generic polarization hook (optimizer-agnostic)
# =============================================================================

def create_polarization_hook(
    model: torch.nn.Module,
    target_scale: float = 0.01,
    polarization_strength: float = 0.1,
    apply_to_patterns: Optional[List[str]] = None,
):
    """Create a callable that applies BitNet anti-collapse polarization.

    Why:
      `PolarizedOptimizer` wraps a torch `Optimizer`, but some training loops in
      this repo use hybrid optimizers (e.g., `MuonAdamW`) that are not subclasses
      of `torch.optim.Optimizer`.

    This hook lets you do:
      optimizer.step()
      polarization_hook()  # post-step anti-collapse

    Returns:
      A zero-arg callable.
    """
    patterns = apply_to_patterns or PolarizedOptimizer.DEFAULT_PATTERNS

    polarized_params: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(p in name for p in patterns):
            polarized_params.append(param)

    @torch.no_grad()
    def _hook():
        target = float(target_scale)
        strength = float(polarization_strength)

        for param in polarized_params:
            w = param.data
            dead_zone = w.abs() < (target * 0.5)
            danger_zone = (w.abs() >= target * 0.5) & (w.abs() < target)

            if dead_zone.any():
                push_direction = w[dead_zone].sign()
                zero_mask = push_direction == 0
                if zero_mask.any():
                    push_direction[zero_mask] = (
                        torch.randint_like(push_direction[zero_mask], 0, 2) * 2 - 1
                    ).float()
                param.data[dead_zone] += push_direction * strength * 2.0

            if danger_zone.any():
                push_direction = w[danger_zone].sign()
                deficit = target - w[danger_zone].abs()
                param.data[danger_zone] += push_direction * deficit * strength

    return _hook


def create_polarized_adamw(
    model: torch.nn.Module,
    lr: float = 3e-4,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    target_scale: float = 0.01,
    polarization_strength: float = 0.1,
) -> PolarizedOptimizer:
    """
    Convenience function to create AdamW with polarization.
    
    IMPORTANT: weight_decay is set to 0.0 because:
      1. Weight decay is what causes the collapse in the first place
      2. The polarization term provides implicit regularization
    """
    base_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=0.0,  # CRITICAL: No weight decay!
    )
    
    return PolarizedOptimizer(
        optimizer=base_optimizer,
        model=model,
        target_scale=target_scale,
        polarization_strength=polarization_strength,
    )


# =============================================================================
# Diagnostic Tools
# =============================================================================

def diagnose_weight_health(model: torch.nn.Module, threshold: float = 0.001) -> dict:
    """
    Diagnose weight health across the model.
    
    Returns dict with:
      - healthy_layers: List of layers with Std > threshold
      - collapsed_layers: List of layers with Std < threshold
      - stats: Dict of layer_name -> (mean, std, abs_mean)
    """
    healthy = []
    collapsed = []
    stats = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            std = param.std().item()
            mean = param.mean().item()
            abs_mean = param.abs().mean().item()
            
            stats[name] = {
                'mean': mean,
                'std': std,
                'abs_mean': abs_mean,
                'shape': list(param.shape),
            }
            
            if std > threshold:
                healthy.append(name)
            else:
                collapsed.append(name)
    
    return {
        'healthy_layers': healthy,
        'collapsed_layers': collapsed,
        'stats': stats,
        'health_ratio': len(healthy) / max(len(healthy) + len(collapsed), 1),
    }


def print_weight_health(model: torch.nn.Module, threshold: float = 0.001):
    """Print formatted weight health report."""
    report = diagnose_weight_health(model, threshold)
    
    print("=" * 60)
    print("WEIGHT HEALTH REPORT")
    print("=" * 60)
    print(f"Health ratio: {report['health_ratio']:.1%}")
    print(f"Healthy layers: {len(report['healthy_layers'])}")
    print(f"Collapsed layers: {len(report['collapsed_layers'])}")
    print()
    
    if report['collapsed_layers']:
        print("COLLAPSED (Std < {:.0e}):".format(threshold))
        for name in report['collapsed_layers'][:10]:  # Show first 10
            s = report['stats'][name]
            print(f"  {name}: Std={s['std']:.2e}, AbsMean={s['abs_mean']:.2e}")
        if len(report['collapsed_layers']) > 10:
            print(f"  ... and {len(report['collapsed_layers']) - 10} more")
        print()
    
    if report['healthy_layers']:
        print("HEALTHY (Std > {:.0e}):".format(threshold))
        for name in report['healthy_layers'][:5]:  # Show first 5
            s = report['stats'][name]
            print(f"  {name}: Std={s['std']:.4f}, AbsMean={s['abs_mean']:.4f}")
        if len(report['healthy_layers']) > 5:
            print(f"  ... and {len(report['healthy_layers']) - 5} more")
    
    print("=" * 60)
