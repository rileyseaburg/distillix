"""
FOGZO: First-Order-Guided Zeroth-Order Gradient Descent

Fixes STE (Straight-Through Estimator) gradient instability in BitNet training
by combining first-order gradients (biased but informative) with zeroth-order
gradients (unbiased but expensive) in a cost-effective manner.

Paper: "Improving the Straight-Through Estimator with Zeroth-Order Information"
       https://arxiv.org/abs/2510.23926

Key insight:
  - STE gradients are biased (sign(x) has zero gradient, STE pretends it's 1)
  - Pure zeroth-order (SPSA) is unbiased but requires many forward passes
  - FOGZO uses STE gradient direction to guide perturbation, reducing variance
  
Algorithm:
  1. Compute STE gradient g via backprop (biased first-order)
  2. Normalize: g_norm = g / ||g||
  3. Sample random noise z matching gradient estimator distribution
  4. Blend: u = sqrt(beta) * sign(s) * g_norm + sqrt(1-beta) * z
  5. Perturb weights: w+ = w + epsilon*u, w- = w - epsilon*u
  6. Compute finite difference: fd = (L(w+) - L(w-)) / (2*epsilon)
  7. Estimate gradient: g_zo = fd * u

where beta controls the blend between first-order guidance and randomness.
beta=1 is pure first-order (STE), beta=0 is pure zeroth-order (SPSA).

Copyright (c) 2025 Distillix. All Rights Reserved.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, List, Tuple, Callable
import math


class FOGZOEstimator:
    """
    First-Order-Guided Zeroth-Order Gradient Estimator.
    
    Reduces STE bias while avoiding the computational cost of pure zeroth-order methods.
    Uses n=1 perturbation (single forward-backward pair) for efficiency.
    
    Args:
        n: Number of perturbation samples (default: 1, as recommended in paper)
        beta_min: Minimum first-order guidance strength (default: 0.999)
               Higher = more STE-like, Lower = more SPSA-like
               Paper uses 1-1e-9 for very strong first-order guidance
        epsilon_scale: Perturbation magnitude scale (default: 1.0)
        estimator: Gradient estimator type ("ste", "tanh", "as")
                   Determines the noise distribution for perturbation
        device: Device for random number generator
    """
    
    def __init__(
        self,
        n: int = 1,
        beta_min: float = 0.999,
        epsilon_scale: float = 1.0,
        estimator: str = "ste",
        device: str = "cuda",
        seed: int = 42,
    ):
        self.n = n
        self.beta_min = beta_min
        self.epsilon_scale = epsilon_scale
        self.estimator = estimator
        self.device = device
        
        # Compute epsilon based on estimator (variance-matching)
        if estimator == "ste":
            # STE: uniform[-sqrt(3), sqrt(3)] has unit variance
            self.base_epsilon = 0.5 / (3 ** 0.5)
        elif estimator == "tanh":
            # Tanh estimator
            self.base_epsilon = math.pi / (12 ** 0.5)
        elif estimator == "as":
            # ApproxSign estimator  
            self.base_epsilon = 1 / (6 ** 0.5)
        else:
            raise ValueError(f"Unknown estimator: {estimator}")
        
        self.epsilon = self.base_epsilon * epsilon_scale
        
        # Random generator for reproducibility
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        
        # State tracking
        self.step = 0
        self.saved_gradients: Dict[str, Tensor] = {}
        self.perturbations: Dict[str, Tensor] = {}
        
    def compute_beta(self, current_step: int, total_steps: int) -> float:
        """
        Linear decay schedule for beta.
        
        Starts at beta=1.0 (pure first-order) and decays to beta_min.
        This allows early training to benefit from STE signal while
        later training gets unbiased gradients.
        
        Args:
            current_step: Current training step
            total_steps: Total training steps
            
        Returns:
            beta value for current step
        """
        progress = current_step / max(total_steps, 1)
        beta = (1 - progress) * (1 - self.beta_min) + self.beta_min
        return beta
    
    def sample_noise(self, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> Tensor:
        """
        Sample noise matching the gradient estimator's distribution.
        
        The noise distribution is chosen to match the STE/tanh/as estimator
        for variance-matching purposes.
        
        Args:
            shape: Shape of noise tensor
            device: Target device
            dtype: Target dtype
            
        Returns:
            Noise tensor with unit variance
        """
        if self.estimator == "ste":
            # STE: uniform[-sqrt(3), sqrt(3)] -> unit variance
            noise = (3 ** 0.5) * (2 * torch.rand(shape, generator=self.generator, device=device, dtype=dtype) - 1)
        elif self.estimator == "tanh":
            # Tanh: inverse CDF of sech^2 distribution
            cdf = torch.rand(shape, generator=self.generator, device=device, dtype=dtype)
            noise = torch.atanh(2 * cdf.clamp(1e-7, 1-1e-7) - 1) / (math.pi / (12 ** 0.5))
        elif self.estimator == "as":
            # ApproxSign: triangular-like distribution
            cdf = torch.rand(shape, generator=self.generator, device=device, dtype=dtype)
            noise = (6 ** 0.5) * torch.where(
                cdf <= 0.5,
                torch.sqrt(2 * cdf) - 1,
                1 - torch.sqrt(2 - 2 * cdf)
            )
        else:
            raise ValueError(f"Unknown estimator: {self.estimator}")
        
        return noise
    
    @torch.no_grad()
    def prepare_perturbation(
        self,
        model: nn.Module,
        param_names: List[str],
        beta: float,
    ) -> int:
        """
        Prepare first-order guided perturbation vectors.
        
        Uses the current STE gradients to guide the perturbation direction,
        blending with random noise based on beta.
        
        Args:
            model: Model with computed gradients
            param_names: Names of parameters to perturb (BitLinear weights)
            beta: First-order guidance strength (0-1)
            
        Returns:
            Random sign s (+1 or -1)
        """
        # Compute gradient norm across all target parameters
        grad_norm_squared = torch.zeros([], device=self.device, dtype=torch.float64)
        for name, param in model.named_parameters():
            if name not in param_names or param.grad is None:
                continue
            grad_norm_squared += param.grad.float().pow(2).sum()
        
        grad_norm = torch.sqrt(grad_norm_squared).item()
        if grad_norm < 1e-12:
            grad_norm = 1.0  # Avoid division by zero
        inv_grad_norm = 1.0 / grad_norm
        
        # Random sign for directional derivative
        s = 2 * torch.randint(0, 2, [], generator=self.generator, device=self.device) - 1
        s = s.item()
        
        # Compute blended perturbation for each parameter
        self.perturbations.clear()
        self.saved_gradients.clear()
        
        for name, param in model.named_parameters():
            if name not in param_names or param.grad is None:
                continue
            
            # Normalize gradient
            normalized_grad = inv_grad_norm * param.grad.float()
            
            # Save gradient for later (we'll clear it during perturbation)
            self.saved_gradients[name] = param.grad.clone()
            
            # Sample noise
            noise = self.sample_noise(param.shape, param.device, torch.float32)
            
            # Blend: u = sqrt(beta) * s * g_norm + sqrt(1-beta) * z
            u = (beta ** 0.5) * (s * normalized_grad) + ((1 - beta) ** 0.5) * noise
            self.perturbations[name] = u.to(param.dtype)
        
        return s
    
    @torch.no_grad()
    def perturb_positive(self, model: nn.Module, param_names: List[str]):
        """Apply positive perturbation: w+ = w + epsilon * u"""
        for name, param in model.named_parameters():
            if name in self.perturbations:
                param.data.add_(self.epsilon * self.perturbations[name])
    
    @torch.no_grad()
    def perturb_negative(self, model: nn.Module, param_names: List[str]):
        """Apply negative perturbation: w- = w - 2*epsilon * u (from w+)"""
        for name, param in model.named_parameters():
            if name in self.perturbations:
                param.data.add_(-2 * self.epsilon * self.perturbations[name])
    
    @torch.no_grad()
    def restore_weights(self, model: nn.Module, param_names: List[str]):
        """Restore original weights: w = w + epsilon * u (from w-)"""
        for name, param in model.named_parameters():
            if name in self.perturbations:
                param.data.add_(self.epsilon * self.perturbations[name])
    
    @torch.no_grad()
    def compute_zo_gradient(
        self,
        model: nn.Module,
        param_names: List[str],
        loss_positive: Tensor,
        loss_negative: Tensor,
    ):
        """
        Compute zeroth-order gradient estimate from finite differences.
        
        g_zo = (L(w+) - L(w-)) / (2 * epsilon) * u
        
        This replaces the biased STE gradient with an unbiased estimate
        guided by the first-order direction.
        
        Args:
            model: Model to update gradients on
            param_names: Names of parameters that were perturbed
            loss_positive: Loss at w + epsilon*u
            loss_negative: Loss at w - epsilon*u
        """
        # Finite difference
        fd = (loss_positive.item() - loss_negative.item()) / (2 * self.epsilon)
        
        # Set gradients
        for name, param in model.named_parameters():
            if name in self.perturbations:
                grad = fd * self.perturbations[name]
                param.grad = grad.to(param.dtype)
        
        self.step += 1
    
    def get_target_param_names(self, model: nn.Module) -> List[str]:
        """
        Get names of parameters that should use FOGZO gradient estimation.
        
        Targets BitLinear weights (the quantized layers) since these are
        where STE bias is most problematic.
        
        Args:
            model: Model to analyze
            
        Returns:
            List of parameter names to apply FOGZO to
        """
        target_names = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Target linear projection weights in transformer
            # These are the BitLinear layers where STE is used
            if any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                              'gate_proj', 'up_proj', 'down_proj']):
                if 'weight' in name:
                    target_names.append(name)
        return target_names


def fogzo_training_step(
    model: nn.Module,
    batch: Dict[str, Tensor],
    loss_fn: Callable,
    fogzo: FOGZOEstimator,
    param_names: List[str],
    current_step: int,
    total_steps: int,
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
) -> Tuple[float, float]:
    """
    Execute a single FOGZO training step.
    
    This replaces the standard forward-backward pass with the FOGZO procedure:
    1. Forward + backward with STE (get biased gradient)
    2. Prepare perturbation (blend STE direction with noise)
    3. Forward pass at w + epsilon*u (positive perturbation)
    4. Forward pass at w - epsilon*u (negative perturbation)
    5. Compute ZO gradient from finite difference
    6. Restore weights
    
    Args:
        model: Model to train
        batch: Input batch with input_ids, attention_mask, labels
        loss_fn: Loss function (model forward returns loss)
        fogzo: FOGZOEstimator instance
        param_names: Names of parameters to apply FOGZO to
        current_step: Current training step
        total_steps: Total training steps
        device: Device to run on
        use_amp: Whether to use automatic mixed precision
        amp_dtype: AMP dtype (float16 or bfloat16)
        
    Returns:
        Tuple of (loss value, beta value)
    """
    from contextlib import nullcontext
    
    model.train()
    
    # Compute beta (first-order guidance strength)
    beta = fogzo.compute_beta(current_step, total_steps)
    
    # Step 1: Forward + backward with STE
    amp_context = torch.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else nullcontext()
    
    with amp_context:
        outputs = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch.get('attention_mask', None),
            labels=batch['labels'].to(device) if 'labels' in batch else batch['input_ids'].to(device),
        )
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
    
    # Backward to get STE gradients
    loss.backward()
    original_loss = loss.item()
    
    # Step 2: Prepare perturbation using STE gradients
    s = fogzo.prepare_perturbation(model, param_names, beta)
    
    # Step 3: Forward at positive perturbation
    model.eval()  # No dropout during perturbation evals
    with torch.no_grad():
        fogzo.perturb_positive(model, param_names)
        
        with amp_context:
            outputs_pos = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch.get('attention_mask', None),
                labels=batch['labels'].to(device) if 'labels' in batch else batch['input_ids'].to(device),
            )
            loss_pos = outputs_pos['loss'] if isinstance(outputs_pos, dict) else outputs_pos.loss
    
    # Step 4: Forward at negative perturbation  
    with torch.no_grad():
        fogzo.perturb_negative(model, param_names)
        
        with amp_context:
            outputs_neg = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch.get('attention_mask', None),
                labels=batch['labels'].to(device) if 'labels' in batch else batch['input_ids'].to(device),
            )
            loss_neg = outputs_neg['loss'] if isinstance(outputs_neg, dict) else outputs_neg.loss
    
    # Step 5: Restore weights
    fogzo.restore_weights(model, param_names)
    
    # Step 6: Compute ZO gradient
    fogzo.compute_zo_gradient(model, param_names, loss_pos, loss_neg)
    
    model.train()
    
    return original_loss, beta


class FOGZOWrapper:
    """
    High-level wrapper for FOGZO training integration.
    
    Simplifies integration with existing training loops by providing
    a drop-in replacement for the standard backward pass.
    
    Args:
        model: Model to train
        n: Number of perturbation samples (default: 1)
        beta_min: Minimum first-order guidance strength (default: 0.999)
        epsilon_scale: Perturbation magnitude scale (default: 1.0)
        estimator: Gradient estimator type (default: "ste")
    """
    
    def __init__(
        self,
        model: nn.Module,
        n: int = 1,
        beta_min: float = 0.999,
        epsilon_scale: float = 1.0,
        estimator: str = "ste",
        device: str = "cuda",
    ):
        self.model = model
        self.estimator = FOGZOEstimator(
            n=n,
            beta_min=beta_min,
            epsilon_scale=epsilon_scale,
            estimator=estimator,
            device=device,
        )
        self.param_names = self.estimator.get_target_param_names(model)
        
        print(f"FOGZO targeting {len(self.param_names)} parameters")
        
    def step(
        self,
        batch: Dict[str, Tensor],
        current_step: int,
        total_steps: int,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
    ) -> Tuple[float, float]:
        """
        Execute FOGZO training step.
        
        Returns:
            Tuple of (loss, beta)
        """
        return fogzo_training_step(
            model=self.model,
            batch=batch,
            loss_fn=None,  # Uses model's built-in loss
            fogzo=self.estimator,
            param_names=self.param_names,
            current_step=current_step,
            total_steps=total_steps,
            device=next(self.model.parameters()).device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing FOGZO estimator...")
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    ).cuda()
    
    # Create estimator
    fogzo = FOGZOEstimator(n=1, beta_min=0.999, device="cuda")
    
    # Dummy forward-backward
    x = torch.randn(4, 64).cuda()
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    # Test perturbation
    param_names = ['0.weight', '2.weight']
    s = fogzo.prepare_perturbation(model, param_names, beta=0.999)
    print(f"Random sign s = {s}")
    print(f"Prepared {len(fogzo.perturbations)} perturbations")
    
    # Test perturb
    fogzo.perturb_positive(model, param_names)
    fogzo.perturb_negative(model, param_names)
    fogzo.restore_weights(model, param_names)
    
    print("FOGZO estimator test passed!")
