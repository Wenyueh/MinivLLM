import torch
import time 

class LayerNorm(torch.nn.Module):
    def __init__(self, gamma: torch.Tensor, eps: float = 1e-5):
        super().__init__()
        # Use nn.Parameter to make gamma learnable and loadable from checkpoints
        self.weight = torch.nn.Parameter(gamma)
        self.eps = eps

    @property
    def gamma(self):
        """Backward compatibility: gamma alias for weight"""
        return self.weight

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm(x) = (x / sqrt(mean(x²) + ε)) ⊙ γ

        variance = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        sqrt_variance = variance.sqrt()
        x_norm = (x / sqrt_variance * self.weight)

        return x_norm

    def residual_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x = x + residual
        return self.rms_forward(x), x

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if residual is not None:
            return self.residual_rms_forward(x, residual)
        else:
            return self.rms_forward(x)

if __name__ == "__main__":
    # Example usage
    x = torch.randn(8,4000,8000).cuda()
    gamma = torch.full((8000,), 0.5, device="cuda", dtype=x.dtype)
    layer = LayerNorm(gamma=gamma).cuda()
    residual = torch.full_like(x,fill_value=1)

    for _ in range(10): # Warm-up iterations
        _ = layer(x)
    
    # Without residuals
    times = [] 
    for _ in range(100): # Timing iterations
        torch.cuda.synchronize()
        start_time = time.time()
        _ = layer(x)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"[Without residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")

    # With residuals
    times.clear()
    for _ in range(100): # Timing iterations
        torch.cuda.synchronize()
        start_time = time.time()
        _ = layer(x,residual)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"[With residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
    
