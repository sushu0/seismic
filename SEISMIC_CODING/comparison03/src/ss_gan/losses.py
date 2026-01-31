from __future__ import annotations
import torch

def gradient_penalty(critic, x: torch.Tensor, y_real: torch.Tensor, y_fake: torch.Tensor, lambda_gp: float = 10.0) -> torch.Tensor:
    B = x.size(0)
    c = torch.rand(B, 1, 1, device=x.device)
    # Paper formula uses \nabla_{\tilde{y}} D(x, \tilde{y}); keep x fixed.
    y_tilde = (c * y_real + (1.0 - c) * y_fake).requires_grad_(True)
    out = critic(torch.cat([x.detach(), y_tilde], dim=1))
    grad = torch.autograd.grad(
        outputs=out,
        inputs=y_tilde,
        grad_outputs=torch.ones_like(out),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(B, -1)
    gp = ((grad.norm(2, dim=1) - 1.0) ** 2).mean()
    return lambda_gp * gp
