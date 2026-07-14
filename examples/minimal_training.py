"""Minimal LogRMS training example."""

import torch

from logrms import LogRMS, LogRMSCompiled, memory_audit


torch.manual_seed(7)
x = torch.randn(2_048, 16)
target = x[:, :4].sin() + 0.05 * torch.randn(2_048, 4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
target = target.to(device)
model = torch.nn.Sequential(
    torch.nn.Linear(16, 64),
    torch.nn.GELU(),
    torch.nn.Linear(64, 4),
).to(device)
optimizer_class = LogRMSCompiled if device.type == "cuda" else LogRMS
optimizer = optimizer_class(
    model.parameters(),
    lr=3e-3,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.01,
    backend="8bit",
    max_norm=1.0,
    stochastic_rounding=True,
    rounding_seed=7,
)

for step in range(500):
    offset = (step * 64) % len(x)
    batch = slice(offset, offset + 64)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x[batch]), target[batch])
    loss.backward()
    optimizer.step()

print(f"final loss: {loss.item():.6f}")
memory_audit(model, optimizer)
