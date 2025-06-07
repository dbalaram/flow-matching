import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Step 1: Create sample pairs
def sample_pairs(batch_size):
    x1 = torch.randn(batch_size, 2) * 0.5 + 2.0  # Target distribution (mean 2, std 0.5)
    x0 = torch.randn(batch_size, 2)              # Standard Gaussian
    t = torch.rand(batch_size, 1)                # Uniform time in [0, 1]
    xt = (1 - t) * x0 + t * x1                   # Linear interpolation
    v = x1 - x0                                   # Velocity
    return xt, t, v

# Step 2: Define the vector field model
class VectorField(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)

# Step 3: Training
model = VectorField()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for step in range(5000):
    xt, t, v_true = sample_pairs(128)
    pred_v = model(xt, t)
    loss = ((pred_v - v_true)**2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# Step 4: Sample trajectory by integrating the vector field (Euler)
def integrate(x0, model, steps=20):
    x = x0.clone()
    dt = 1.0 / steps
    traj = [x0]
    for i in range(steps):
        t = torch.ones_like(x[:, :1]) * (i * dt)
        v = model(x, t)
        x = x + v * dt
        traj.append(x)
    return traj

# Visualize
x0 = torch.randn(100, 2)
traj = integrate(x0, model, steps=40)

plt.figure(figsize=(6, 6))
for i in range(0, len(traj), 5):
    x = traj[i].detach().numpy()
    plt.scatter(x[:, 0], x[:, 1], alpha=0.5, label=f't={i/40:.2f}')
plt.legend()
plt.title("Learned Flow Trajectories")
plt.grid()
plt.show()
