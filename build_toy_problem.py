import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create directory for saving plots
os.makedirs("figures", exist_ok=True)

# 1. Sample batch: (x0, x1) pairs and interpolated midpoint
def sample_pairs(batch_size):
    x1 = torch.randn(batch_size, 2) * 0.5 + 2.0  # Target: Gaussian at (2, 2)
    x0 = torch.randn(batch_size, 2)              # Source: standard Gaussian
    t = torch.rand(batch_size, 1)                # Time ∈ [0, 1]
    xt = (1 - t) * x0 + t * x1                   # Linear interpolation
    v = x1 - x0                                   # Constant velocity
    return xt, t, v

# 2. Visualize source and target distributions using ellipses
def visualize_source_and_target_distributions_with_ellipses(n_points=300):
    x0 = torch.randn(n_points, 2)
    x1 = torch.randn(n_points, 2) * 0.5 + 2.0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x0[:, 0], x0[:, 1], alpha=0.3, label="Source samples")
    ax.scatter(x1[:, 0], x1[:, 1], alpha=0.3, label="Target samples")

    # Draw ellipses representing 1-sigma contours
    source_ellipse = patches.Ellipse((0, 0), width=2, height=2,
                                     edgecolor='blue', facecolor='none', linestyle='--', linewidth=2, label="Source N(0, I)")
    target_ellipse = patches.Ellipse((2, 2), width=1.0, height=1.0,
                                     edgecolor='red', facecolor='none', linestyle='--', linewidth=2, label="Target N([2,2], 0.5² I)")

    ax.add_patch(source_ellipse)
    ax.add_patch(target_ellipse)

    ax.set_title("Source vs Target Distributions (with 1-sigma Ellipses)")
    ax.legend()
    ax.grid()
    ax.set_aspect('equal')
    ax.set_xlim(-3, 5)
    ax.set_ylim(-3, 5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig("figures/source_target_ellipses.png", dpi=150)
    plt.show()

# NEW: Combined plot of Gaussians + vector field
def visualize_distributions_with_vector_field(model, grid_size=20, n_samples=300):
    x0 = torch.randn(n_samples, 2)
    x1 = torch.randn(n_samples, 2) * 0.5 + 2.0

    # Vector field grid
    x_range = torch.linspace(-3, 5, grid_size)
    y_range = torch.linspace(-3, 5, grid_size)
    grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing="ij")
    xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    t = torch.ones(xy.shape[0], 1) * 0.5

    with torch.no_grad():
        v = model(xy, t).numpy()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x0[:, 0], x0[:, 1], alpha=0.3, label="Source samples", color="blue")
    ax.scatter(x1[:, 0], x1[:, 1], alpha=0.3, label="Target samples", color="red")

    source_ellipse = patches.Ellipse((0, 0), width=2, height=2,
                                     edgecolor='blue', facecolor='blue', alpha=0.1, label="Source N(0, I)")
    target_ellipse = patches.Ellipse((2, 2), width=1.0, height=1.0,
                                     edgecolor='red', facecolor='red', alpha=0.1, label="Target N([2,2], 0.5² I)")
    ax.add_patch(source_ellipse)
    ax.add_patch(target_ellipse)

    ax.quiver(xy[:, 0], xy[:, 1], v[:, 0], v[:, 1], angles='xy',
              scale_units='xy', scale=10, width=0.002, color='black', alpha=0.7, label="Vector field")

    ax.set_xlim(-3, 5)
    ax.set_ylim(-3, 5)
    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Gaussians and Learned Vector Field")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig("figures/combined_ellipses_vector_field.png", dpi=150)
    plt.show()

# 3. MLP to model the vector field
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

# 4. Integrate the learned flow field from x0 using Euler steps
def integrate(x0, model, steps=20):
    x = x0.clone()
    dt = 1.0 / steps
    traj = [x0]
    for i in range(steps):
        t = torch.ones_like(x[:, :1]) * (i * dt)
        with torch.no_grad():
            v = model(x, t)
        x = x + v * dt
        traj.append(x)
    return traj

# 5. Quiver plot of learned vector field
def visualize_vector_field(model, grid_size=20):
    x_range = torch.linspace(-2, 4, grid_size)
    y_range = torch.linspace(-2, 4, grid_size)
    grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing="ij")
    xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    t = torch.ones(xy.shape[0], 1) * 0.5

    with torch.no_grad():
        v = model(xy, t).numpy()

    plt.figure(figsize=(7, 7))
    plt.quiver(xy[:, 0], xy[:, 1], v[:, 0], v[:, 1], angles='xy',
               scale_units='xy', scale=10, width=0.002)
    plt.title("Learned Vector Field at t = 0.5")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid()
    plt.tight_layout()
    plt.savefig("figures/vector_field.png", dpi=150)

# 6. Visualize trajectories over vector field
def visualize_trajectories_and_field(model):
    x0 = torch.randn(100, 2)
    traj = integrate(x0, model, steps=40)

    visualize_vector_field(model)

    for i in range(0, len(traj), 5):
        x = traj[i].detach().numpy()
        plt.scatter(x[:, 0], x[:, 1], alpha=0.3, label=f't={i/40:.2f}')
    plt.legend()
    plt.title("Trajectories Overlayed on Vector Field")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid()
    plt.tight_layout()
    plt.savefig("figures/trajectories_on_field.png", dpi=150)
    plt.show()

# 7. Main training logic
def main():
    print("Visualizing source and target distributions with ellipses...")
    visualize_source_and_target_distributions_with_ellipses()

    model = VectorField()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Starting training...")
    for step in range(5000):
        xt, t, v_true = sample_pairs(128)
        pred_v = model(xt, t)
        loss = ((pred_v - v_true) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"[Step {step:04d}] Loss: {loss.item():.6f}")

    print("Training complete.\nVisualizing learned vector field and trajectories...")
    visualize_trajectories_and_field(model)

    print("Generating combined ellipse and vector field visualization...")
    visualize_distributions_with_vector_field(model)

if __name__ == "__main__":
    main()
