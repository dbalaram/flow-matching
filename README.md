# Toy Example: Flow Matching in 2D

This repository demonstrates a minimal implementation of flow matching using a toy problem in 2D. The goal is to learn a neural vector field that maps samples from a standard Gaussian distribution to a target Gaussian distribution by modeling the intermediate "flow" over time.

---

## Concept

In flow matching, we train a time-conditioned vector field v_theta(x, t) such that:

- Given a source point x0 ~ p0 (e.g., standard Gaussian)
- And a target point x1 ~ p1 (e.g., translated Gaussian)
- We supervise the model using interpolated points:
  x(t) = (1 - t) * x0 + t * x1
- And their corresponding velocity:
  dx/dt = x1 - x0

The model learns to predict dx/dt at intermediate points x(t), effectively learning a flow field that maps noise into structure.

---

## Why This Works

Linear interpolation between x0 and x1 has constant velocity v = x1 - x0. By sampling time points t in [0, 1], we create a dense supervision signal to train our model without needing full trajectory simulation.

---

## Components

### `sample_pairs(batch_size)`
Generates:

- x0 ~ N(0, I)
- x1 ~ N(μ, σ² I)
- x(t) = (1 - t) * x0 + t * x1
- dx/dt = x1 - x0

### `VectorField`
A small MLP that takes in `[x, t]` and predicts the velocity.

### Training Loop
The model is trained to minimize the mean squared error (MSE) between the predicted velocity and the true velocity:

    Loss = E[ || v_theta(x(t), t) - (x1 - x0) ||^2 ]

### Trajectory Integration
Once trained, we integrate the flow field starting from x0 ~ N(0, I) forward in time using Euler’s method.

---

## Visualization

We generate a trajectory starting from noise and show how the learned field pushes samples toward the target distribution.

Example output:

- t = 0.0: Standard Gaussian samples
- t = 1.0: Shifted Gaussian (centered at (2, 2))
- Intermediate steps show the learned flow transformation

---

## Run It

You’ll need `torch` and `matplotlib` installed.

To run the example:

```bash
python build_toy_problem.py
```

## Q&A: Flow Matching Concepts

#### Q1: Are we trying to learn an approximation to the flow?
Yes — more precisely, we are learning an approximation to the vector field that defines the flow. The flow is a collection of trajectories — solutions to a differential equation. These trajectories are induced by a vector field v(x, t). We train a neural network to approximate this vector field. Once the vector field is learned, we can compute the flow by integrating it over time starting from a noise sample.

#### Q2: Is the neural network predicting the vector field at different points?

Exactly.The network takes input: a position x and a time t. It outputs a velocity vector v(x, t), representing the direction and speed of motion at that point in space-time.This defines a time-dependent vector field, which we can integrate to produce trajectories (the flow).

#### Q3: What's the difference between a vector field, flow, and dx/dt?

Vector Field: A function defining the velocity of a point at position x and time t(Predicted by the neural network)

Flow: The trajectory x(t) followed by a point moving under the vector field	Computed by integrating v(x, t) from x0

dx/dt: The instantaneous velocity of a moving point x(t)	Supervised using x1 - x0 during training

##### In this toy example:

We generate training samples by linearly interpolating between two points:
x(t) = (1 - t) * x0 + t * x1

The ground truth velocity is constant:
dx/dt = x1 - x0

The model is trained to match:
v_theta(x(t), t) ≈ dx/dt

After training, we can generate flows by integrating from x0 using the learned vector field.

## References
Lipman et al., "Flow Matching for Generative Modeling", NeurIPS 2023

Song et al., "Score-Based Generative Modeling through SDEs