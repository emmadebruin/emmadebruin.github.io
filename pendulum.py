##############################################################################################################################################################################################################
## Pendulum problem
# With L-BFGS

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import datasets, transforms, models,utils
import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy import symbols, Function, diff, sin, cos, Matrix, Rational, Eq, solve, lambdify, Derivative

# Set random seed
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

# Set constants
m_val = 1
l_val = 1
g_val = 9.8

# Set variables
t_min = 0
t_max = 10
n_samples = 1000
t_train = torch.linspace(t_min, t_max, n_samples).unsqueeze(1)

# Define the neural network class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 2)
        self.bn4 = nn.BatchNorm1d(2)

    def forward(self, x):
        x = torch.tanh(self.bn1(self.fc1(x)))
        x = torch.tanh(self.bn2(self.fc2(x)))
        x = torch.tanh(self.bn3(self.fc3(x)))
        x = self.bn4(self.fc4(x))
     
        # Split x into phi and tau outputs
        phi, tau = torch.split(x, 1, dim=1)

        # Apply tanh activation only to the tau output
        tau = torch.tanh(tau) * 1.5

        # Concatenate phi and tau back together
        return torch.cat((phi, tau), dim=1)

# Initialize the network
net = Net()

# Define loss function
def loss(t, outputs):
    phi_predicted = outputs[:, 0]
    tau_predicted = outputs[:, 1]

  # Numerical approximation for dphi/dt
    dt = t[1] - t[0] # Assuming uniform time steps
    dphi_predicted = torch.diff(phi_predicted, dim=0) / dt
    dphi_predicted = torch.cat((dphi_predicted, torch.zeros(1, device=dphi_predicted.device)), dim=0)

  # Numerical approximation for d^2phi/dt^2
    d2phi_predicted = torch.diff(dphi_predicted, dim=0) / dt
    d2phi_predicted = torch.cat((d2phi_predicted, torch.zeros(1, device=d2phi_predicted.device)), dim=0)

    F = m_val * l_val**2 * d2phi_predicted - (tau_predicted - m_val * g_val * l_val * torch.sin(phi_predicted))
    Lphys = torch.mean(F**2)

    # Constraint loss (initial conditions)
    Lcon = torch.mean((phi_predicted[0] - 0)**2 + (0 - dphi_predicted[0])**2)

    # Lgoal
    target_time_index = torch.argmin(torch.abs(t_train - 10.0)).item()
    Lgoal = (torch.cos(phi_predicted[target_time_index]) - (-1))**2

    w_phys = 1
    w_con = 10
    w_goal = 1

    # Total loss
    total_loss = w_phys * Lphys + w_con * Lcon + w_goal * Lgoal
    return total_loss, [Lphys.item(), Lcon.item(), Lgoal.item()]  # Return tuple with total loss and individual losses


# Define optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.02)

# Training loop
num_epochs = 5000

loss_history = [] # Store total loss for all iterations
physics_losses = []
constraint_losses = []
goal_losses = []

for epoch in range(num_epochs):
    # Forward pass
    outputs = net(t_train)
    phi_pred, tau_pred = torch.split(outputs, 1, dim=1)

    # Calculate loss
    loss_val, individual_losses = loss(t_train, outputs)
   
    physics_losses.append(individual_losses[0])
    constraint_losses.append(individual_losses[1])
    goal_losses.append(individual_losses[2])
    loss_history.append(loss_val.item())

    # Backward pass and update weights
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()

    # Print training progress (optional)
    if epoch % 100 == 0:
        print(f"Epoch: {epoch+1}, Loss: {loss_val.item():.4f}")
     
     
# Define the optimizer for L-BFGS
optimizer_lbfgs = torch.optim.LBFGS(net.parameters())

# Fine-tune the model using L-BFGS optimizer
def closure():
    optimizer_lbfgs.zero_grad()
    outputs = net(t_train)
    loss_val, _ = loss(t_train, outputs)
    loss_val.backward()
    return loss_val

# Train with L-BFGS
num_iterations_lbfgs = 7000
for iteration in range(num_iterations_lbfgs):
    optimizer_lbfgs.step(closure)
    if iteration % 100 == 0:
        print(f"L-BFGS Epoch: {iteration+1}, Loss: {closure().item():.4f}")


plt.figure(figsize=(20, 5), dpi=75)

# Plot position of the pendulum
plt.plot(t_train.numpy(), phi_pred.detach().numpy(), linewidth=3, color='black', label='Position (φ)')

# Plot torque
plt.plot(t_train.numpy(), tau_pred.detach().numpy(), linewidth=3, color='blue', label='Torque')

plt.ylabel('Value')
plt.xlabel('Time [s]')
plt.legend()
plt.grid(True)
plt.show()

# Define epochs array (after the training loop)
epochs = np.arange(num_epochs)

# Plot losses
plt.figure(figsize=(12, 6))

plt.plot(epochs, loss_history, label='Total Loss')
plt.plot(epochs, physics_losses, label='Physics Loss')
plt.plot(epochs, constraint_losses, label='Constraint Loss')
plt.plot(epochs, goal_losses, label='Goal Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves During Training')
plt.legend()
plt.grid(True)
plt.show()

####################################################################################################################################################################################################################
## Pendulum without L-BFGS

# Initialize the network
net = Net()

# Define loss function
def loss(t, outputs):
    phi_predicted = outputs[:, 0]
    tau_predicted = outputs[:, 1]

  # Numerical approximation for dphi/dt
    dt = t[1] - t[0] # Assuming uniform time steps
    dphi_predicted = torch.diff(phi_predicted, dim=0) / dt
    dphi_predicted = torch.cat((dphi_predicted, torch.zeros(1, device=dphi_predicted.device)), dim=0)

  # Numerical approximation for d^2phi/dt^2
    d2phi_predicted = torch.diff(dphi_predicted, dim=0) / dt
    d2phi_predicted = torch.cat((d2phi_predicted, torch.zeros(1, device=d2phi_predicted.device)), dim=0)

    F = m_val * l_val**2 * d2phi_predicted - (tau_predicted - m_val * g_val * l_val * torch.sin(phi_predicted))
    Lphys = torch.mean(F**2)

    # Constraint loss (initial conditions)
    Lcon = torch.mean((phi_predicted[0] - 0)**2 + (0 - dphi_predicted[0])**2)

    # Lgoal
    target_time_index = torch.argmin(torch.abs(t_train - 10.0)).item()
    Lgoal = (torch.cos(phi_predicted[target_time_index]) - (-1))**2

    w_phys = 1
    w_con = 10
    w_goal = 1

    # Total loss
    total_loss = w_phys * Lphys + w_con * Lcon + w_goal * Lgoal
    return total_loss, [Lphys.item(), Lcon.item(), Lgoal.item()]  # Return tuple with total loss and individual losses


# Define optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.02)

# Training loop
num_epochs = 15000

loss_history = [] # Store total loss for all iterations
physics_losses = []
constraint_losses = []
goal_losses = []

for epoch in range(num_epochs):
    # Forward pass
    outputs = net(t_train)
    phi_pred, tau_pred = torch.split(outputs, 1, dim=1)

    # Calculate loss
    loss_val, individual_losses = loss(t_train, outputs)
   
    physics_losses.append(individual_losses[0])
    constraint_losses.append(individual_losses[1])
    goal_losses.append(individual_losses[2])
    loss_history.append(loss_val.item())

    # Backward pass and update weights
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()

    # Print training progress (optional)
    if epoch % 100 == 0:
        print(f"Epoch: {epoch+1}, Loss: {loss_val.item():.4f}")
    
    plt.figure(figsize=(20, 5), dpi=75)

# Plot position of the pendulum
plt.plot(t_train.numpy(), phi_pred.detach().numpy(), linewidth=3, color='black', label='Position (φ)')

# Plot torque
plt.plot(t_train.numpy(), tau_pred.detach().numpy(), linewidth=3, color='blue', label='Torque')

plt.ylabel('Value')
plt.xlabel('Time [s]')
plt.legend()
plt.grid(True)
plt.show()

# Define epochs array (after the training loop)
epochs = np.arange(num_epochs)

# Plot losses
plt.figure(figsize=(12, 6))

plt.plot(epochs, loss_history, label='Total Loss')
plt.plot(epochs, physics_losses, label='Physics Loss')
plt.plot(epochs, constraint_losses, label='Constraint Loss')
plt.plot(epochs, goal_losses, label='Goal Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves During Training')
plt.legend()
plt.grid(True)
plt.show()

####################################################################################################################################################################################################################
# Pendulum problem: figure 5

all_goal_losses = []
all_phi_predictions = []

# Loop over different random seeds
for seed in range(10):
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Reinitialize the model with the same architecture
    net = Net()

    # Reinitialize optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.02)

    # Training loop
    loss_history = [] # Store total loss for all iterations
    physics_losses = []
    constraint_losses = []
    goal_losses = []

    for epoch in range(num_epochs):
        # Forward pass
        outputs = net(t_train)
        phi_pred, tau_pred = torch.split(outputs, 1, dim=1)

        # Calculate loss
        loss_val, individual_losses = loss(t_train, outputs)
       
        physics_losses.append(individual_losses[0])
        constraint_losses.append(individual_losses[1])
        goal_losses.append(individual_losses[2])
        loss_history.append(loss_val.item())

        # Backward pass and update weights
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Print training progress (optional)
        if epoch % 100 == 0:
            print(f"Seed: {seed}, Epoch: {epoch+1}, Loss: {loss_val.item():.4f}")
    
    all_goal_losses.append(goal_losses)
    all_phi_predictions.append(phi_pred.detach().numpy())


 # Plot goal loss for each seed
plt.figure(figsize=(12, 6))
for seed in range(10):
    plt.plot(epochs, all_goal_losses[seed], label=f'Seed {seed}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Goal Loss Curves for Different Seeds')
plt.legend()
plt.grid(True)
plt.show()

# Plot position of the pendulum for each seed
plt.figure(figsize=(12, 6))
for seed in range(10):
    plt.plot(t_train.numpy(), all_phi_predictions[seed], linewidth=3, label=f'Seed {seed}')
plt.ylabel('Value')
plt.xlabel('Time [s]')
plt.title('Position of Pendulum (φ) for Different Seeds')
plt.legend()
plt.grid(True)
plt.show()
