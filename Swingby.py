
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Set random seed
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

# Set hyperparameters
n_output = 2  # x, y
num_domain = 1000
n_adam = 60000
lr = 3e-3  # for Adam
loss_weights = [1., 1., 1., 1., 1., 1.]

# Set physical parameters
tmin, tmax = 0.0, 1.0  # normalized time
T = torch.tensor(1.0, requires_grad=True)  # end time
bh_xygm = [
    [-0.5, -1.0, 0.5],
    [-0.2, 0.4, 1.0],
    [0.8, 0.3, 0.5],
]
m0 = 1.
x0 = -1
x1 = 1
y0 = -1
y1 = 1

# Generate domain data
domain_data = torch.linspace(tmin, tmax, num_domain).unsqueeze(1)

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
        self.fc4 = nn.Linear(64, n_output)
        self.bn4 = nn.BatchNorm1d(n_output)

    def forward(self, x):
        x = torch.tanh(self.bn1(self.fc1(x)))
        x = torch.tanh(self.bn2(self.fc2(x)))
        x = torch.tanh(self.bn3(self.fc3(x)))
        x = torch.tanh(self.bn4(self.fc4(x)))
        return x

# Initialize the network
net = Net()

def loss(outputs):
    x, y = torch.split(outputs, 1, dim=1)
    x_predicted = outputs[:, 0]
    y_predicted = outputs[:, 1]

    # Initialize sumx and sumy
    sumx = torch.zeros_like(x_predicted)
    sumy = torch.zeros_like(y_predicted)

    dt = domain_data[1] - domain_data[0]
    
    # Calculate the forces
    for x_o, y_o, GM_o in bh_xygm:
        sumx += -GM_o * m0* (x_predicted - x_o) / ((x_predicted - x_o)**2 + (y_predicted - y_o)**2)**1.5
        sumy += -GM_o * m0* (y_predicted - y_o) / ((x_predicted - x_o)**2 + (y_predicted - y_o)**2)**1.5

    # Calculate force_x and force_y
    force_x = 1 / T**2 * torch.diff(torch.diff(x_predicted, dim=0) / dt, dim=0) / dt
    force_y = 1 / T**2 * torch.diff(torch.diff(y_predicted, dim=0) / dt, dim=0) / dt

    force_x = torch.cat((force_x, torch.zeros(2)))
    force_y = torch.cat((force_y, torch.zeros(2)))

    thrust_x = force_x-sumx
    thrust_y = force_y-sumy

    # Calculate the physical loss (mean squared error)
    Lphys = torch.mean(thrust_x**2 + thrust_y**2)

    # Boundary loss (Dirichlet boundary conditions)
    loss_bc_left = torch.mean((x_predicted[:1] - x0)**2 + (y_predicted[:1] - y0)**2)
    loss_bc_right = torch.mean((x_predicted[-1:] - x1)**2 + (y_predicted[-1:] - y1)**2)
    Lcon = torch.mean(torch.stack([loss_bc_left, loss_bc_right]))

    # Weighting factors
    w_phys = 1
    w_con = 1

    # Total loss
    total_loss = w_phys * Lphys + w_con * Lcon
    return total_loss


# Define optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

for iteration in range(n_adam):
    optimizer.zero_grad()
    # Forward pass
    predictions = net(domain_data)
    # Loss calculation
    lossval = loss(predictions)
    lossval.backward()
    optimizer.step()

    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Loss: {lossval.item()}")

# After training, make predictions and plot
with torch.no_grad():
    predictions = net(domain_data).numpy()

# Plot x against y
plt.figure(figsize=(8, 6))
plt.plot(predictions[:, 0], predictions[:, 1], label='Predicted Trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Predicted Trajectory')
plt.legend()
for xtmp, ytmp, gmtmp in bh_xygm:
    plt.scatter(xtmp, ytmp, s=gmtmp * 500, c='r', marker='o')
plt.show()

# Calculate the magnitude of the total force exerted on the spacecraft at each time step
total_force = torch.zeros(num_domain)
# Calculate the magnitude of the total thrust exerted on the spacecraft at each time step
total_thrust = torch.zeros(num_domain)
# Calculate the gravitational forces exerted by each object
gravity_x = 0
gravity_y = 0
for xtmp, ytmp, gmtmp in bh_xygm:
    r_square = (predictions[:, 0] - xtmp) ** 2 + (predictions[:, 1] - ytmp) ** 2
    gravity_x += gmtmp * m0 * (predictions[:, 0] - xtmp) / r_square ** 1.5
    gravity_y += gmtmp * m0 * (predictions[:, 1] - ytmp) / r_square ** 1.5
    
# Convert to tensors
force_x = torch.tensor(gravity_x)
force_y = torch.tensor(gravity_y)

# Convert predictions to PyTorch tensor
predictions_tensor = torch.tensor(predictions)

# Calculate sumx and sumy for the entire trajectory
sumx = 0
sumy = 0
sumx = torch.tensor(sumx)
sumy = torch.tensor(sumy)
for x_o, y_o, GM_o in bh_xygm:
    sumx += -GM_o * m0* (predictions[:, 0] - x_o) / ((predictions[:, 0] - x_o)**2 + (predictions[:, 1] - y_o)**2)**1.5
    sumy += -GM_o * m0* (predictions[:, 1] - y_o) / ((predictions[:, 0] - x_o)**2 + (predictions[:, 1] - y_o)**2)**1.5

# Calculate derivatives
dt = domain_data[1] - domain_data[0]
diffx = 1 / T**2 * torch.diff(torch.diff(predictions_tensor[:, 0], dim=0) / dt, dim=0) / dt
diffy = 1 / T**2 * torch.diff(torch.diff(predictions_tensor[:, 1], dim=0) / dt, dim=0) / dt

diffx = torch.cat((diffx, torch.zeros(2)))
diffy = torch.cat((diffy, torch.zeros(2)))
# Calculate thrust_x and thrust_y for the current position
thrust_x = diffx -sumx
thrust_y = diffy -sumy

# Calculate the magnitude of the total gravity force and thrust
total_gravity = torch.sqrt(force_x ** 2 + force_y ** 2)
total_thrust=torch.sqrt(thrust_x ** 2 + thrust_y ** 2)
total_force = torch.sqrt(thrust_x ** 2 + thrust_y ** 2)+ total_gravity


# Plot time against gravity
plt.figure(figsize=(8, 6))
plt.plot(domain_data.numpy(), total_gravity.numpy(), label='Gravity')
plt.xlabel('Time')
plt.ylabel('Gravity')
plt.title('Gravity Over Time')
plt.legend()
plt.show()


# Plot time against exerted force
plt.figure(figsize=(8, 6))
plt.plot(domain_data.numpy(), total_force.detach().numpy(), label='Exerted Force')
plt.xlabel('Time')
plt.ylabel('Exerted Force')
plt.title('Exerted Force Over Time')
plt.legend()
plt.show()

# Plot time against total thrust
plt.figure(figsize=(8, 6))
plt.plot(domain_data.numpy(), total_thrust.detach().numpy(), label='Total Thrust')
plt.xlabel('Time')
plt.ylabel('Total Thrust')
plt.title('Total Thrust Over Time')
plt.legend()
plt.show()