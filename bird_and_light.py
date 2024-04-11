# Model and First Optimizer
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import datetime

from torch.distributions import constraints

device = torch.device("cpu") # forcing a device speeds it up a bit, no gpu because model is too small and it would actually slow down
dtype = torch.float

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

tmin, tmax = 0.0, 1.0
T = torch.tensor(1., device=device, dtype=dtype).requires_grad_(True)

# number of inputs {t}
n_input = 1
# number of outputs {x, y}
n_output = 2

# number of epochs
n_boundary = 2
n_domain = 1000
n_adam = 2000
lr = 1e-3

n_log = 100
n_save = 20

bird = False


# time domain options
continuousBoundaryDomain = True

tol = 1e-05

if not continuousBoundaryDomain:
    a = np.repeat(tmin, n_boundary - 1)
    b = np.linspace(tmin, tmax, n_domain)
    c = np.repeat(tmax, n_boundary - 1)
else:
    if n_boundary <= 2:
        a = np.repeat(tmin, n_boundary - 1)
        c = np.repeat(tmax, n_boundary - 1)

        if n_boundary == 1:
            b = np.linspace(tmin, tmax, n_domain)
        else:
            b = np.linspace(tmin + tol, tmax - tol, n_domain)
    else:
        a = np.linspace(tmin, tmin + tol, n_boundary - 1)
        c = np.linspace(tmax - tol, tmax, n_boundary - 1)

        b = np.linspace(tmin + 2*tol, tmax - 2*tol, n_domain)

t_domain = np.concatenate((a, b, c))

print(t_domain[:10])
# np.random.shuffle(t_domain)
t_domain = torch.tensor(t_domain, requires_grad=True, device=device, dtype=dtype).unsqueeze(1)

class Loss():
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0 = x0, y0
        self.x1, self.y1 = x1, y1

        self.bc = [
            torch.tensor([x0, y0], requires_grad=True, device=device, dtype=dtype),
            torch.tensor([x1, y1], requires_grad=True, device=device, dtype=dtype)
        ]

    # Default function to calculate left and right boundary loss
    def loss_bc_fn(self, model, t, x, y, dx, dy):
        # loss_bc_left = torch.linalg.vector_norm(torch.stack((x[0], y[0]), dim=1) - self.bc[0])
        # loss_bc_right = torch.linalg.vector_norm(torch.stack((x[-1], y[-1]), dim=1) - self.bc[1])
        loss_bc_left = torch.linalg.vector_norm(torch.stack((x[:n_boundary], y[:n_boundary]), dim=1).squeeze(2) - self.bc[0])
        loss_bc_right = torch.linalg.vector_norm(torch.stack((x[-n_boundary:], y[-n_boundary:]), dim=1).squeeze(2) - self.bc[1])

        # shuffled t_domain
        # indices_left = torch.isclose(t, torch.tensor(tmin), atol=1e-5)
        # indices_right = torch.isclose(t, torch.tensor(tmax), atol=1e-5)
        # loss_bc_left = torch.linalg.vector_norm(torch.stack((x[indices_left], y[indices_left]), dim=1) - self.bc[0])
        # loss_bc_right = torch.linalg.vector_norm(torch.stack((x[indices_right], y[indices_right]), dim=1) - self.bc[1])

        return torch.mean(torch.stack([loss_bc_left, loss_bc_right])**2)
    
    # Default function to calculate physical loss
    def loss_phys_fn(self, F):
        return torch.mean(torch.linalg.vector_norm(F, dim=1)**2)
    
    # Default function to calculate  goal loss
    def loss_goal_fn(self, model, t, x, y, dx, dy):
        return model.T
    
    # Default implementation to calculate loss
    def loss_fn(self, model, t, x, y, dx, dy):
        # Goal loss
        loss_goal = self.loss_goal_fn(model, t, x, y, dx, dy)

        # Boundary loss
        loss_bc = self.loss_bc_fn(model, t, x, y, dx, dy)

        # Physical loss
        F = self.F_fn(model, t, x, y, dx, dy)
        loss_phys = self.loss_phys_fn(F)

        total_loss = self.loss_weights[0] * loss_phys + self.loss_weights[1] * loss_bc + self.loss_weights[2] * loss_goal

        return total_loss, [loss_phys, loss_bc, loss_goal]
    
    # Should be overwritten by child
    def F_fn(self, t, x, y, dx, dy):
        return None

class LightLoss(Loss):
    """Loss implementation for light refraction"""
    def __init__(self):
        x0, y0 = 0., 0.
        x1, y1 = 1., 1.
        super(LightLoss, self).__init__(x0, y0, x1, y1)

        self.c0, self.n1, self.n2 = 1., 1., 2.

        self.loss_weights = torch.tensor([1., 1., .01], requires_grad=True, device=device, dtype=dtype)

    def F_fn(self, model, t, x, y, dx, dy):
        refraction = self.n1 + (self.n2 - self.n1) * 0.5 * (1. - torch.cos(2. * np.pi * y))
        vel = self.c0 / refraction
        F = (1/model.T*dx)**2 + (1/model.T*dy)**2 - vel**2
        return F

class BirdLoss(Loss):
    """Loss implementation for brachistochrone curve"""
    def __init__(self):
        x0, y0 = 0., 1.
        x1, y1 = 1., 0.
        super(BirdLoss, self).__init__(x0, y0, x1, y1)

        self.g = 9.8

        self.loss_weights = torch.tensor([1., 1., .01], requires_grad=True, device=device, dtype=dtype)

    def F_fn(self, model, t, x, y, dx, dy):
        vel = (1/model.T*dx)**2 + (1/model.T*dy)**2
        F = self.g*self.y0 - (self.g*y + 0.5 * vel)
        return F

class Net(nn.Module):
    def __init__(self, n_input, n_output, loss_cls):
        super(Net, self).__init__()

        self.T = nn.Parameter(data=T, requires_grad=True)

        self.layers = nn.DataParallel(nn.Sequential(
            nn.Linear(n_input, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_output),
            nn.Sigmoid()
        ))

        self.loss_cls = loss_cls

        self.layers.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        x = self.layers(x)

        return x
    
    def loss_fn(self, model, t, values):
        x, y = torch.split(values, 1, dim=1)

        t_tensor = t.expand_as(x).requires_grad_(True)

        # define how we calculate the dx/dt and dy/dt, only the first option seems to work
        if True:
            dt = torch.diff(t_tensor, dim=0)

            offset = 1
            if not continuousBoundaryDomain and n_boundary > offset:
                dt[:n_boundary-offset] = 1

                # 0: (or -0:) will select ALL values therefore we must check n_boundary > offset
                dt[-n_boundary+offset:] = 1

            # dt = 0.001

            dx = torch.diff(x, dim=0) / dt
            dy = torch.diff(y, dim=0) / dt

            # dx = torch.nan_to_num(dx)
            # dy = torch.nan_to_num(dy)

            dx = torch.cat((dx, torch.zeros(1, 1, requires_grad=True, device=device, dtype=dtype)), dim=0)
            dy = torch.cat((dy, torch.zeros(1, 1, requires_grad=True, device=device, dtype=dtype)), dim=0)
        else:
            # jac = torch.autograd.functional.jacobian(lambda t: torch.stack([x, y]), t_tensor)
            # dx = jac[:, 0]
            # dy = jac[:, 1]

            dx = torch.autograd.grad(x, t_tensor, grad_outputs=torch.ones_like(x), create_graph=True, allow_unused=True)[0]
            dy = torch.autograd.grad(y, t_tensor, grad_outputs=torch.ones_like(y), create_graph=True, allow_unused=True)[0]

        if dx is None:
            dx = torch.zeros_like(x, device=device, dtype=dtype).requires_grad_(True)
        if dy is None:
            dy = torch.zeros_like(y, device=device, dtype=dtype).requires_grad_(True)

        # Loss
        return self.loss_cls.loss_fn(model, t_tensor, x, y, dx, dy)

net = Net(n_input, n_output, BirdLoss() if bird else LightLoss())
net = net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

lbfgs_loss_history = None
lbfgs_partial_loss_history = None
    
loss_history = []
partial_loss_history = []

print(f'Model for {"brachistochrone curve" if bird else "light refraction"}')

starttime = datetime.datetime.now()

for epoch in range(n_adam):
    optimizer.zero_grad()

    outputs = net(t_domain)

    loss, partial_losses = net.loss_fn(net, t_domain, outputs)

    if epoch % n_save == 0:
        loss_history.append(loss.item())
        partial_loss_history.append([item.item() for item in partial_losses])

    loss.backward()
    optimizer.step()

    if epoch % n_log == 0:
        now = datetime.datetime.now()
        elapsed_time = (now - starttime).total_seconds()
        elapsed_epochs = epoch + 1

        remaining = elapsed_time / (elapsed_epochs) * (n_adam - elapsed_epochs)
        print(f"Epoch: {epoch+1:5d}, Loss: {loss.item():.4f}, Remaining time: {remaining:0.1f}s")







# Second Optimizer

# Define the optimizer for L-BFGS
optimizer_lbfgs = torch.optim.LBFGS(net.parameters(), lr=lr)

lbfgs_loss_history = []
lbfgs_partial_loss_history = []
lbfgs_loss_temp = None
lbfgs_partial_loss_temp = None

local_iteration = 0

# Fine-tune the model using L-BFGS optimizer
def closure():
    global lbfgs_loss_temp, lbfgs_partial_loss_temp
    
    optimizer_lbfgs.zero_grad()

    outputs = net(t_domain)
    
    loss, partial_losses = net.loss_fn(net, t_domain, outputs)
    
    lbfgs_loss_temp = loss
    lbfgs_partial_loss_temp = partial_losses

    loss.backward()

    return loss

starttime = datetime.datetime.now()

# Train with L-BFGS
n_lbfgs = 400
for iteration in range(n_lbfgs):
    local_iteration = iteration
    loss = optimizer_lbfgs.step(closure)

    if local_iteration % n_save == 0:
        lbfgs_loss_history.append(lbfgs_loss_temp.item())
        lbfgs_partial_loss_history.append([item.item() for item in lbfgs_partial_loss_temp])

    if iteration % n_log == 0:
        now = datetime.datetime.now()
        elapsed_time = (now - starttime).total_seconds()
        elapsed_iterations = iteration + 1

        remaining = elapsed_time / elapsed_iterations * (n_lbfgs - elapsed_iterations)
        print(f"L-BFGS Epoch: {iteration+1:5d}, Loss: {lbfgs_loss_temp.item():.4f}, Remaining time: {remaining:0.1f}s")






# Plotting

pred_domain = torch.linspace(tmin, tmax, 101, device=device).unsqueeze(1)
pred = net(pred_domain)

pred = torch.Tensor.cpu(pred)

plt.plot(pred[:, 0].detach().numpy(), pred[:, 1].detach().numpy(), linewidth=2.5, color='blue', linestyle="-", label='PINN')

ax = plt.gca()
ax.set_ylim([-0.1, 1.1])
ax.set_xlim([-0.1, 1.1])

# analytical solution
if bird:
    r = 0.5729
    theta = np.linspace(0, np.arccos(1 - 1 / r), 101)
    x_analytical = r * (theta - np.sin(theta))
    y_analytical = 1 - r * (1 - np.cos(theta))
else:
    y_analytical = np.linspace(0, 1, 101)
    x_analytical = np.arctan(2. * np.tan(np.pi * y_analytical)) / np.pi
    x_analytical[len(x_analytical) // 2 + 1:] += 1

plt.plot(x_analytical, y_analytical, linewidth=1.5, color='orange', linestyle="--", label='Analytical')
plt.legend()

ax.set_aspect('equal', 'box')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_name = "Shortest path under gravity" if bird else "Light path of shortest time"
ax.set_title(f'{plot_name}\nT = %.3f s' % net.T.item())

plt.tight_layout()
plt.show()

def plot_losses(name, loss, partial_losses, numiter):
    xrange = np.linspace(0, numiter, num=len(loss))

    loss_phys = [item[0] for item in partial_losses]
    loss_con = [item[1] for item in partial_losses]
    loss_goal = [item[2] for item in partial_losses]

    ax = plt.subplot(2, 1, 1)
    ax.plot(xrange, loss)
    ax.set_title(f'{name} - {plot_name}\nLoss')
    plt.xlabel("Num iterations")

    ax = plt.subplot(2, 3, 4)
    ax.plot(xrange, loss_phys)
    ax.set_title(f'Physical loss')
    ax.set_ylim([0, 1.1])

    ax = plt.subplot(2, 3, 5)
    ax.plot(xrange, loss_con)
    ax.set_title(f'Boundary loss')

    plt.xlabel("Num iterations")

    ax = plt.subplot(2, 3, 6)
    ax.plot(xrange, loss_goal)
    ax.set_title(f'Goal loss')

    plt.tight_layout()
    plt.show()

plot_losses("PINN", loss_history, partial_loss_history, n_adam)
if lbfgs_loss_history is not None:
    plot_losses("LBFGS", lbfgs_loss_history, lbfgs_partial_loss_history, n_lbfgs)
