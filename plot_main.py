from scipy.integrate import quad
import matplotlib.pyplot as plt
import os
from model import PINN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from train import train
from dataset import DomainDataset, ICDataset

name = "output"
current_file = os.path.abspath(__file__)
output_dir = os.path.join(os.path.dirname(current_file), name)

model_dir = os.path.join(output_dir, "model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

#model1_path = os.path.join(model_dir, 'corda_semplice.pt')
model_path = os.path.join(model_dir, 'main.pt')

num_ic = 10
num_inputs = 2 + (num_ic-2)*2 #x, t, 10 initial conditions on position, 10 initial conditions on velocity

FLAG_STEP_BY_STEP = True

def hard_constraint(x, y):
    res = x[:, 0:1] * (1 - x[:, 0:1]) * y
    return res


def w1(x):
    ic = np.sin(x*np.pi)
    return ic

def w2(x):
    return 0

a = 1
def exact(x):
    x, t = np.split(x, 2, axis=1)

    return (w1(x-a*t) + w1(x+a*t))/2

def compose_input(x, pos, vel, t):
    X = x
    for i in range(num_ic-2):
        X = np.hstack((X, np.ones_like(x)*pos[i]))
    for i in range(num_ic-2):
        X = np.hstack((X, np.ones_like(x)*vel[i]))
    X = np.hstack((X, np.ones_like(x)*t))
    X = torch.Tensor(X).to(torch.device("cuda:0")).requires_grad_()
    return X


model = PINN([num_inputs] + [100]*3 + [1], nn.Tanh, hard_constraint).to(torch.device('cuda:0'))
model = torch.load(model_path)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

tt = np.linspace(0, 1, num=100)
delta_t = 1/100
x = np.linspace(0, 1, num=100).reshape(-1, 1)
x_ic = np.linspace(0, 1, num = num_ic - 1)
pos_ics = np.array([w1(x) for x in x_ic[1:]])
vel_ics = np.array([w2(x) for x in x_ic[1:]])
preds = []
for t in tt:  
    if t != 0.0 and FLAG_STEP_BY_STEP:
        X = compose_input(x_ic[1:].reshape(-1, 1), pos_ics, vel_ics, delta_t)
        pred  = model(X)
        #print(f"PRIMA: {pos_ics}")
        pos_ics = pred[:, 0].cpu().detach().numpy()
        #print(f"DOPO: {pos_ics}")
        vel_ics = pred[:, -1].cpu().detach().numpy()    
    if FLAG_STEP_BY_STEP:
        X = compose_input(x, pos_ics, vel_ics, tt[0])
    else:
        X = compose_input(x, pos_ics, vel_ics, t)
    pred = model(X)
    preds.append(pred[:, 0].cpu().detach().numpy())
preds = np.array(preds)
print(preds)

xx, tt = np.meshgrid(x, tt)
X = np.vstack((np.ravel(xx), np.ravel(tt))).T
ttrue = exact(X)

la = len(np.unique(X[:, 0:1]))
le = len(np.unique(X[:, 1:]))

pred = preds.reshape((le, la))
true = ttrue.reshape((le, la))



# Plot Theta Predicted
im1 = axes[0].imshow(pred, cmap='inferno', aspect='auto', origin='lower',
                        extent=[np.unique(X[:, 0:1]).min(), np.unique(X[:, 0:1]).max(), np.unique(X[:, 1:]).min(), np.unique(X[:, 1:]).max()])#, vmin=true.min(), vmax = true.max())
axes[0].set_title(f'Predicted')
axes[0].set_xlabel('X')
axes[0].set_ylabel('T')
plt.colorbar(im1, ax=axes[0])

# Plot Theta True
im2 = axes[1].imshow(true, cmap='inferno', aspect='auto', origin='lower',
                        extent=[np.unique(X[:, 0:1]).min(), np.unique(X[:, 0:1]).max(), np.unique(X[:, 1:]).min(), np.unique(X[:, 1:]).max()])
axes[1].set_title(f'True')
axes[1].set_xlabel('X')
axes[1].set_ylabel('T')
plt.colorbar(im2, ax=axes[1])

# Plot Difference
im3 = axes[2].imshow(np.abs(pred-true), cmap='inferno', aspect='auto', origin='lower',
                        extent=[np.unique(X[:, 0:1]).min(), np.unique(X[:, 0:1]).max(), np.unique(X[:, 1:]).min(), np.unique(X[:, 1:]).max()])
axes[2].set_title(f'Difference')
axes[2].set_xlabel('X')
axes[2].set_ylabel('T')
plt.colorbar(im3, ax=axes[2])

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f'{output_dir}/plot_corda_semplice.png')

plt.show()