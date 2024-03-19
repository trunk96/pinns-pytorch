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

model1_path = os.path.join(model_dir, 'corda_semplice.pt')
model2_path = os.path.join(model_dir, 'main.pt')


def hard_constraint(x, y):
    res = x[:, 0:1] * (1 - x[:, 0:1]) * y
    return res


def w1(x):

    ic = torch.sin(x/torch.pi)
    return ic


batchsize = 32
learning_rate = 1e-3 

domainDataset = DomainDataset([0.0]*2, [1.0]*2, 1000)
icDataset = ICDataset([0.0]*2, [1.0]*2, 1000)

model1 = PINN([2] + [100]*3 + [1], nn.Tanh, hard_constraint).to(torch.device('cuda:0'))
model1 = torch.load(model1_path)

model2 = PINN([8] + [100]*3 + [1], nn.Tanh, hard_constraint).to(torch.device('cuda:0'))
model2 = torch.load(model2_path)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
x = np.linspace(0, 1, num=100)
t = np.linspace(0, 1, num=100)
xx, tt = np.meshgrid(x, t)
X = np.vstack((np.ravel(xx), np.ravel(tt))).T
la = len(np.unique(X[:, 0:1]))
le = len(np.unique(X[:, 1:]))

Xp = torch.Tensor(X).to(torch.device('cuda:0'))
ssimple = model1(Xp)
simple = ssimple.reshape((le, la)).cpu()
simple = simple.detach().numpy()

X_grid1 = np.vstack((np.ravel(xx), np.full_like(np.ravel(xx), w1(0.25)), np.full_like(np.ravel(xx), w1(0.5)), np.full_like(np.ravel(xx), w1(0.75)),
                     np.zeros_like(np.ravel(xx)), np.zeros_like(np.ravel(xx)), np.zeros_like(np.ravel(xx)), np.ravel(tt))).T

Xgrid1 = torch.Tensor(X_grid1).to(torch.device('cuda:0'))
ffirst = model2(Xgrid1)
first = ffirst.reshape((le, la)).cpu()
first = first.detach().numpy()

# ssecond = []
# for el in np.unique(X[:, 1:]):
#     if el == 0:
#         x_grid2 = np.vstack((x, np.full_like(x, w1(0.25)), np.full_like(x, w1(0.5)), np.full_like(x, w1(0.75)),
#                             np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)))
#         xgrid2 = torch.Tensor(x_grid2).to(torch.device('cuda:0'))
#         ssecond.append(model2)
#     else:
#         x_grid2 = np.vstack((x, np.full_like(x, w1(0.25)), np.full_like(x, w1(0.5)), np.full_like(x, w1(0.75)),
#                     np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)))
#         xgrid2 = torch.Tensor(x_grid2).to(torch.device('cuda:0'))
#         ssecond.append(model2)



# Plot corda semplice
im1 = axes[0].imshow(simple, cmap='inferno', aspect='auto', origin='lower',
                        extent=[np.unique(X[:, 0:1]).min(), np.unique(X[:, 0:1]).max(), np.unique(X[:, 1:]).min(), np.unique(X[:, 1:]).max()])
axes[0].set_title(f'Corda semplice')
axes[0].set_xlabel('X')
axes[0].set_ylabel('T')
plt.colorbar(im1, ax=axes[0])

# Plot first scenario
im2 = axes[1].imshow(first, cmap='inferno', aspect='auto', origin='lower',
                        extent=[np.unique(X[:, 0:1]).min(), np.unique(X[:, 0:1]).max(), np.unique(X[:, 1:]).min(), np.unique(X[:, 1:]).max()])
axes[1].set_title(f'Primo scenario')
axes[1].set_xlabel('X')
axes[1].set_ylabel('T')
plt.colorbar(im2, ax=axes[1])

# Plot Difference
im3 = axes[2].imshow(simple, cmap='inferno', aspect='auto', origin='lower',
                        extent=[np.unique(X[:, 0:1]).min(), np.unique(X[:, 0:1]).max(), np.unique(X[:, 1:]).min(), np.unique(X[:, 1:]).max()])
axes[2].set_title(f'Secondo scenario')
axes[2].set_xlabel('X')
axes[2].set_ylabel('T')
plt.colorbar(im3, ax=axes[2])

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f'{output_dir}/plot_main.png')

plt.show()