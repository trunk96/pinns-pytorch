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

model_path = os.path.join(model_dir, 'corda_semplice.pt')

a = 1


def w1(x):

    return np.sin(x/np.pi)


def w2(z):

    return 0


def compute_integral(n):
    integrand = lambda z: w1(z) * np.sin(n * np.pi * z )
    result, _ = quad(integrand, 0, 1)
    return result


def compute_integral2(n):
    integrand = lambda z: w2(z) * np.sin(n * np.pi * z )
    result, _ = quad(integrand, 0, 1)
    return result



def exact(x, n_max=3):
    x, t = np.split(x, 2, axis=1)
    
    integrals = [compute_integral(n) for n in range(1, n_max+1)]
    integrals2 = [compute_integral2(n) for n in range(1, n_max+1)]

    terms = [((2) * integrals[n-1] * np.cos(n * np.pi * a * t) + (2/(n*np.pi*a)) * integrals2[n-1] * np.sin(n * np.pi * a * t)) * np.sin(n * np.pi * x) for n in range(1, n_max+1)]
    return np.sum(terms, axis=0)


def hard_constraint(x, y):
    res = x[:, 0:1] * (1 - x[:, 0:1]) * y
    return res


batchsize = 32
learning_rate = 1e-3 

domainDataset = DomainDataset([0.0]*2, [1.0]*2, 1000)
icDataset = ICDataset([0.0]*2, [1.0]*2, 1000)

model = PINN([2] + [100]*3 + [1], nn.Tanh, hard_constraint).to(torch.device('cuda:0'))
model = torch.load(model_path)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
x = np.linspace(0, 1, num=100)
t = np.linspace(0, 1, num=100)
xx, tt = np.meshgrid(x, t)
X = np.vstack((np.ravel(xx), np.ravel(tt))).T

Xp = torch.Tensor(X).to(torch.device('cuda:0'))
ttrue = exact(X)
ppred = model(Xp)

la = len(np.unique(X[:, 0:1]))
le = len(np.unique(X[:, 1:]))

pred = ppred.reshape((le, la)).cpu()
pred = pred.detach().numpy()
true = ttrue.reshape((le, la))

# Plot Theta Predicted
im1 = axes[0].imshow(pred, cmap='inferno', aspect='auto', origin='lower',
                        extent=[np.unique(X[:, 0:1]).min(), np.unique(X[:, 0:1]).max(), np.unique(X[:, 1:]).min(), np.unique(X[:, 1:]).max()], vmin=true.min(), vmax = true.max())
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