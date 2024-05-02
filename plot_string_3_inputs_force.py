from scipy.integrate import quad
import matplotlib.pyplot as plt
import os
from pinns.model import PINN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pinns.train import train
from pinns.dataset import DomainDataset, ICDataset

name = "output"
experiment_name = "string_adim-all_3inputs_nostiffness_force_ic0hard_icv0"
current_file = os.path.abspath(__file__)
output_dir = os.path.join(os.path.dirname(current_file), name)
output_dir = os.path.join(output_dir, experiment_name)

model_dir = os.path.join(output_dir, "model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, 'model.pt')

num_inputs = 3 #x, x_f, t

u_min = -0.4
u_max = 0.4
x_min = 0.0
x_max = 1.0
t_f = 1.0
f_min = -3.0
f_max = 0.0
delta_u = u_max - u_min
delta_x = x_max - x_min
delta_f = f_max - f_min

def hard_constraint(x, y):
    res = x[:, 0].reshape(-1, 1) * (1 - x[:, 0]).reshape(-1, 1) * y * x[:, -1].reshape(-1, 1)
    res = (res - u_min)/delta_u
    return res

def w1(x):
    return 0

def w2(x):
    return 0

a = 1
def exact(x):
    x, t = np.split(x, 2, axis=1)

    return (w1(x-a*t) + w1(x+a*t))/2

def compose_input(x, x_f, t):
    X = (x-x_min)/delta_x
    X = np.hstack((X, np.ones_like(x)*x_f))
    X = np.hstack((X, np.ones_like(x)*(t/t_f)))
    X = torch.Tensor(X).to(torch.device("cuda:0")).requires_grad_()
    return X


model = PINN([num_inputs] + [100]*3 + [1], nn.Tanh, hard_constraint).to(torch.device('cuda:0'))
model = torch.load(model_path)

fig, axes = plt.subplots(1, 1, figsize=(15, 5))

tt = np.linspace(0, t_f, num=1000)
x = np.linspace(x_min, x_max, num=1000).reshape(-1, 1)
x_f = 0.5
delta = u_max - u_min
preds = []
for t in tt:     
    X = compose_input(x, x_f, t)
    pred = model(X)
    pred = pred.cpu().detach().numpy()
    pred = pred*delta + u_min
    preds.append(pred)
preds = np.array(preds)
#print(preds)

xx, tt = np.meshgrid(x, tt)
X = np.vstack((np.ravel(xx), np.ravel(tt))).T
la = len(np.unique(X[:, 0:1]))
le = len(np.unique(X[:, 1:]))

pred = preds.reshape((le, la))



# Plot Theta Predicted
im1 = axes.imshow(pred, cmap='inferno', aspect='auto', origin='lower',
                        extent=[np.unique(X[:, 0:1]).min(), np.unique(X[:, 0:1]).max(), np.unique(X[:, 1:]).min(), np.unique(X[:, 1:]).max()])#, vmin=true.min(), vmax = true.max())
axes.set_title(f'Predicted')
axes.set_xlabel('X')
axes.set_ylabel('T')
plt.colorbar(im1, ax=axes)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f'{output_dir}/plot_corda_semplice.png')

plt.show()