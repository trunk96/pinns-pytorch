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
import json

name = "output"
experiment_name = "string_2inputs_nostiffness_force_ic0hard_icv0_prova_1"
current_file = os.path.abspath(__file__)
output_dir = os.path.join(os.path.dirname(current_file), name)
output_dir = os.path.join(output_dir, experiment_name)

model_dir = os.path.join(output_dir, "model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, 'model.pt')


exact_solution = "C:\\Users\\desan\\Documents\\Wolfram Mathematica\\file.csv"

def exact():
    sol = []
    with open(exact_solution, "r") as f:
        for line in f:
            s = line.split(",")[2].strip()
            s = s.replace('"', '').replace("{", "").replace("}", "").replace("*^", "E")
            s = float(s)
            sol.append(s)
    return np.array(sol)

def load_params(path):
    global u_min, u_max, f_min, f_max, x_min, x_max, t_f, delta_u, delta_x, delta_f
    with open(path, "r") as fp:
        params = json.load(fp)["additionalData"]
        u_min = params["u_min"]
        u_max = params["u_max"]
        x_min = params["x_min"]
        x_max = params["x_max"]
        f_min = params["f_min"]
        f_max = params["f_max"]
        t_f = params["t_f"]
        delta_u = u_max - u_min
        delta_x = x_max - x_min
        delta_f = f_max - f_min
    return

load_params(os.path.join(output_dir, "params.json"))

""" def hard_constraint(x, y):
    res = x[:, 0].reshape(-1, 1) * (1 - x[:, 0]).reshape(-1, 1) * y * x[:, -1].reshape(-1, 1)
    res = (res - u_min)/delta
    return res """

def hard_constraint(x, y):
    """ s = x[:, 0].reshape(-1, 1)*(delta_x) + x_min #x
    t = x[:, -1].reshape(-1, 1)*t_f #t
    y = y*(delta_u)+u_min
    u = (x_min - s) * (x_max - s) * y * t
    U = (u-u_min)/delta_u """
    X = x[:, 0].reshape(-1, 1)
    tau = x[:, -1].reshape(-1, 1)
    U = ((X-1)*X*(delta_x**2)*t_f*tau)*(y+(u_min/delta_u)) - (u_min/delta_u)
    return U

model = torch.load(model_path)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
x = np.linspace(0, 1, num=101, endpoint=True)
t = np.linspace(0, 1, num=101, endpoint=True)
xx, tt = np.meshgrid(x, t)
X = np.vstack((np.ravel(xx), np.ravel(tt))).T

Xp = torch.Tensor(X).to(torch.device('cuda:0')).requires_grad_()
# Xp = X
ttrue = exact()
ppred = model(Xp)

ppred = ppred.cpu().detach().numpy()
ppred = ppred*delta_u + u_min

la = len(np.unique(X[:, 0:1]))
le = len(np.unique(X[:, 1:]))

#pred = ppred.reshape((le, la)).cpu()
pred = ppred.reshape((le, la))
#pred = pred.detach().numpy()
true = ttrue.reshape((le, la), order="F")

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