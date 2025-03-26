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
from scipy.io import savemat

name = "output"
experiment_name = "string_4inputs_force_time_damping_ic0hard_icv0_t10.0_MLP_rff1.0"
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

num_inputs = 4 #x, x_f, f, t

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

def hard_constraint(x, y):
    X = x[:, 0].reshape(-1, 1)
    tau = x[:, -1].reshape(-1, 1)
    U = ((X-1)*X*(delta_x**2)*t_f*tau)*(y+(u_min/delta_u)) - (u_min/delta_u)
    return U
   

def compose_input(x, x_f_1, x_f_2, tt):
    X_ = np.array((x-x_min)/delta_x)
    X_ = np.tile(X_, (tt.shape[0], 1))
    X = np.hstack((X_, np.ones_like(X_)*((x_f_1-x_min)/delta_x)))
    X = np.hstack((X, np.ones_like(X_)*((x_f_2-x_min)/delta_x)))
    T_ = np.repeat(tt/t_f, x.shape[0]).reshape(-1, 1)
    X = np.hstack((X, T_))
    X = torch.Tensor(X).to(torch.device("cuda:0")).requires_grad_()
    return X


model = torch.load(model_path)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

tt = np.linspace(0, t_f, num=1001, endpoint=True)
x = np.linspace(x_min, x_max, num=101, endpoint=True).reshape(-1, 1)
x_f_1 = 0.2
x_f_2 = 0.8
f = -1.0

 
X = compose_input(x, x_f_1, x_f_2, tt)
preds = model(X)
preds = preds.cpu().detach().numpy()
preds = preds*delta_u + u_min
preds = np.array(preds)

preds_matlab = preds.reshape((len(tt), x.shape[0]))
mdic = {"pinn_data": preds_matlab, "x": x, "t": tt, "x_f_1": x_f_1, "x_f_2": x_f_2}
savemat(output_dir+"/data.mat", mdic)

ttrue = exact()

xx, tt = np.meshgrid(x, tt)
X = np.vstack((np.ravel(xx), np.ravel(tt))).T
la = len(np.unique(X[:, 0:1]))
le = len(np.unique(X[:, 1:]))

pred = preds.reshape((le, la))
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