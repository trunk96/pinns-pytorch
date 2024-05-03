from pinns.model import PINN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pinns.train import train
from pinns.gradient import jacobian
from pinns.dataset import DomainDataset, ICDataset, ValidationDataset, ValidationICDataset


epochs = 1000
num_inputs = 2 #x, t


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

def f(sample):
    x = sample[:, 0].reshape(-1, 1)
    x_f = 0.6
    height = 1.0
    t = sample[:, -1].reshape(-1, 1)

    alpha = 8.9
    z = -height * torch.exp(-400*((x-x_f)**2)) * (4**alpha * t**(alpha - 1) * (1 - t)**(alpha - 1))
    return (z-f_min)/delta_f


def pde_fn(prediction, sample):
    T = 1
    mu = 1
    dx = jacobian(prediction, sample, j=0)
    dt = jacobian(prediction, sample, j=1)
    ddx = jacobian(dx, sample, j = 0)
    ddt = jacobian(dt, sample, j = 1)
    return (delta_u/(t_f**2)) * ddt - (delta_u/(delta_x**2))*(T/mu)*ddx - f(sample)*delta_f - f_min


def ic_fn_vel(prediction, sample):
    dt = jacobian(prediction, sample, j=1)/delta_u
    ics = torch.zeros_like(dt)
    return dt, ics


batchsize = 10000
learning_rate = 1e-3

print("Building Domain Dataset")
domainDataset = DomainDataset([0.0]*num_inputs,[1.0]*num_inputs, 100000, period = 3)
print("Building IC Dataset")
icDataset = ICDataset([0.0]*(num_inputs-1),[1.0]*(num_inputs-1), 100000, period = 3)
print("Building Validation Dataset")
validationDataset = ValidationDataset([0.0]*num_inputs,[1.0]*num_inputs, batchsize)
print("Building Validation IC Dataset")
validationicDataset = ValidationICDataset([0.0]*(num_inputs-1),[1.0]*(num_inputs-1), batchsize)

model = PINN([num_inputs] + [100]*3 + [1], nn.Tanh, hard_constraint).to(torch.device('cuda:0'))

def init_normal(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

model.apply(init_normal)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

data = {
    "name": "string_adim_2inputs_nostiffness_ic0hard_icv0",
    "model": model,
    "epochs": epochs,
    "batchsize": batchsize,
    "optimizer": optimizer,
    "scheduler": scheduler,
    "pde_fn": pde_fn,
    "ic_fns": [ic_fn_vel],
    "domain_dataset": domainDataset,
    "ic_dataset": icDataset,
    "validation_domain_dataset": validationDataset,
    "validation_ic_dataset": validationicDataset
}

train(data)