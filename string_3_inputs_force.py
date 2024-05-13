from pinns.model import PINN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pinns.train import train
from pinns.gradient import jacobian
from pinns.dataset import DomainDataset, ICDataset, ValidationDataset, ValidationICDataset


epochs = 1000
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

params = {
    "u_min": u_min,
    "u_max": u_max,
    "x_min": x_min,
    "x_max": x_max,
    "t_f": t_f,
    "f_min": f_min,
    "f_max": f_max
}

def hard_constraint(x, y):
    X = x[:, 0].reshape(-1, 1)
    tau = x[:, -1].reshape(-1, 1)
    U = ((X-1)*X*(delta_x**2)*t_f*tau)*(y+(u_min/delta_u)) - (u_min/delta_u)
    return U

def f(sample):
    x = sample[:, 0].reshape(-1, 1)*(delta_x) + x_min
    x_f = sample[:, 1].reshape(-1, 1)*(delta_x) + x_min
    t = sample[:, -1].reshape(-1, 1)*t_f
    
    height = -1

    alpha = 53.59
    z = height * torch.exp(-400*((x-x_f)**2)) * (4**alpha * t**(alpha - 1) * (1 - t)**(alpha - 1))
    return z


def pde_fn(prediction, sample):
    T = 1
    mu = 1
    alpha_2 = (T/mu)*(t_f**2)/(delta_x**2)
    beta = (t_f**2)/delta_u
    dX = jacobian(prediction, sample, j=0)
    dtau = jacobian(prediction, sample, j=1)
    ddX = jacobian(dX, sample, j = 0)
    ddtau = jacobian(dtau, sample, j = 2)
    return ddtau - alpha_2*ddX - beta*f(sample)


def ic_fn_vel(prediction, sample):
    dtau = jacobian(prediction, sample, j=2)
    dt = dtau*delta_u/t_f
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
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

data = {
    "name": "string_2inputs_nostiffness_force_ic0hard_icv0_prova",
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
    "validation_ic_dataset": validationicDataset,
    "additional_data": params
}

train(data)
