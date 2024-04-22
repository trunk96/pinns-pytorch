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


def hard_constraint(x, y):
    return x[:, 0] * (1 - x[:, 0]) * y * x[:,-1]

def f(sample):
    x = sample[:, 0]
    #y = sample[:, 1]
    """ x_f = sample[:, 2]
    y_f = sample[:, 3]
    height = sample[:, 4] """
    x_f = 0.8
    #y_f = 0.8
    height = 1
    t = sample[:, -1]

    alpha = 8.9
    za = -height * torch.exp(-400*((x-x_f)**2)) * (4**alpha * t**(alpha - 1) * (1 - t)**(alpha - 1))
    return za


def pde_fn(prediction, sample):
    T = 1
    mu = 1
    dx = jacobian(prediction, sample, j=0)
    dt = jacobian(prediction, sample, j=1)
    ddx = jacobian(dx, sample, j = 0)
    ddt = jacobian(dt, sample, j = 1)
    return ddt - (T/mu)*ddx


def ic_fn_pos(prediction, sample):
    #ics = torch.zeros_like(prediction[:, 0])
    return prediction[:, 0], torch.sin(sample[:, 0]*np.pi)
    return prediction[:, 0], ics

def ic_fn_vel(prediction, sample):
    dt = jacobian(prediction, sample, j=1)
    ics = torch.zeros_like(dt)
    return dt, ics



batchsize = 10000 
learning_rate = 1e-5 

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
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

data = {
    "name": "string_2inputs_nostiffness_ic0hard_icv0",
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
