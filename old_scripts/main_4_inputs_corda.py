from pinns.model import PINN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pinns.train import train
from pinns.dataset import DomainDataset, ICDataset, ValidationDataset, ValidationICDataset


epochs = 1000
num_inputs = 4 #x, x_f, f, t


def hard_constraint(x, y):
    return x[:, 0:1] * (1 - x[:, 0:1]) * y

def f(sample):
    x = sample[:, 0]
    x_f = sample[:, 1]
    height = sample[:, 2]
    t = sample[:, -1]

    alpha = 8.9
    za = -height * torch.exp(-400*((x-x_f)**2)) * (4**alpha * t**(alpha - 1) * (1 - t)**(alpha - 1))
    return za

def pde_fn(prediction, sample):
    T = 1
    mu = 1
    ESK2 = 3.926790540455574e-06
    grads = torch.zeros_like(prediction)
    grads[:, 0] = 1 #first component of the output is the actual output and ont all the derivatives w.r.t. the inputs
    d = torch.autograd.grad(prediction, sample, grad_outputs=grads,create_graph = True)[0]
    dd = torch.autograd.grad(d, sample, grad_outputs=torch.ones_like(d),create_graph = True)[0]
    ddd = torch.autograd.grad(dd, sample, grad_outputs=torch.ones_like(dd),create_graph = True)[0]
    dddd = torch.autograd.grad(ddd, sample, grad_outputs=torch.ones_like(ddd),create_graph = True)[0]
    return dd[:, -1] - (T/mu)*dd[:, 0] + (ESK2/mu)*(dddd[:, 0]) - f(sample)


def ic_fn_pos(prediction, sample):
    ics = torch.zeros_like(prediction[:, 0])
    return prediction[:, 0], ics

def ic_fn_vel(prediction, sample):
    ics = torch.zeros_like(prediction[:, -1])
    return prediction[:, -1], ics



batchsize = 100000
learning_rate = 1e-3 

print("Building Domain Dataset")
domainDataset = DomainDataset([0.0]*num_inputs,[1.0]*num_inputs, 1000000, period = 3)
print("Building IC Dataset")
icDataset = ICDataset([0.0]*(num_inputs-1),[1.0]*(num_inputs-1), 1000000, period = 3)
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
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

train("main", model, epochs, batchsize, optimizer, pde_fn, [ic_fn_pos, ic_fn_vel], domainDataset, icDataset, validationdatasets = (validationDataset, validationicDataset))
