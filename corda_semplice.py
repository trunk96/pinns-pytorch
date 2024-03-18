from model import PINN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from train import train
from dataset import DomainDataset, ICDataset


#components: [x, ic_p_0, ic_p_1, ic_p_2, ic_v_0, ic_v_1, ic_v_2, t]

def hard_constraint(x, y):
    res = x[:, 0:1] * (1 - x[:, 0:1]) * y
    return res


def pde_fn(prediction, sample):
    #dudt = torch.autograd.grad(prediction, sample, create_graph=True, retain_graph=True)[0][0][-1]
    #dudx = torch.autograd.grad(prediction, sample, create_graph=True, retain_graph=True)[0][0][0]
    #dduddt = torch.autograd.grad(dudt, sample, retain_graph=True)[0][0][-1].reshape((1, 1))
    #dduddx = torch.autograd.grad(dudx, sample, retain_graph=True)[0][0][0].reshape((1, 1))
    d = torch.autograd.grad(prediction, sample, grad_outputs=torch.ones_like(prediction),create_graph = True,only_inputs=True)[0]
    dd = torch.autograd.grad(d, sample, grad_outputs=torch.ones_like(d),create_graph = True,only_inputs=True)[0]
    a = 1
    return dd[:, -1] - a*dd[:, 0]

def ic_fn_pos(prediction, sample):
    #print(sample)
    x = sample[:, 0]
    ic = torch.sin(x/torch.pi)
    ic = torch.Tensor(ic).to(device=prediction.device).reshape(prediction.shape)
    return prediction, ic

def ic_fn_vel(prediction, sample):
    d = torch.autograd.grad(prediction, sample, grad_outputs=torch.ones_like(prediction),create_graph = True,only_inputs=True)[0]
    ic = [0.0]*sample.shape[0]
    ic = torch.Tensor(ic).to(device=sample.device)
    return d[:, -1], ic



batchsize = 32
learning_rate = 1e-3 

domainDataset = DomainDataset([0.0]*2, [1.0]*2, 1000)
icDataset = ICDataset([0.0]*2, [1.0]*2, 1000)

model = PINN([2] + [100]*3 + [1], nn.Tanh, hard_constraint).to(torch.device('cuda:0'))

def init_normal(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)

model.apply(init_normal)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

train(model, 100, batchsize, optimizer, pde_fn, [ic_fn_pos, ic_fn_vel], domainDataset, icDataset)
