from model import PINN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from train import train
from dataset import DomainDataset, ICDataset


#components: [x, ic_p_0, ic_p_1, ic_p_2, ic_v_0, ic_v_1, ic_v_2, t]

def hard_constraint(x, y):
    return x[:, 0:1] * (1 - x[:, 0:1]) * y


def pde_fn(prediction, sample):
    dudt = torch.autograd.grad(prediction, sample, create_graph=True, retain_graph=True)[0][0][-1]
    dudx = torch.autograd.grad(prediction, sample, create_graph=True, retain_graph=True)[0][0][0]
    dduddt = torch.autograd.grad(dudt, sample, retain_graph=True)[0][0][-1].reshape((1, 1))
    dduddx = torch.autograd.grad(dudx, sample, retain_graph=True)[0][0][0].reshape((1, 1))
    a = 1
    return dduddt - a*dduddx

def interpolate(x, points, ic_points):
    selected_points = []
    for i in range(len(points)-1):
        if points[i] <= x and points[i+1] >= x:
            selected_points = [i, i+1]
    u_ic= (x - points[selected_points[0]])*(ic_points[selected_points[1]]-ic_points[selected_points[0]])/(points[selected_points[1]] - points[selected_points[0]]) + ic_points[selected_points[0]]
    return u_ic

def ic_fn_pos(prediction, sample):
    x = sample[0][0]
    points = [0, 0.25, 0.5, 0.75, 1]
    ic_points = [0, sample[0][1], sample[0][2], sample[0][3], 0]
    ic = interpolate(x, points, ic_points)
    ic = torch.Tensor(ic).to(device=prediction.device).reshape(prediction.shape)
    return prediction, ic

def ic_fn_vel(prediction, sample):
    x = sample[0][0]
    points = [0, 0.25, 0.5, 0.75, 1]
    ic_points = [0, sample[0][4], sample[0][5], sample[0][6], 0]
    dudt = torch.autograd.grad(prediction, sample, create_graph=True, retain_graph=True)[0][0][-1]
    dudt = dudt.reshape((1,1))
    ic = interpolate(x, points, ic_points)
    ic = torch.Tensor(ic).to(device=dudt.device).reshape(dudt.shape)
    return dudt, ic



batchsize = 32
learning_rate = 1e-3 

domainDataset = DomainDataset([0.0]*8, [1.0]*8, 1000)
icDataset = ICDataset([0.0]*8, [1.0]*8, 1000)

model = PINN([8] + [100]*3 + [1], nn.ReLU, hard_constraint).to(torch.device('cuda:0'))
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)

train(model, 10, batchsize, optimizer, pde_fn, [ic_fn_pos, ic_fn_vel], domainDataset, icDataset)
