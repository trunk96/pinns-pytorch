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

def pde_fn(out, input):
    """ t = torch.zeros_like(input)
    t[-1] = 1
    x = torch.zeros_like(input)
    x[0] = 1 """
    a = 1
    res = []
    for i in range(len(input)):
        print(out[i])
        print(input[i])
        dx = torch.autograd.grad(out[i], input[i], grad_outputs=torch.ones_like(input[i]), create_graph = True)
        ddx = torch.autograd.grad(dx, input[i], grad_outputs=torch.ones_like(input[i]), create_graph = True)
        res.append(ddx[-1] - a*ddx[0])
    return res

def interpolate(x, points, ic_points):
    selected_points = []
    u_ic = []
    for j in range(len(x)):
        for i in range(len(points)-1):
            if points[i] < x[j] and points[i+1] > x:
                selected_points = [i, i+1]
        u_ic.append((x[j] - points[selected_points[0]])*(ic_points[selected_points[1]]-ic_points[selected_points[0]])/(points[selected_points[1]] - points[selected_points[0]]) + ic_points[selected_points[0]])
    return u_ic

def ic_fn_pos(out, input):
    x = input[:, 0]

    points = [0, 0.25, 0.5, 0.75, 1]
    ic_points = [0, input[:, 1], input[:, 2], input[:, 3], 0]
    
    return [out, interpolate(x, points, ic_points)]

def ic_fn_vel(out, input):
    x = input[:, 0]
    points = [0, 0.25, 0.5, 0.75, 1]
    ic_points = [0, input[:, 4], input[:, 5], input[:, 6], 0]
    vel = []
    for i in range(len(input)):
        vel.append(torch.autograd.grad(out[i], input[i], grad_outputs=torch.ones_like(input[i]), create_graph = True)[-1])
    return [vel, interpolate(x, points, ic_points)]



batchsize = 32
learning_rate = 1e-3 

domainDataset = DomainDataset([0.0]*8, [1.0]*8, 100)
icDataset = ICDataset([0.0]*8, [1.0]*8, 100)

model = PINN([8] + [100]*3 + [1], nn.ReLU, hard_constraint).to(torch.device('cuda:0'))
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)

train(model, 1, 1, optimizer, pde_fn, [ic_fn_pos, ic_fn_vel], domainDataset, icDataset)
