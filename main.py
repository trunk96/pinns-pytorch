from model import PINN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def hard_constraint(x, y):
    return x[:, 0:1] * (1 - x[:, 0:1]) * y

def pde_fn(out, input):
    """ t = torch.zeros_like(input)
    t[-1] = 1
    x = torch.zeros_like(input)
    x[0] = 1 """
    dx = torch.autograd.grad(out, input, grad_outputs=torch.ones_like(input), create_graph = True)
    ddx = torch.autograd.grad(dx, input, grad_outputs=torch.ones_like(input), create_graph = True)

    a = 1
    return ddx[-1] - a*ddx[0]

def ic_fn(out, input):
    return


batchsize = 32
learning_rate = 1e-3 

model = PINN([22] + [100]*3 + [1], nn.ReLU, hard_constraint)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
