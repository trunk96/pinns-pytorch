import torch
import torch.nn as nn
import numpy as np


def residual_loss(x, model, pde_fn):

    net_in = torch.Tensor(x)
    net_in.requires_grad = True

    out = model(net_in)
    out = out.view(len(out), -1)

    l = pde_fn(out, net_in) 
    l_f = nn.MSELoss()
    return l_f(l, torch.zeros_like(l))


def ic_loss(x, model, ic_fn):

    net_in = torch.Tensor(x)
    net_in.requires_grad = True

    out = model(net_in)
    out = out.view(len(out), -1)

    [out, ic] = ic_fn(out, net_in)
    l_f = nn.MSELoss()
    return l_f(out, ic)
