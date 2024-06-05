import torch
import torch.nn as nn
import numpy as np
from torch.func import vmap
from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def residual_loss(model, pde_fn, x, t):
    x_in = torch.hstack((x, t))
    r = pde_fn(model, x_in)
    return r

def ic_loss(model, ic_fn, x, t):
    x_in = torch.hstack((x, t))
    u, true = ic_fn(model, x_in)
    loss_ic = torch.mean((u.flatten() - true.flatten())**2)
    return loss_ic

def compute_loss_ic(model, ic_fns, x_ic):
    splitted_dataset = torch.hsplit(x_ic, [x_ic.shape[1] - 1])
    x = splitted_dataset[0]
    t = splitted_dataset[1]
    loss_ic = None
    for i in range(len(ic_fns)):
        loss_ic_i = vmap(partial(ic_loss, model, ic_fns[i]), (0, 0))(x, t)
        if i == 0:
            loss_ic = loss_ic_i
        else:
            loss_ic += loss_ic_i
    return loss_ic

def compute_loss_r(model, pde_fn, eps_time, x_in):
    splitted_dataset = torch.hsplit(x_in, [x_in.shape[1] - 1])
    x = splitted_dataset[0]
    t = splitted_dataset[1]
    r_pred = vmap(vmap(partial(residual_loss, model, pde_fn), (0, None)), (None, 0)) (x, t)
    pde_loss_t = torch.mean(r_pred **2, axis = 1)
    with torch.no_grad():
        M = np.triu(np.ones((x_in.shape[0], x_in.shape[0])), k=1).T
        M = torch.Tensor(M).to(device)
        W = torch.exp(- eps_time * (M @ pde_loss_t))
    return W, pde_loss_t


def loss(model, pde_fn, ic_fns, eps_time, x_in, x_ic):
    loss_ic = compute_loss_ic(model, ic_fns, x_ic)
    W, pde_loss_t = compute_loss_r(model, pde_fn, eps_time, x_in)
    loss = torch.mean(W*pde_loss_t + loss_ic)
    return loss
