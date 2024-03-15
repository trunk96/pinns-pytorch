import torch
import torch.nn as nn
import numpy as np



def compute_residual_sample_loss(model, sample, loss_fn, pde_fn):
    sample = sample.unsqueeze(0)  # prepend batch dimension for processing
    sample.requires_grad = True
    prediction = model(sample)
    res = pde_fn(prediction, sample)
    zero = torch.zeros_like(res)
    loss = loss_fn(res, zero)
    return loss

def compute_ic_sample_loss(model, sample, loss_fn, ic_fn):
    sample = sample.unsqueeze(0)
    sample.requires_grad = True
    prediction = model(sample)
    res, out = ic_fn(prediction, sample)
    loss = loss_fn(out, res)
    return loss

def compute_residual_loss(model, data, loss_fn, pde_fn):
    sample_loss = [compute_residual_sample_loss(model, data[i], loss_fn, pde_fn) for i in range(len(data))]
    loss = torch.sum(torch.Tensor(sample_loss))
    return loss  

def compute_ic_loss(model, data, loss_fn, ic_fn):
    sample_loss = [compute_ic_sample_loss(model, data[i], loss_fn, ic_fn) for i in range(len(data))] 
    loss = torch.sum(torch.Tensor(sample_loss))
    return loss

def residual_loss(data, model, pde_fn):
    #data = torch.Tensor(data).to(torch.device('cuda:0'))
    loss = compute_residual_loss(model, data, nn.MSELoss(), pde_fn)
    return loss

def ic_loss(data, model, ic_fn):
    #data = torch.Tensor(data).to(torch.device("cuda:0"))
    loss = compute_ic_loss(model, data, nn.MSELoss(), ic_fn)
    return loss
