import torch
import torch.nn as nn
import numpy as np



def compute_residual_loss(model, samples, loss_fn, pde_fn):
    #print(samples.shape)
    samples.requires_grad = True
    predictions = model(samples)
    #print(predictions.shape)
    res = pde_fn(predictions, samples)
    #print(res)
    zero = torch.zeros_like(res)
    loss = loss_fn(res, zero)
    return loss

def compute_ic_loss(model, samples, loss_fn, ic_fn):
    samples.requires_grad = True
    predictions = model(samples)
    res, out = ic_fn(predictions, samples)
    loss = loss_fn(out, res)
    return loss


""" def compute_ic_loss(model, data, loss_fn, ic_fn):
    sample_loss = [compute_ic_sample_loss(model, data[i], loss_fn, ic_fn) for i in range(len(data))] 
    loss = torch.sum(torch.Tensor(sample_loss))
    return loss """

def residual_loss(data, model, pde_fn):
    #data = torch.Tensor(data).to(torch.device('cuda:0'))
    loss = compute_residual_loss(model, data, nn.MSELoss(), pde_fn)
    return loss

def ic_loss(data, model, ic_fn):
    #data = torch.Tensor(data).to(torch.device("cuda:0"))
    loss = compute_ic_loss(model, data, nn.MSELoss(), ic_fn)
    return loss
