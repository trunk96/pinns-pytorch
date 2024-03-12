import torch
import numpy as np
from loss import residual_loss, ic_loss
from dataset import DomainDataset, ICDataset
from torch.utils.data import DataLoader


def train(model, epochs, batchsize, optimizer, pde_fn, ic_fn):
    dataloader = DataLoader(DomainDataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = False)
    ic_dataloader = DataLoader(ICDataset, batch_size=batchsize, shuffle=True, num_workers = 0, drop_last = False)

    for epoch in range(epochs):
        for batch_idx, (x_in) in enumerate(dataloader):
            
            batch_idx, (x_ic) = next(iter(ic_dataloader))
            model.zero_grad()
            loss_eqn = residual_loss(x_in, model, pde_fn)
            loss_ic = ic_loss(x_ic, model, ic_fn)
            loss = loss_eqn + loss_ic
            loss.backward()
    
            optimizer.step() 
            if batch_idx % 10 ==0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                    epoch, batch_idx, int(len(dataloader.dataset)/batchsize),
                    100. * batch_idx / len(dataloader), loss.item()))