import torch
import numpy as np
from loss import residual_loss, ic_loss
from torch.utils.data import DataLoader


def train(model, epochs, batchsize, optimizer, pde_fn, ic_fns, domaindataset, icdataset):
    dataloader = DataLoader(domaindataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = False)
    ic_dataloader = DataLoader(icdataset, batch_size=batchsize, shuffle=True, num_workers = 0, drop_last = False)

    for epoch in range(epochs):
        for batch_idx, (x_in) in enumerate(dataloader):          
            (x_ic) = next(iter(ic_dataloader))
            model.zero_grad()
            x_in = torch.Tensor(x_in).to(torch.device('cuda:0'))
            x_ic = torch.Tensor(x_ic).to(torch.device('cuda:0'))
            loss_eqn = residual_loss(x_in, model, pde_fn)
            loss = loss_eqn
            for i in range(len(ic_fns)):
                loss_ic = ic_loss(x_ic, model, ic_fns[i])
                loss += loss_ic
            loss.requires_grad = True
            loss.backward()
    
            optimizer.step() 
            if batch_idx % 10 ==0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                    epoch, batch_idx, int(len(dataloader.dataset)/batchsize),
                    100. * batch_idx / len(dataloader), loss.item()))