import torch
import numpy as np
from pinns.loss import residual_loss, ic_loss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

train_losses = []  # To store losses
test_losses = []
current_file = os.getcwd()
output_dir = os.path.join(current_file, "output")

model_dir = os.path.join(output_dir, "model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)



def train(name, model, epochs, batchsize, optimizer, pde_fn, ic_fns, domaindataset, icdataset, validationdatasets = None):
    model_path = os.path.join(model_dir, f"{name}.pt")
    file_path = f"{output_dir}/train_{name}.txt"
    dataloader = DataLoader(domaindataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = False)
    ic_dataloader = DataLoader(icdataset, batch_size=batchsize, shuffle=True, num_workers = 0, drop_last = False)
    if validationdatasets != None and len(validationdatasets) == 2:
        validation_dataloader = DataLoader(validationdatasets[0], batch_size=batchsize, shuffle=False, num_workers = 0, drop_last = False)
        validation_ic_dataloader = DataLoader(validationdatasets[1], batch_size=batchsize, shuffle=False, num_workers = 0, drop_last = False)
    # Open the log file for writing
    with open(file_path, "w") as log_file:
        for epoch in range(epochs):
            model.train(True)
            for batch_idx, (x_in) in enumerate(dataloader):          
                (x_ic) = next(iter(ic_dataloader))
                #print(f"{x_in}, {x_ic}")
                x_in = torch.Tensor(x_in).to(torch.device('cuda:0'))
                x_ic = torch.Tensor(x_ic).to(torch.device('cuda:0'))
                loss_eqn = residual_loss(x_in, model, pde_fn)
                loss = loss_eqn
                for i in range(len(ic_fns)):
                    loss_ic = ic_loss(x_ic, model, ic_fns[i])
                    loss += loss_ic
                #loss.requires_grad = True
                optimizer.zero_grad()
                loss.backward()
        
                optimizer.step() 
                if batch_idx % 10 ==0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                        epoch, batch_idx, int(len(dataloader.dataset)/batchsize),
                        100. * batch_idx / len(dataloader), loss.item()))
                    
                    # Save to log file
                    log_file.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}\n'.format(
                        epoch, batch_idx, int(len(dataloader.dataset)/batchsize),
                        100. * batch_idx / len(dataloader), loss.item()))
                    
                train_losses.append(loss.item())  # Storing the loss
            
            if validationdatasets != None and len(validationdatasets) == 2:
                model.eval()
                validation_losses = []
                for batch_idx, (x_in) in enumerate(validation_dataloader):
                    (x_ic) = next(iter(validation_ic_dataloader))
                    x_in = torch.Tensor(x_in).to(torch.device('cuda:0'))
                    x_ic = torch.Tensor(x_ic).to(torch.device('cuda:0'))
                    loss_eqn = residual_loss(x_in, model, pde_fn)
                    loss = loss_eqn
                    for i in range(len(ic_fns)):
                        loss_ic = ic_loss(x_ic, model, ic_fns[i])
                        loss += loss_ic
                    validation_losses.append(loss.item())
                print('Validation Epoch: {} \tLoss: {:.10f}'.format(
                        epoch, np.average(validation_losses)))
                log_file.write('Validation Epoch: {} \tLoss: {:.10f}'.format(epoch, np.average(validation_losses)))
                test_losses.append(np.average(validation_losses))
                    
            if epoch % 20 == 0:
                epoch_path = os.path.join(model_dir, f"{name}_{epoch}.pt")
                torch.save(model, epoch_path)
                
    # Save the model
    torch.save(model, model_path)
    
    plt.plot(train_losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(f'{output_dir}/training_loss_{name}.png')
    plt.plot(test_losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(f'{output_dir}/test_loss_{name}.png')
    plt.show()
