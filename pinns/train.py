import torch
import numpy as np
from pinns.loss import residual_loss, ic_loss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import json

all_train_losses = []
train_losses = []  # To store losses
test_losses = []




def train(data):  
    name = data.get("name", "main")
    model = data.get("model")
    epochs = data.get("epochs")
    batchsize = data.get("batchsize")
    optimizer = data.get("optimizer")
    scheduler = data.get("scheduler")
    pde_fn = data.get("pde_fn")
    ic_fns = data.get("ic_fns")
    domaindataset = data.get("domain_dataset")
    icdataset = data.get("ic_dataset")
    validationdomaindataset = data.get("validation_domain_dataset")
    validationicdataset = data.get("validation_ic_dataset")
    additional_data = data.get("additional_data")

    current_file = os.getcwd()
    output_dir = os.path.join(current_file, "output", name)
    
    if os.path.exists(output_dir):
        counter = 1
        while True:
            output_dir = os.path.join(current_file, "output", name+"_"+str(counter))
            if not os.path.exists(output_dir):
                break
            else:
                counter +=1
                
    model_dir = os.path.join(output_dir, "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        

    model_path = os.path.join(model_dir, f"model.pt")
    file_path = f"{output_dir}/train.txt"

    params_path = f"{output_dir}/params.json"
    params = {
        "name": name,
        "model": str(model),
        "epochs": epochs,
        "batchsize": batchsize,
        "optimizer": str(optimizer),
        "scheduler": str(scheduler.state_dict()) if scheduler!=None else "None",
        "domainDataset": str(domaindataset),
        "icDataset": str(icdataset),
        "validationDomainDataset": str(validationdomaindataset),
        "validationICDataset": str(validationicdataset)
    } 
    if additional_data != None:
        params["additionalData"] = additional_data
    fp = open(params_path, "w", newline='\r\n') 
    json.dump(params, fp)
    fp.close()  

    dataloader = DataLoader(domaindataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = False)
    ic_dataloader = DataLoader(icdataset, batch_size=batchsize, shuffle=True, num_workers = 0, drop_last = False)
    if validationicdataset != None and validationdomaindataset != None:
        validation_dataloader = DataLoader(validationdomaindataset, batch_size=batchsize, shuffle=False, num_workers = 0, drop_last = False)
        validation_ic_dataloader = DataLoader(validationicdataset, batch_size=batchsize, shuffle=False, num_workers = 0, drop_last = False)

    residual_losses = []
    ic_losses = [[] for i in range(len(ic_fns))]

    # Open the log file for writing
    with open(file_path, "w") as log_file:
        for epoch in range(epochs):
            model.train(True)
            l = []
            for batch_idx, (x_in) in enumerate(dataloader):          
                (x_ic) = next(iter(ic_dataloader))
                #print(f"{x_in}, {x_ic}")
                x_in = torch.Tensor(x_in).to(torch.device('cuda:0'))
                x_ic = torch.Tensor(x_ic).to(torch.device('cuda:0'))
                loss_eqn = residual_loss(x_in, model, pde_fn)
                loss = loss_eqn
                residual_losses.append(loss_eqn.item())
                for i in range(len(ic_fns)):
                    loss_ic = ic_loss(x_ic, model, ic_fns[i])
                    loss += loss_ic
                    ic_losses[i].append(loss_ic.item())
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
                    
                all_train_losses.append(loss.item())  # Storing the loss
                l.append(loss.item())

            torch.cuda.empty_cache()
            if validationicdataset != None and validationdomaindataset != None:
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
                epoch_path = os.path.join(model_dir, f"model_{epoch}.pt")
                torch.save(model, epoch_path)
            
            if scheduler != None:
                scheduler.step()
            train_losses.append(np.average(l))
            torch.cuda.empty_cache()
    
    # Save the model
    torch.save(model, model_path)
    
    plt.plot(all_train_losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(f'{output_dir}/training_loss.png')
    plt.clf()
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.savefig(f'{output_dir}/test_loss.png')
    plt.clf()
    label = ["Residual loss"]
    plt.plot(residual_losses)
    for i in range(len(ic_fns)):
        plt.plot(ic_losses[i])
        label.append("IC_loss_"+str(i))
    plt.legend(label)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.savefig(f'{output_dir}/train_losses.png')
    plt.show()
