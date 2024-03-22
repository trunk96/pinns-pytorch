import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import itertools


class DomainDataset(Dataset):
    def __init__(self, xmin, xmax, n):
        self.xmin = np.array(xmin, dtype="f")
        self.xmax = np.array(xmax, dtype="f")
        self.dim = len(xmin)
        self.n = n
        self.side_length = self.xmax- self.xmin
        self.volume = np.prod(self.side_length)
        self.compute_items()

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx]
        
    def compute_items(self):    
        dx = (self.volume / self.n) ** (1 / self.dim)
        xi = []
        for i in range(self.dim):
            ni = int(np.ceil(self.side_length[i] / dx))
            s = np.linspace(self.xmin[i],self.xmax[i],num=ni + 1,endpoint=False)[1:]
            xi.append(s)
        x = np.array(list(itertools.product(*xi)), dtype="f")
        #x = np.array(xi)
        """ if self.n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(self.n, len(x))
            ) """
        self.x = x
    

class ICDataset(DomainDataset):
    def __init__(self, xmin, xmax, n):
        super().__init__(xmin, xmax, n)
    
    def __getitem__(self, idx):
        return self.x[idx]
    
    def compute_items(self):
        dx = (self.volume / self.n) ** (1 / self.dim)
        xi = []
        for i in range(self.dim):
            if i == self.dim - 1:
                xi.append([0]*ni)
            else:
                ni = int(np.ceil(self.side_length[i] / dx))
                s = np.linspace(self.xmin[i],self.xmax[i],num=ni + 1,endpoint=False)[1:]
                xi.append(s)
            

        x = np.array(list(itertools.product(*xi)), dtype="f")
        #if self.n != len(x):
            #print(
            #    "Warning: {} points required, but {} points sampled.".format(self.n, len(x))
            #)
        self.x = x

   
class BCDataset(DomainDataset):
    def __init__(self, xmin, xmax, n):
        super().__init__(xmin, xmax, n)
    
    #TO BE IMPLEMENTED