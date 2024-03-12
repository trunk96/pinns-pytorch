import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np


class DomainDataset(Dataset):
    def __init__(self, xmin, xmax, n):
        self.xmin = np.array(xmin, dtype = np.real)
        self.xmax = np.array(xmax, dtype = np.real)
        self.dim = len(xmin)
        self.n = n
        self.side_length = self.xmin - self.xmax
        self.volume = np.prod(self.side_length)

    def __len__(self):
        return self.n
    
    def __get_item__(self, idx):
        dx = (self.volume / self.n) ** (1 / self.dim)
        xi = []
        for i in range(self.dim):
            ni = int(np.ceil(self.side_length[i] / dx))
            xi.append(np.linspace(self.xmin[i],self.xmax[i],num=ni + 1,endpoint=False,dtype=np.real)[1:][idx])

        x = np.array(xi)
        if self.n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(self.n, len(x))
            )
        return x
    

class ICDataset(DomainDataset):
    def __init__(self, xmin, xmax, n):
        super().__init__(xmin, xmax, n)
    
    def __get_item__(self, idx):
        dx = (self.volume / self.n) ** (1 / self.dim)
        xi = []
        for i in range(self.dim):
            if i != self.dim - 1:
                ni = int(np.ceil(self.side_length[i] / dx))
                xi.append(np.linspace(self.xmin[i],self.xmax[i],num=ni + 1,endpoint=False,dtype=np.real)[1:][idx])
            else:
                xi.append(0)

        x = np.array(xi)
        if self.n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(self.n, len(x))
            )
        return x

   
class BCDataset(DomainDataset):
    def __init__(self, xmin, xmax, n):
        super().__init__(xmin, xmax, n)
    
    #TO BE IMPLEMENTED