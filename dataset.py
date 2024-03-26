import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import itertools


class DomainDataset(Dataset):
    def __init__(self, xmin, xmax, n, rand = True):
        self.xmin = np.array(xmin, dtype="f")
        self.xmax = np.array(xmax, dtype="f")
        self.dim = len(xmin)
        self.n = n
        self.side_length = self.xmax - self.xmin
        self.volume = np.prod(self.side_length)
        self.rand = rand
        if self.rand:
            self.counter = 0
            self.compute_items_rand()
        else:
            self.compute_items_sequential()    

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        ret = self.x[idx]
        if self.rand:
            self.counter += 1
            if self.counter == self.__len__():
                self.compute_items_rand()
                self.counter = 0
        return ret
    
    def compute_items_sequential(self):
        n_points_per_axis = np.ceil(self.n ** (1/self.dim))
        xi = []
        for i in range(self.dim):
            s = np.linspace(self.xmin[i], self.xmax[i], num = n_points_per_axis + 1, endpoint=False)[1:]
            xi.append(s)
        self.x = np.array(list(itertools.product(*xi)), dtype = "f")
        return

    def compute_items_rand(self):
        n_points_per_axis = np.ceil(self.n ** (1/self.dim))
        xi = []
        for i in range(self.dim):
            s = np.random.uniform(low=self.xmin[i], high=self.xmax[i], size=(n_points_per_axis, ))
            xi.append(s)
        self.x = np.array(list(itertools.product(*xi)), dtype = "f")

    def _compute_items(self):    
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
    def __init__(self, xmin, xmax, n, rand=True):
        super().__init__(xmin, xmax, n, rand=rand)
        
    def compute_items_sequential(self):
        n_points_per_axis = np.ceil(self.n ** (1/self.dim))
        xi = []
        for i in range(self.dim):
            s = np.linspace(self.xmin[i], self.xmax[i], num = n_points_per_axis + 1, endpoint=False)[1:]
            xi.append(s)
        xi.append([0.0]*n_points_per_axis)
        self.x = np.array(list(itertools.product(*xi)), dtype = "f")
        return

    def compute_items_rand(self):
        n_points_per_axis = np.ceil(self.n ** (1/self.dim))
        xi = []
        for i in range(self.dim):
            s = np.random.uniform(low=self.xmin[i], high=self.xmax[i], size=(n_points_per_axis, ))
            xi.append(s)
        xi.append([0.0]*n_points_per_axis)
        self.x = np.array(list(itertools.product(*xi)), dtype = "f")
    
    def _compute_items(self):
        dx = (self.volume / self.n) ** (1 / self.dim)
        xi = []
        for i in range(self.dim):
            ni = int(np.ceil(self.side_length[i] / dx))
            s = np.linspace(self.xmin[i],self.xmax[i],num=ni + 1,endpoint=False)[1:]
            xi.append(s)
        ni = int(np.ceil(self.side_length[0] / dx))
        xi.append([0.0]*ni)          

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