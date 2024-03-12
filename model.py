import torch
import torch.nn as nn
import numpy as np

class PINN(nn.Module):

    def __init__(self, layers, activation_function, hard_constraint_fn = None):
        super().__init__()
        self.net = []
        for i in range(len(layers)-1):
            self.net.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers) - 1:
                self.net.append(activation_function)
        self.hard_constraint_fn = hard_constraint_fn
		
    def forward(self, x):
        output = x
        for el in self.net:
            output = el(output)
        if self.hard_constraint_fn != None:
            output = self.hard_constraint_fn(x, output)
        return output
	
					