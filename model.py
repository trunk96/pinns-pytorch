import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

class PINN(nn.Module):

    def __init__(self, layers, activation_function, hard_constraint_fn = None):
        super(PINN, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        
        # set up layer order dict
        self.activation = activation_function
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        self.hard_constraint_fn = hard_constraint_fn
		
    def forward(self, x):
        output = self.layers(x)
        if self.hard_constraint_fn != None:
            output = self.hard_constraint_fn(x, output)
        return output
	
					