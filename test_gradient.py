from pinns_v2.model import MLP, ModifiedMLP
from pinns_v2.components import ComponentManager, ResidualComponent, ICComponent, SupervisedComponent
from pinns_v2.rff import GaussianEncoding 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#from pinns.train import train
from pinns_v2.train import train
from pinns_v2.gradient import _jacobian, _hessian
from pinns_v2.dataset import DomainDatasetRandom, ICDatasetRandom, DomainSupervisedDataset

from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_inputs = 3 #delta_0, omega_0, t

model = MLP([num_inputs] + [2], nn.Identity, None, p_dropout=0.0)
model = model.to(device=device)


x_in = np.ones((4, 3))
#x_in = np.random.randn(2, 3)
x_in = torch.Tensor(x_in).requires_grad_(). to(device=device)
print(x_in)
print(model(x_in))
print("w", model.mlp[0].weight)
print("b", model.mlp[0].bias)


def calcolo_jacobian(model, sample):
    j_val, j = _jacobian(model, sample)
    dt, ddt = j_val[:, -1]
    print(j_val)
    print(dt, ddt)
    return j_val

j_val = torch.func.vmap(partial(calcolo_jacobian, model), (0), randomness="different")(x_in)
