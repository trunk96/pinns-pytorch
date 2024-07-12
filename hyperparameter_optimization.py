from pinns_v2.model import PINN, Sin, ModifiedMLP
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pinns_v2.train import train
from pinns_v2.gradient import _jacobian, _hessian
from pinns_v2.dataset import DomainDataset, ICDataset

import skopt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
import gc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#epochs = 2000
num_inputs = 2 #x, t

u_min = -0.21
u_max = 0.0
x_min = 0.0
x_max = 1.0
t_f = 10
f_min = -3.0
f_max = 0.0
delta_u = u_max - u_min
delta_x = x_max - x_min
delta_f = f_max - f_min

params = {
    "u_min": u_min,
    "u_max": u_max,
    "x_min": x_min,
    "x_max": x_max,
    "t_f": t_f,
    "f_min": f_min,
    "f_max": f_max
}

def hard_constraint(x, y):
    X = x[0]
    tau = x[-1]
    U = ((X-1)*X*(delta_x**2)*t_f*tau)*(y+(u_min/delta_u)) - (u_min/delta_u)
    return U

def f(sample):
    x = sample[0]*(delta_x) + x_min
    #x_f = sample[1]*(delta_x) + x_min
    x_f = 0.2*(delta_x) + x_min
    #h = sample[2]*(delta_f) + f_min
    h = f_min
    
    z = h * torch.exp(-400*((x-x_f)**2))
    return z


def pde_fn(model, sample):
    T = 1
    mu = 1
    k = 1
    alpha_2 = (T/mu)*(t_f**2)/(delta_x**2)
    beta = (t_f**2)/delta_u
    K = k * t_f
    J, d = _jacobian(model, sample)
    dX = J[0][0]
    dtau = J[0][-1]
    H = _jacobian(d, sample)[0]
    ddX = H[0][0, 0]
    ddtau = H[0][-1, -1]
    return ddtau - alpha_2*ddX - beta*f(sample) + K*dtau


def ic_fn_vel(model, sample):
    J, d = _jacobian(model, sample)
    dtau = J[0][-1]
    dt = dtau*delta_u/t_f
    ics = torch.zeros_like(dt)
    return dt, ics



n_calls = 50
dim_learning_rate = Real(low=1e-4, high=5e-2, name="learning_rate", prior="log-uniform")
dim_num_dense_layers = Integer(low=1, high=10, name="num_dense_layers")
dim_num_dense_nodes = Integer(low=5, high=500, name="num_dense_nodes")
dim_activation = Categorical(categories=[Sin, nn.Sigmoid, nn.Tanh, nn.SiLU], name="activation")
dim_epochs = Integer(low=1, high=2000, name="epochs")
dim_lr_scheduler_epochs = Integer(low=1, high=2000, name="lr_scheduler_epochs")
dim_lr_scheduler_gamma = Real(low=1e-2, high=1.0, name="lr_scheduler_gamma")
dim_eps_time = Real(low = 0.1, high = 1000, name="eps_time", prior = "log-uniform")
dim_period = Integer(low=1, high=5, name="period")
dim_dataset_size = Integer(low=100, high=10000, name="dataset_size")

dimensions = [
    dim_learning_rate,
    dim_num_dense_layers,
    dim_num_dense_nodes,
    dim_activation,
    dim_epochs,
    dim_lr_scheduler_epochs,
    dim_lr_scheduler_gamma,
    dim_eps_time,
    dim_period,
    dim_dataset_size
]

default_parameters = [1e-3, 3, 100, nn.Tanh, 500, 750, 0.1, 100, 3, 1000]
ITERATION = 0

@use_named_args(dimensions = dimensions)
def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation, epochs, lr_scheduler_epochs, lr_scheduler_gamma, eps_time, period, dataset_size):
    global ITERATION
    print(ITERATION, "it number")
    # Print the hyper-parameters.
    print("learning rate: {0:.1e}".format(learning_rate))
    print("num_dense_layers:", num_dense_layers)
    print("num_dense_nodes:", num_dense_nodes)
    print("activation:", activation)
    print("epochs:", epochs)
    print("scheduler_lr_gamma:", lr_scheduler_gamma)
    print("scheduler_lr_epochs:", lr_scheduler_epochs)
    print("dataset size:", dataset_size)
    print("period:", period)
    print("epsilon time causality:", eps_time)
    print()

    batchsize = 256
    domainDataset = DomainDataset([0.0]*num_inputs,[1.0]*num_inputs, dataset_size, batchsize=batchsize, period = period)
    icDataset = ICDataset([0.0]*(num_inputs-1),[1.0]*(num_inputs-1), dataset_size, batchsize = batchsize, period = period)
    validationDataset = DomainDataset([0.0]*num_inputs,[1.0]*num_inputs, batchsize, shuffle = False)
    validationicDataset = ICDataset([0.0]*(num_inputs-1),[1.0]*(num_inputs-1), batchsize, shuffle = False)

    model = PINN([num_inputs] + [num_dense_nodes]*num_dense_layers + [1], activation, hard_constraint, modified_MLP=True)

    def init_normal(m):
        if type(m) == torch.nn.Linear or type(m) == ModifiedMLP:
            torch.nn.init.xavier_uniform_(m.weight)

    model = model.apply(init_normal)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_epochs, gamma=lr_scheduler_gamma)

    data = {
        "name": "string_2inputs_nostiffness_force_damping_ic0hard_icv0_causality_t10.0_rff_0.5",
        #"name": "prova",
        "model": model,
        "epochs": epochs,
        "batchsize": batchsize,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "pde_fn": pde_fn,
        "ic_fns": [ic_fn_vel],
        "eps_time": eps_time,
        "domain_dataset": domainDataset,
        "ic_dataset": icDataset,
        "validation_domain_dataset": validationDataset,
        "validation_ic_dataset": validationicDataset,
        "additional_data": params
    }

    
    min_test_loss = train(data, output_to_file=False)
    del model, optimizer, scheduler, domainDataset, icDataset, validationDataset, validationicDataset
    gc.collect()
    torch.cuda.empty_cache()

    if np.isnan(min_test_loss):
        min_test_loss = 10**5

    ITERATION += 1
    return min_test_loss


search_result = gp_minimize(
    func=fitness,
    dimensions=dimensions,
    acq_func="EI",  # Expected Improvement.
    n_calls=n_calls,
    x0=default_parameters,
    random_state=1234,
)

print(search_result.x)

axes = plot_convergence(search_result)
axes.figure.savefig("plot_convergence.png")
axes = plot_objective(search_result, show_points=True, size=3.8)
axes.figure.savefig("plot_objective.png")



