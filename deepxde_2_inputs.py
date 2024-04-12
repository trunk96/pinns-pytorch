import deepxde as dde
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.integrate import quad
from matplotlib.animation import FuncAnimation
import torch



epochs = 20000
name = "pinns_2_inputs"
current_file = os.path.abspath(__file__)
output_dir = os.path.join(os.path.dirname(current_file), name)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)



def func(x):

    return w1(x[:, 0])


def func2(x):

    return w2(x[:, 0])


def w1(x):


    return 0

def w2(x):

    return 0



def force(sample):
    x = sample[:, 0]
    # y = sample[:, 1]
    x_f = 0.8
    # y_f = 0.8
    height = 1
    t = sample[:, -1]

    alpha = 8.9
    za = -height * torch.exp(-400*((x-x_f)**2)) * (4**alpha * t**(alpha - 1) * (1 - t)**(alpha - 1))
    return za


def pde(x, z):
    T = 1
    mu = 1
    ESK2 = 3.926790540455574e-06

    dz_tt = dde.grad.hessian(z, x, i=1, j=1)
    dz_xx = dde.grad.hessian(z, x, i=0, j=0)

    dz_xxxx = dde.grad.hessian(dz_xx, x, i=0, j=0)
    return dz_tt - (T/mu)*dz_xx + (ESK2/mu) * (dz_xxxx) - force(x)





def create_model():

    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    ic_1 = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

    ic_2 = dde.icbc.OperatorBC(
        geomtime,
        lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1) - func2(x),
        lambda x, _: np.isclose(x[1], 0),
    )
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [ic_1, ic_2],
        num_domain=1000,
        # num_boundary=360,
        num_initial=200,
        num_test=1024,
    )

    layer_size = [2] + [100] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

    net.apply_output_transform(lambda x, y: x[:, 0:1] * (1 - x[:, 0:1]) * y)

    model = dde.Model(data, net)

    model.compile(
        "adam",
        lr=0.001
    )
    return model

def train(mm):

    pde_residual_resampler = dde.callbacks.PDEPointResampler(period=100)
    # early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-6, patience=5000)
    losshistory, train_state = mm.train(
        iterations=epochs, callbacks=[pde_residual_resampler], display_every=500, model_save_path=f"{output_dir}/pinns_{name}"
    )
    dde.saveplot(losshistory, train_state, output_dir=f"{output_dir}")



    # Convert the list of arrays to a 2D NumPy array
    matrix = np.array(losshistory.loss_train)

    # Separate the components into different arrays
    loss_res = matrix[:, 0]
    # loss_bcs = matrix[:, 1]
    loss_u_t_ics = matrix[:, 1]
    loss_du_t_ics = matrix[:, 2]

    l2_error = np.array(losshistory.metrics_test)

    fig = plt.figure(figsize=(6, 5))
    iters = 500 * np.arange(len(loss_res))
    with sns.axes_style("darkgrid"):
        plt.plot(iters, loss_res, label='$\mathcal{L}_{r}$')
        # plt.plot(iters, loss_bcs, label='$\mathcal{L}_{u}$')
        plt.plot(iters, loss_u_t_ics, label='$\mathcal{L}_{u_0}$')
        plt.plot(iters, loss_du_t_ics, label='$\mathcal{L}_{u_t}$')
        plt.plot(iters, l2_error, label='$\mathcal{L}^2 error$')
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dde2.png")
        plt.show()

    # plot_animation(model)
    return mm


def restore_model(model, name):

    model.restore(f"{output_dir}/membrane_{name}-{epochs}.ckpt", verbose=0)
    return model


saved_model = None
def load_model(path):
    global saved_model
    if saved_model == None:
        saved_model = create_model()
        saved_model.restore(path, verbose = 1)
        return True
    else:
        return False



if __name__ == "__main__":
    a = create_model()
    b = train(a)
    fig, axes = plt.subplots(1, 1, figsize=(15, 5))
    x = np.linspace(0, 1, num=100)
    t = np.linspace(0, 1, num=100)
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T

    Xp = torch.Tensor(X)#.to(torch.device('cuda:0')).requires_grad_()
    ppred = b.predict(Xp)

    #ppred = ppred[:, 0].cpu().detach().numpy()

    la = len(np.unique(X[:, 0:1]))
    le = len(np.unique(X[:, 1:]))

    #pred = ppred.reshape((le, la)).cpu()
    pred = ppred.reshape((le, la))


    # Plot Theta Predicted
    im1 = axes.imshow(pred, cmap='inferno', aspect='auto', origin='lower',
                            extent=[np.unique(X[:, 0:1]).min(), np.unique(X[:, 0:1]).max(), np.unique(X[:, 1:]).min(), np.unique(X[:, 1:]).max()])#, vmin=true.min(), vmax = true.max())
    axes.set_title(f'Predicted')
    axes.set_xlabel('X')
    axes.set_ylabel('T')
    plt.colorbar(im1, ax=axes)


    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'{output_dir}/plot_{name}.png')

    plt.show()