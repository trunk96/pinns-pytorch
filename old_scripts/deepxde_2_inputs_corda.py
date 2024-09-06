import deepxde as dde
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pathlib
from scipy.integrate import quad
from matplotlib.animation import FuncAnimation
import torch
# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()

# tf.reset_default_graph()


T = 1
mu = 1
ESK2 = 3.926790540455574e-06
sig = 10



def func(x):
    # x, t = np.split(x, 2, axis=1)

    return w1(x[:, 0])

def func2(z):
    # x, t = torch.split(z, 2, axis=1)

    return w2(z[:, 0])


def w1(x):
    # sigma = 0.05
    # return np.exp(-(x - 0.23)**2 / (2 * sigma**2))
    # condition = np.less_equal(x, 0.2)
    # return np.where(condition, 5 * x, 1.25 * (1 - x))
    return 0


def w2(x):

    # return 0
    return 0

def f(sample):
    x = sample[:, 0]

    x_f = 0.8
    #y_f = 0.8
    height = 1
    t = sample[:, -1]

    alpha = 8.9
    za = -height * torch.exp(-400*((x-x_f)**2)) * (4**alpha * t**(alpha - 1) * (1 - t)**(alpha - 1))
    return za



def go_sc(epochs):


    print(f"Start training iter={epochs}")

    name = f"deepxde_corda"
    script_directory = pathlib.Path.cwd() / f"{name}"
    if not script_directory.exists():
        os.makedirs(script_directory, exist_ok=True)

    def pde(x, y):
        dy_tt = dde.grad.hessian(y, x, i=1, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        dy_xxxx = dde.grad.hessian(dy_xx, x, i=0, j=0)
        return dy_tt - (T/mu) * dy_xx  + (ESK2/mu) * dy_xxxx - f(x)


    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
    ic_1 = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)
    # do not use dde.NeumannBC here, since `normal_derivative` does not work with temporal coordinate.
    ic_2 = dde.icbc.OperatorBC(
        geomtime,
        lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1) - func2(x),
        lambda x, _: np.isclose(x[1], 0),
    )
    data = dde.data.TimePDE(
        geomtime,
        pde,
        # [bc, ic_1, ic_2],
        [ic_1, ic_2],
        # [ic_2],
        num_domain=1440,
        num_boundary=360,
        num_initial=360,
        # solution=summative,
        num_test=10000,
    )

    layer_size = [2] + [100] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    # net = dde.nn.STMsFFN(
    #     layer_size, activation, initializer, sigmas_x=[1], sigmas_t=[1, sig]
    # )

    net = dde.nn.FNN(layer_size, activation, initializer)


    net.apply_output_transform(lambda x, y: x[:, 0:1] * (1 - x[:, 0:1]) * y)


    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=0.001,
        # metrics=["l2 relative error"],
        # decay=("inverse time", 2000, 0.9),
    )
    pde_residual_resampler = dde.callbacks.PDEPointResampler(period=1)
    # early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-6, patience=5000)
    losshistory, train_state = model.train(
        iterations=epochs, callbacks=[pde_residual_resampler], display_every=500, model_save_path=f"{script_directory}/{name}"
    )
    dde.saveplot(losshistory, train_state, output_dir=f"{script_directory}")

    plt.close()
    plt.clf()
    # Predictions
    t = np.linspace(0, 1, num=100)
    x = np.linspace(0, 1, num=100)
    t, x = np.meshgrid(t, x)
    X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))

    u_pred = model.predict(X_star)
    U_pred = u_pred.reshape(100, 100)

    # Predictions
    f1 = plt.figure()


    plt.pcolor(t, x, U_pred, cmap='inferno')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Predicted u(x)')

    plt.tight_layout()
    plt.savefig(f"{script_directory}/dde1_{name}.png")
    plt.show()
    plt.close()
    plt.clf()

    # Convert the list of arrays to a 2D NumPy array
    matrix = np.array(losshistory.loss_train)

    # Separate the components into different arrays
    loss_res = matrix[:, 0]
    # loss_bcs = matrix[:, 1]
    loss_u_t_ics = matrix[:, 1]
    loss_du_t_ics = matrix[:, 2]

    # l2_error = np.array(losshistory.metrics_test)

    fig = plt.figure(figsize=(6, 5))
    iters = 500 * np.arange(len(loss_res))
    with sns.axes_style("darkgrid"):
        plt.plot(iters, loss_res, label='$\mathcal{L}_{r}$')
        # plt.plot(iters, loss_bcs, label='$\mathcal{L}_{u}$')
        plt.plot(iters, loss_u_t_ics, label='$\mathcal{L}_{u_0}$')
        plt.plot(iters, loss_du_t_ics, label='$\mathcal{L}_{u_t}$')
        # plt.plot(iters, l2_error, label='$\mathcal{L}^2 error$')
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(f"{script_directory}/dde2_{name}.png")
        plt.show()

    # plot_animation(model, name)

if __name__ == "__main__":
    go_sc(20000)