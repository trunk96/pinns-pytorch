import deepxde as dde
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.integrate import quad
from matplotlib.animation import FuncAnimation
import torch
# import tensorflow as tf



epochs = 20000
name = "pinns_3_inputs"
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
    y = sample[:, 1]
    x_f = 0.8
    y_f = 0.8
    height = 1
    t = sample[:, -1]

    alpha = 8.9
    za = - 300 * height * torch.exp(-400*((x-x_f)**2+(y-y_f)**2)) * (4**alpha * t**(alpha - 1) * (1 - t)**(alpha - 1))
    return za


def pde(x, z):
    # rubber parameters
    v = 0.4999 # Poisson ratio, always in [-1, 0.5], with 0.5 meaning incompressible material
    E = 3e6 # Young's modulus in Pa
    h = 0.01 # thinkness of the surface in meters
    rho = 1.34 # density in kg/m^3

    k = (E*(h**2))/(12*rho*(1-(v**2)))

    dz_tt = dde.grad.hessian(z, x, i=2, j=2)
    dz_xx = dde.grad.hessian(z, x, i=0, j=0)
    dz_yy = dde.grad.hessian(z, x, i=1, j=1)

    dz_xxxx = dde.grad.hessian(dz_xx, x, i=0, j=0)
    dz_yyyy = dde.grad.hessian(dz_yy, x, i=1, j=1)
    return dz_tt + k * (dz_xxxx+dz_yyyy) - force(x)


def plot_animation(c):

    dt = 0.01  # Time step

    # Generate meshgrid for x and y values
    x_vals = np.linspace(0, 1, 100)
    y_vals = np.linspace(0, 1, 100)
    x, y = np.meshgrid(x_vals, y_vals)
    t_values = np.arange(0, 1, dt)

    p = np.zeros((len(x_vals), len(y_vals), len(t_values)))

    # Set initial condition
    XX = np.vstack((x.flatten(), y.flatten(), np.zeros_like(x.flatten()))).T
    p[:, :, 0] = c.predict(XX).reshape(len(x_vals), len(y_vals))

    # Update function using predicted solution
    def update(frame):
        t = t_values[frame]
        XX = np.vstack((x.flatten(), y.flatten(), np.full_like(x.flatten(), t))).T

        p[:, :, frame] = c.predict(XX).reshape(len(x_vals), len(y_vals))

        ax.clear()
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Deformation (Z-axis)')
        ax.set_title(f'Membrane Deformation at t={t:.2f}')

        # Plot the 3D surface without colormap
        surf = ax.plot_surface(x, y, p[:, :, frame], rstride=1, cstride=1, alpha=0.8, antialiased=True)

        # Set fixed z-axis limits
        ax.set_zlim(-1, 1)
        return surf,

    # Create the animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ani = FuncAnimation(fig, update, frames=len(t_values), interval=1.0, blit=True)

    # Save the animation as a GIF file
    ani.save(f"{output_dir}/animation.gif", writer='imagemagick')


def create_model():

    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    ic_1 = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

    ic_2 = dde.icbc.OperatorBC(
        geomtime,
        lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=2) - func2(x),
        lambda x, _: np.isclose(x[2], 0),
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

    layer_size = [3] + [100] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

    net.apply_output_transform(lambda x, y: x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2]) * y)

    model = dde.Model(data, net)

    model.compile(
        "adam",
        lr=0.001
    )
    return model

def train(mm):

    pde_residual_resampler = dde.callbacks.PDEPointResampler(period=3)
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
    # b = restore_model(a, name)
    plot_animation(b)