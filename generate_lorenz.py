import os
import math
import numpy as np


def make_folder(name):
    if not os.path.isdir(name):
        os.makedirs(name)

def lorenz(x, y, z, s=10, r=28, b=2.667, param_std=1.0, noise_type='x'):
    """
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
       param_std: std of noise
       noise_type: x or sinz
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z with multiplicative noise
    """

    if noise_type == 'x':
        x_dot = s * (y - x) + np.random.normal(0, param_std) * x
        y_dot = r * x - y - x * z + np.random.normal(0, param_std) * x
        z_dot = x * y - b * z + np.random.normal(0, param_std) * x
    elif noise_type == 'sinz':
        x_dot = s * (y - x) + np.random.normal(0, param_std) * np.sin(z)
        y_dot = r * x - y - x * z + np.random.normal(0, param_std) * np.sin(z)
        z_dot = x * y - b * z + np.random.normal(0, param_std) * np.sin(z)
    elif noise_type == 'raw':
        x_dot = s * (y - x) + np.random.normal(0, param_std)
        y_dot = r * x - y - x * z + np.random.normal(0, param_std)
        z_dot = x * y - b * z + np.random.normal(0, param_std)
    elif noise_type == 'after':
        x_dot = s * (y - x)
        y_dot = r * x - y - x * z
        z_dot = x * y - b * z
    elif noise_type == 'cosalphax':
        x_dot = s * (y - x) + -2 * np.cos(np.sqrt(3) * x)
        y_dot = r * x - y - x * z + -2 * np.cos(np.sqrt(3) * x)
        z_dot = x * y - b * z + -2 * np.cos(np.sqrt(3) * x)
        print(1 * np.cos(np.sqrt(3) * x))
    
    return x_dot, y_dot, z_dot

def simulate_lorenz(param_std, init_conds, s=10, r=28, b=2.667, dt=0.01,
                    num_steps=10000, noise_type='x'):
    # Need one more for the initial values
    xs = np.empty(num_steps)
    ys = np.empty(num_steps)
    zs = np.empty(num_steps)

    # Set initial values
    xs[0], ys[0], zs[0] = init_conds[0], init_conds[1], init_conds[2]
    xd, yd, zd = [], [], []

    # Step through "time", calculating the partial derivatives at the current
    # point and using them to estimate the next point
    for i in range(num_steps):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], s, r, b, param_std,
                                     noise_type)
        xd.append(x_dot)
        yd.append(y_dot)
        zd.append(z_dot)
        if i == num_steps - 1:
            break
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
    xd = np.array(xd)
    yd = np.array(yd)
    zd = np.array(zd)
    return xs, ys, zs, xd, yd, zd

def simulation(init_conds, dt, steps, scale=1.0, s=10, r=28, b=2.667,
               noise_type='x'):
    xs, ys, zs, xd, yd, zd = simulate_lorenz(scale, init_conds, s=s, r=r, b=b,
                                             dt=dt, num_steps=steps,
                                             noise_type=noise_type)
    return np.stack((xs, ys, zs), axis=1), np.stack((xd, yd, zd), axis=1)

def pipeline(folder, scale=1.0, s=10, r=28, b=8.0/3, steps=10000, dt=1e-2,
             init_conds=(0., 1., 1.05), noise_type='x'):
    x_train, x_dot_train_measured = simulation(init_conds, dt=dt, steps=steps,
                                               scale=scale, s=s, r=r, b=b,
                                               noise_type=noise_type)
    if noise_type == 'after':
        xs = (x_train.shape[0], x_train.shape[1])
        x_train = x_train + np.random.normal(0, scale, xs)
    make_folder(folder)
    if folder[-1] != "/":
        folder += "/"
    np.save(folder + "x_train", x_train)
    np.save(folder + "x_dot", x_dot_train_measured)

    """
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(x_train[:,0], x_train[:,1], x_train[:,2])
    plt.savefig("attractor.png")
    """

def main():
    # Seed for reproducibility
    np.random.seed(1000)  
    
    # Data folder
    folder = "data/lorenz/"
    
    #"""
    # generate data with x noise with following scales (standard deviation)
    scales = [0.0, 1.0, 2.5, 5.0]
    for scale in scales:
        pipeline(folder + "state-x_scale-" + str(scale), scale=scale)

    # generate data with sin(z) noise with following scales (standard deviation)
    scales = [0.0, 1.0, 2.5, 5.0, 25.0, 100.0]
    for scale in scales:
        pipeline(folder + "state-sinz_scale-" + str(scale), scale=scale,
                 noise_type='sinz')

    # generate data without state-dependent noise
    scales = [1.0, 5.0, 25.0, 100.0, 200.0, 250.0]
    for scale in scales:
        pipeline(folder + "state-raw_scale-" + str(scale), noise_type='raw',
                 scale=scale)

    # generate data by adding noise to observations after
    scales = [1.0, 2.5, 5.0, 25.0, 100.0]
    for scale in scales:
        pipeline(folder + "state-after_scale-" + str(scale), noise_type='after',
                 scale=scale)
    #"""

    # generate data by adding noise to observations after
    scales = [np.sqrt(3)]
    for scale in scales:
        pipeline(folder + "state-cosalpahax_scale-0.0",
                 noise_type='cosalphax',
                 scale=0.0)
    


if __name__ == "__main__":
    main()