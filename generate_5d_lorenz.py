import os
import math
import numpy as np


def make_folder(name):
    if not os.path.isdir(name):
        os.makedirs(name)

def lorenz(x, a, b, c, d, e):
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
    
    dx = np.zeros(len(x))
    dx[0] = a * (x[2] - x[0])
    dx[1] = a * (x[3] - x[1])
    dx[2] = b * x[0] - d * x[2] - x[0] * x[4]
    dx[3] = b * x[1] - e * x[3] - x[1] * x[4]
    dx[4] = x[0] * x[2] + x[1] * x[3] - c * x[4]
    return dx

def simulate(init_conds, steps, dt, a, b, c, d, e):
    num_vars = len(init_conds)

    # Need one more for the initial values
    x = np.zeros([steps, num_vars])
    dx = np.zeros([steps, num_vars])

    # Set initial values
    for i in range(num_vars):
        x[0][i] = init_conds[i]
    # Step through "time", calculating the partial derivatives at the current
    # point and using them to estimate the next point
    for i in range(steps):
        x_dot = lorenz(x[i], a, b, c, d, e)
        dx[i] = x_dot
        if i == steps - 1:
            break
        x[i + 1] =  x[i] + x_dot * dt
    return x, dx

def pipeline(folder, init_conds=(-1, -2, -3, -4, 1), steps=10000, dt=0.0025,
             a=35.0, b=55.0, c=8.0/3, d=1.0, e=1.0): 
    x, dx = simulate(init_conds, steps, dt, a, b, c, d, e)
    make_folder(folder)
    if folder[-1] != "/":
        folder += "/"
    np.save(folder + "x_train", x)
    np.save(folder + "x_dot", dx)

    #"""
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(x[:,1], x[:,2], x[:,3])
    plt.savefig("bilbo.png")
    #"""

def main():
    # Seed for reproducibility
    np.random.seed(1000)  
    
    # Data folder
    folder = "data/5d_lorenz/"
    
    # generate data with x noise with following scales (standard deviation)
    pipeline(folder + "state-none_scale-0.0")
    


if __name__ == "__main__":
    main()