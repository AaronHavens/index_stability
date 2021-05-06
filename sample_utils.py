import jax.numpy as np
import jax
import numpy.random as rnd
import time as time
from numpy import zeros as np_zeros


def N_sample_n_sphere(n_dim, radius, n_samples):
    u = rnd.normal(0,1, size=(n_dim, n_samples))
    norm  = np.linalg.norm(u,2,axis=0)
    #norm = np.sum(u**2,axis=1)**(0.5)
    return (u/norm).T

def f1(x):
    return np.array([x[0]**2, x[1]**2])

def normalize(f):
    def f_normalized(x):
        y = f(x)
        return y / np.linalg.norm(y, 2)
    
    return f_normalized

def sgn_det_jac(f, x):
    jac = jax.jacobian(f)
    vmap_jac = jax.vmap(jac)    
    result = vmap_jac(x)
    eigs = np.linalg.eigvals(result)
    det  = np_zeros(eigs.shape[0])
    for i, row in enumerate(eigs):
        nonzero = row[abs(row) > 1e-6]
        prod = np.prod(nonzero).real
        if (len(row) - len(nonzero))%2 == 1:
            prod = -1 * prod
        det[i] = prod
        if abs(det[i]) < 1e-6: det[i] = 0
    print(det)
    sgn_det = np.sign(det)
    return sgn_det

def sort_orientation(x, degrees):
    H_minus = x[degrees < 0, :]
    H_plus = x[degrees > 0, :]

    return H_minus, H_plus

def N_epsilon(f, x, y, epsilon=0.01):
    vmap_f = jax.vmap(f)
    f_x = vmap_f(x)
    diff = np.linalg.norm(f_x - y,2,axis=1)
    indices = np.where(diff <= epsilon)
    preimage = x[indices]
    return preimage

