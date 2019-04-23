import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
k = 1.0
m = 1.0

u0 = 1
xi0 = 1

# Integration Parameters

t_start = 0
t_end = 2000


def dXdt(X, t):
    # X - vector: [u, xi]
    # ie. u = X[0] and xi = X[1]
    # Return [du/dt === xi, dxi/dtheta = -k/m*x]
    return [X[1], -(k/m)*X[0]]


def integrate(t_initial, t_final, intervals):
    X0 = [u0, xi0]
    ts = np.linspace(t_start, t_final, intervals)
    Xsolutions = odeint(dXdt, X0, ts, mxordn=1, mxhnil=1)
    return [ts, Xsolutions[:, 0]]

def plot_solutions(ts, us, title):
    # plt.subplot(211)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.plot(ts, us)

def main():
    print('Integrating oscillation...')
    ts, us = integrate(t_start, t_end, t_end*20)

    plt.figure(1)
    plot_solutions(ts, us, title='Numerical solution')

    plt.show()
if __name__ == '__main__':
    main()