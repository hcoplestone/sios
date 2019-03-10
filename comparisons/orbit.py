import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
G = 6.67408e-11
MEarth = 5.972e24
Mmoon = 7.34767309e22

# Orbital Parameters
mu = G*(MEarth + Mmoon)
J = 2.9e34
j = J/Mmoon
perigeeRadius = 3565e5

# Initial Conditions
r0 = perigeeRadius
u0 = 1/r0
xi0 = 0

def dXdtheta(X, theta):
    # X - vector: [u, xi]
    # ie. u = X[0] and xi = X[1]
    # Return [du/dtheta === xi, dxi/dtheta = mu/J^2 - u]
    return [X[1], (mu/np.power(j, 2)) - X[0]]


def integrate(theta_initial, theta_final, intervals):
    X0 = [u0, xi0]
    thetas = np.linspace(theta_initial, theta_final, intervals) 
    Xsolutions = odeint(dXdtheta, X0, thetas)
    return [thetas, Xsolutions[:, 0]]

def plot_solutions(thetas, us):
    rs = np.reciprocal(us)
    xs = rs * np.cos(thetas)
    ys = rs * np.sin(thetas)

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title('Numerical solution of Keplerian orbit of Moon around Earth')
    plt.plot(xs, ys)
    plt.show()

def main():
    print('Integrating orbit...')
    thetas, us = integrate(0, 2*np.pi, 1000)
    plot_solutions(thetas, us)


if __name__ == '__main__':
    main()