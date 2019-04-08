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

# Integration Parameters
cycles = 2

theta_start = 0
theta_end = 2*np.pi*cycles
intervals = 200*cycles

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))

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

def analytic(theta_initial, theta_final, intervals, e, omega):
    X0 = [u0, xi0]
    thetas = np.linspace(theta_initial, theta_final, intervals) 
    us = (mu/np.power(j, 2)) * (1 + e*np.cos(thetas-omega))
    return [thetas, us]

def plot_solutions(thetas, us, title):
    # plt.subplot(211)

    rs = np.reciprocal(us)
    xs = rs * np.cos(thetas)
    ys = rs * np.sin(thetas)

    # plt.xlabel(r'$x$')
    # plt.ylabel(r'$y$')
    # plt.title(title)
    # plt.plot(xs, ys)

    # plt.subplot(212)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$r$')
    plt.plot(thetas, rs)

    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

def main():
    print('Integrating orbit...')
    thetas, us = integrate(theta_start, theta_end, intervals)

    plt.figure(1)
    plot_solutions(thetas, us, title='Numerical solution')

    print('Solving analytically...')
    thetas, us = analytic(theta_start, theta_end, intervals, 0.0549, 2.66166e-06)

    plt.figure(2)
    plot_solutions(thetas, us, title='Analytic solution')

    plt.show()
if __name__ == '__main__':
    main()