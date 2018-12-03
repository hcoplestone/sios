from integrators import GalerkinGaussLobattoIntegrator
import autograd.numpy as np
from autograd import elementwise_grad as egrad

import matplotlib.pyplot as plt


class Sios:
    def doit(self):
        # Create an instance of our integrator
        ggl = GalerkinGaussLobattoIntegrator('t', ['q'], ['v'], verbose=True)

        # Define our properties and the Lagrangian for a free particle
        m = 1.0

        # Lagrangian L = T - V; V = 0 in free space
        v = ggl.symbols['v'][0]
        L = 0.5 * m * v * v

        # Discretise our path
        ggl.discretise(L, 4, 0.0, 10.0)

        # Set the initial conditions for integration
        ggl.set_initial_conditions([1.0], [1.0])

        # Plot the Lagrangian and its derivative wrt v, both as a function of v
        f = ggl.get_expression_evaluator()
        F = lambda v: f(1, 1, v)
        g = egrad(F)
        x = np.linspace(ggl.t_lim_lower, ggl.t_lim_upper)
        y = F(x)
        yprime = g(x)
        plt.plot(x, y)
        plt.plot(x, yprime)
        plt.show()


if __name__ == "__main__":
    sios = Sios()
    sios.doit()
