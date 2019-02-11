from integrators import FirstOrderIntegrator
import autograd.numpy as np
import sympy.functions.elementary.exponential as sp_exp
from autograd import elementwise_grad as egrad

import matplotlib.pyplot as plt


class Sios:
    def doit(self):
        # Create an instance of our integrator
        # ggl = GalerkinGaussLobattoIntegrator('t', ['q1'], ['v1'], verbose=True)
        # foi = FirstOrderIntegrator('t', ['q1', 'q2'], ['v1', 'v2'], verbose=True)
        foi = FirstOrderIntegrator('t', ['x', 'y'], ['v1', 'v2'], verbose=True)

        # Define our properties and the Lagrangian for a free particle
        m = 1.0

        # Lagrangian L = T - V; V = 0 in free space
        v1 = foi.symbols['v'][0]
        v2 = foi.symbols['v'][1]
        L = 0.5 * m * (v1 * v1 + v2 * v2)
        # L = 0.0
        # L = 0.5 * m * sp_exp.exp(v) + 100*v

        # Define discretization parameters
        foi.discretise(L, 5, 0.0, 10.0)

        # Set the initial conditions for integration
        # ggl.set_initial_conditions([1.0], [1.0])
        foi.set_initial_conditions([0.0, 0.0], [1.0, 1.0])

        # Integrate the system
        foi.integrate()

        # Plot the results
        foi.plot_results()

        # Plot the Lagrangian and its derivative wrt v, both as a function of v
        # f = ggl.get_expression_evaluator()
        # F = lambda v: f(1, 1, v)
        # g = egrad(F)
        # x = np.linspace(ggl.t_lim_lower, ggl.t_lim_upper)
        # y = F(x)
        # yprime = g(x)
        # plt.plot(x, y)
        # plt.plot(x, yprime)
        # plt.show()


if __name__ == "__main__":
    sios = Sios()
    sios.doit()
