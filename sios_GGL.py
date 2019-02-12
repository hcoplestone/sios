from integrators import GalerkinGaussLobattoIntegrator
import autograd.numpy as np
import sympy.functions.elementary.exponential as sp_exp
from autograd import elementwise_grad as egrad
import sympy.functions.elementary.complexes as sp_cpx

import matplotlib.pyplot as plt


class SiosGGL:
    def doit(self):
        # Create an instance of our integrator
        ggl = GalerkinGaussLobattoIntegrator('t', ['x', 'y'], ['vx', 'vy'], verbose=True)

        # Define our properties and the Lagrangian for a spring
        m = 1.0

        # Get symbols for use in Lagrangian
        vx, vy = ggl.symbols['v']
        x, y = ggl.symbols['q']

        # Define the Lagrangian for the system
        L = 0.5 * m * (vx * vx + vy * vy) - 2 * (x * x + y * y)

        # Define discretization parameters
        ggl.discretise(L, 200, 0.0, 10.0)

        # Set the initial conditions for integration
        ggl.set_initial_conditions([1.0, 1.0], [0.0, 0.0])

        # Integrate the system
        ggl.integrate()

        # Plot the results
        ggl.plot_results()


if __name__ == "__main__":
    sios = SiosGGL()
    sios.doit()
