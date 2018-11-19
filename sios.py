from integrators import GalerkinGaussLobattoIntegrator
from sympy import lambdify
import autograd.numpy as np

import matplotlib.pyplot as plt

class Sios:
    def doit(self):
        # Create an instance of our integrator
        ggl = GalerkinGaussLobattoIntegrator('t', ['q'], ['v'], verbose=True)

        # Define our properties and the Lagrangian for a free particle
        m = 1.0

        # Lagrangian L = T - V; V = 0 in free space
        L = 0.5 * m * np.dot(ggl.symbols['v'], ggl.symbols['v'])

        ggl.discretise(L, 4, 1.0, 2.0)

        f = lambdify(tuple([ggl.symbols['t']] + ggl.symbols['q'] + ggl.symbols['v']), ggl.expression)

        x = np.linspace(1, 100, 100)

        for i in x:
            print(f"{float(i)}  -  {f(i,i,i)}")


if __name__ == "__main__":
    sios = Sios()
    sios.doit()