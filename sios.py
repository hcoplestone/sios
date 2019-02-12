from integrators import FirstOrderIntegrator


class Sios:
    def doit(self):
        # Create an instance of our integrator
        foi = FirstOrderIntegrator('t', ['x', 'y'], ['vx', 'vy'], verbose=True)

        # Define our properties and the Lagrangian for a spring
        m = 1.0

        x, y = foi.symbols['q']
        vx, vy = foi.symbols['v']

        L = 0.5 * m * (vx * vx + vy * vy) - 2 * (x * x + y * y)

        # Define discretization parameters
        foi.discretise(L, 200, 0.0, 10.0)

        # Set the initial conditions for integration
        foi.set_initial_conditions([1.0, 1.0], [0.0, 0.0])

        # Integrate the system
        foi.integrate()

        # Plot the results
        foi.plot_results()


if __name__ == "__main__":
    sios = Sios()
    sios.doit()
