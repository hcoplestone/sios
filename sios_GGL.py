from integrators import GalerkinGaussLobattoIntegrator
from timeit import default_timer as timer


class SiosGGL:
    def doit(self):
        # Create an instance of our integrator
        integrator = GalerkinGaussLobattoIntegrator('t', ['x'], ['vx'], 1, verbose=True)

        # Define our properties and the Lagrangian for a spring
        m = 1.0

        # Get symbols for use in Lagrangian
        vx = integrator.symbols['v'][0]
        x = integrator.symbols['q'][0]

        # Define the Lagrangian for the system
        L = 0.5 * m * (vx * vx) - 2 * (x * x)

        # Define discretization parameters
        integrator.discretise(L, 50, 0.0, 8.0)

        # Set the initial conditions for integration
        integrator.set_initial_conditions([1.0], [0.0])

        # Integrate the system
        start_time = timer()
        integrator.integrate()
        end_time = timer()

        # Display elapsed time
        elapsed_time = end_time - start_time
        print('Elapsed time is {0:.2f}'.format(elapsed_time))

        # Plot the results
        integrator.plot_results()


if __name__ == "__main__":
    sios = SiosGGL()
    sios.doit()
