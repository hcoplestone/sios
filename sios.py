from integrators import FirstOrderIntegrator
from timeit import default_timer as timer
import matplotlib.pyplot as plt


class Sios:
    def timeit(self, n=200):
        # Create an instance of our integrator
        foi = FirstOrderIntegrator('t', ['x', 'y'], ['vx', 'vy'], verbose=True)

        # Define our properties and the Lagrangian for a spring
        m = 1.0

        x, y = foi.symbols['q']
        vx, vy = foi.symbols['v']

        L = 0.5 * m * (vx * vx + vy * vy) - 2 * (x * x + y * y)

        # Define discretization parameters
        foi.discretise(L, n, 0.0, 10.0)

        # Set the initial conditions for integration
        foi.set_initial_conditions([1.0, 1.0], [0.0, 0.0])

        # Integrate the system

        start_time = timer()
        foi.integrate()
        end_time = timer()

        # Plot the results
        # foi.plot_results()

        # Display elapsed time while integrating
        elapsed_time = end_time - start_time
        print('Elapsed time is {0:.2f}'.format(elapsed_time))
        return elapsed_time

    def doit(self):
        # Create an instance of our integrator
        foi = FirstOrderIntegrator('t', ['x', 'y'], ['vx', 'vy'], verbose=False)

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

        start_time = timer()
        foi.integrate()
        end_time = timer()

        # Display the solutions and plot the results
        # foi.display_solutions()
        foi.plot_results()

        # Display elapsed time while integrating
        elapsed_time = end_time - start_time
        print('\nElapsed time is {0:.2f} seconds'.format(elapsed_time))
        return elapsed_time


if __name__ == "__main__":
    sios = Sios()
    sios.doit()
    # n = range(1,5)
    # times = []
    # for i in n:
    #     times.append(sios.timeit(i))
    # plt.plot(n, times)
    # plt.show()
