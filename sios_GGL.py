from integrators import GalerkinGaussLobattoIntegrator
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt

# Define our properties and the Lagrangian for a spring
m = 1.0
k = 1.0


class SiosGGL:
    def doit(self, order):
        # Create an instance of our integrator
        integrator = GalerkinGaussLobattoIntegrator('t', ['x', 'y'], ['vx', 'vy'], order, verbose=True)

        # Get symbols for use in Lagrangian
        vx, vy = integrator.symbols['v']
        x, y = integrator.symbols['q']

        # Define the Lagrangian for the system
        L = 0.5 * m * (vx * vx + vy * vy) - 0.5 * k * (x * x + y * y)

        # Define discretization parameters
        integrator.discretise(L, 5000, 0.0, 500.0)

        # Set the initial conditions for integration
        integrator.set_initial_conditions([1.0, 1.0], [0.0, 0.0])

        # Integrate the system
        start_time = timer()
        integrator.integrate()
        end_time = timer()

        # Display elapsed time
        elapsed_time = end_time - start_time
        print('Elapsed time is {0:.2f}'.format(elapsed_time))

        return integrator


if __name__ == "__main__":
    sios = SiosGGL()
    integrator_1 = sios.doit(1)
    integrator_2 = sios.doit(2)
    integrator_4 = sios.doit(4)

    # Work out analytic q and p trajectories
    omega = np.sqrt(k / m)
    A = 1
    analytic_q_solutions = []
    analytic_p_solutions = []
    for t in integrator_1.t_list:
        q = np.array([A * np.cos(omega * t) for q in integrator_1.q_list])
        analytic_q_solutions.append(q)
        p = np.array([-1 * A * omega * np.sin(omega * t) for q in integrator_1.q_list])
        analytic_p_solutions.append(p)

    # Energy evolution - analytic
    analytic_energies = []
    for index, phase_space_vector in enumerate(analytic_q_solutions):
        momenta = analytic_p_solutions[index]
        kinetic = np.linalg.norm(momenta) ** 2
        potential = k * np.linalg.norm(phase_space_vector) ** 2
        analytic_energies.append(kinetic + potential)

    # Energy evolution - integrated - 1st order
    energies_1 = []
    for index, phase_space_vector in enumerate(integrator_1.q_solutions):
        momenta = integrator_1.p_solutions[index]
        kinetic = np.linalg.norm(momenta) ** 2
        potential = k * np.linalg.norm(phase_space_vector) ** 2
        energies_1.append(kinetic + potential)

    # Energy evolution - integrated - 2nd order
    energies_2 = []
    for index, phase_space_vector in enumerate(integrator_2.q_solutions):
        momenta = integrator_2.p_solutions[index]
        kinetic = np.linalg.norm(momenta) ** 2
        potential = k * np.linalg.norm(phase_space_vector) ** 2
        energies_2.append(kinetic + potential)

    # Energy evolution - integrated - 4th order
    energies_4 = []
    for index, phase_space_vector in enumerate(integrator_4.q_solutions):
        momenta = integrator_4.p_solutions[index]
        kinetic = np.linalg.norm(momenta) ** 2
        potential = k * np.linalg.norm(phase_space_vector) ** 2
        energies_4.append(kinetic + potential)

    # Plot fractional energy error
    plt.figure()
    plt.loglog(integrator_1.t_list, np.abs(np.array(energies_1) / np.array(analytic_energies) - 1.), 'r-',
               linewidth=1.0, label='1st order SIOS')
    plt.loglog(integrator_2.t_list, np.abs(np.array(energies_2) / np.array(analytic_energies) - 1.), 'b-',
               linewidth=1.0, label='2nd order SIOS')
    plt.loglog(integrator_4.t_list, np.abs(np.array(energies_4) / np.array(analytic_energies) - 1.), 'g-',
               linewidth=1.0, label='4th order SIOS')
    plt.xlabel('Time, $t$', fontsize=10)
    plt.ylabel('Fractional energy error, $\delta E/E$', fontsize=10)
    plt.legend(loc='best')
    plt.show()
