from integrators import FirstOrderIntegrator
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
from progress.bar import IncrementalBar

# Define our properties and the Lagrangian for a spring
m = 1.0
k = 1.0


class Sios:
    def doit(self):
        # Create an instance of our integrator
        foi = FirstOrderIntegrator('t', ['x', 'y'], ['vx', 'vy'], verbose=True)

        x, y = foi.symbols['q']
        vx, vy = foi.symbols['v']

        L = 0.5 * m * (vx * vx + vy * vy) - 0.5 * k * (x * x + y * y)

        # Define discretization parameters
        foi.discretise(L, 100000, 0.0, 5000.0)

        # Set the initial conditions for integration
        foi.set_initial_conditions([1.0, 1.0], [1.0, 0.0])

        # Integrate the system
        start_time = timer()
        foi.integrate()
        end_time = timer()

        # Display elapsed time while integrating
        elapsed_time = end_time - start_time
        print('\nElapsed time is {0:.2f} seconds'.format(elapsed_time))

        # Display the solutions and plot the results
        # foi.display_solutions()
        # foi.plot_results()
        # foi.animate_trajectory()
        return foi


if __name__ == "__main__":
    # sios = Sios()
    # sios.doit()
    # n = range(1,5)
    # times = []
    # for i in n:
    #     times.append(sios.timeit(i))
    # plt.plot(n, times)
    # plt.show()

    sios = Sios()
    integrator = sios.doit()

    # Work out analytic q and p trajectories
    print('\nComputing analytic trajectory...')
    bar = IncrementalBar('Analytic', max=integrator.n)
    omega = np.sqrt(k / m)
    psi1 = np.arctan(-omega)
    A = np.sqrt(1+omega**2)
    psi2 = -np.pi/2
    B = 1
    analytic_q_solutions = []
    analytic_p_solutions = []
    for t in integrator.t_list:
        bar.next()
        x = A * np.sin(omega*t - psi1)
        y = B * np.sin(omega*t - psi2)
        q = np.array([x, y])
        analytic_q_solutions.append(q)

        px = A * omega * np.cos(omega*t - psi1)
        py = B * omega * np.cos(omega*t - psi2)

        p = np.array([px, py])
        analytic_p_solutions.append(p)
    bar.finish()

    # Work out euler method q and p trajectories
    # print('\nIntegrating using Euler method...')
    # euler_q_solutions = []
    # euler_p_solutions = []
    #
    # euler_q_solutions.append(integrator.q_solutions[0])
    # euler_p_solutions.append(integrator.p_solutions[0])
    # dt = integrator.t_list[1] - integrator.t_list[0]
    #
    # bar = IncrementalBar('Euler', max=integrator.n)
    # bar.next()
    #
    # for i in range(1, integrator.n):
    #     # print('.', end='', flush=True)
    #     bar.next()
    #     dx_dt = A*omega*np.cos(omega*integrator.t_list[i-1] - psi1)
    #     dy_dt = B*omega*np.cos(omega*integrator.t_list[i-1] - psi2)
    #     q = euler_q_solutions[i-1] + np.array([dx_dt, dy_dt]) * dt
    #     euler_q_solutions.append(q)
    #     euler_p_solutions.append(np.array(dx_dt, dy_dt))
    # bar.finish()

    # plt.figure()
    # plt.plot(integrator.t_list, [result.item(0) for result in euler_q_solutions], 'o', markersize=1, label='Analytic')
    # plt.plot(integrator.t_list, [result.item(1) for result in integrator.q_solutions], 'x', markersize=1, label='Integrated')
    # plt.legend(loc='best')
    # plt.show()

    # print('Calculating energies...')
    bar = IncrementalBar('Analytic Energies', max=integrator.n)
    # Energy evolution - analytic
    analytic_energies = []
    for index, phase_space_vector in enumerate(analytic_q_solutions):
        bar.next()
        momenta = analytic_p_solutions[index]
        kinetic = np.linalg.norm(momenta) ** 2
        potential = k * np.linalg.norm(phase_space_vector) ** 2
        analytic_energies.append(kinetic + potential)
    bar.finish()

    # Energy evolution - sios integrated - 1st order
    energies_1 = []
    bar = IncrementalBar('First Order Energies', max=integrator.n)
    for index, phase_space_vector in enumerate(integrator.q_solutions):
        momenta = integrator.p_solutions[index]
        kinetic = np.linalg.norm(momenta) ** 2
        potential = k * np.linalg.norm(phase_space_vector) ** 2
        energies_1.append(kinetic + potential)
        bar.next()
    bar.finish()

    # bar = IncrementalBar('Euler Energies', max=integrator.n)
    # # Energy evolution - euler - 1st order
    # energies_euler = []
    # for index, phase_space_vector in enumerate(euler_q_solutions):
    #     momenta = euler_p_solutions[index]
    #     kinetic = np.linalg.norm(momenta) ** 2
    #     potential = k * np.linalg.norm(phase_space_vector) ** 2
    #     energies_euler.append(kinetic + potential)
    #     bar.next()
    # bar.finish()


    # Plot fractional energy error
    # fig2 = plt.figure(figsize=(12, 5), dpi=500)
    fig2 = plt.figure()

    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.set_ylim(1e-10, 1e1)
    ax2.set_xlim(0.1, 5000)

    ax2.loglog(integrator.t_list, np.abs(np.array(energies_1) / np.array(analytic_energies) - 1.), 'r-',
               linewidth=1.0, label='1st order symplectic - trapezium')
    # ax2.loglog(integrator.t_list, np.abs(np.array(energies_euler) / np.array(analytic_energies) - 1.), 'b-',
    #            linewidth=1.0, label='Non symplectic - Euler')
    ax2.set_xlabel('Time, $t$', fontsize=10)
    ax2.set_ylabel('Fractional energy error, $\Delta(t)$', fontsize=10)
    ax2.legend(loc='best')
    plt.show()
