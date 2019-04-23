from integrators import GalerkinGaussLobattoIntegrator
from timeit import default_timer as timer
import numpy as np

import sympy.functions.elementary.trigonometric as sp_trig

# Define our properties
m1 = 1000.0
m2 = 1.0

l1 = 1.0
l2 = 1.0

theta_1_0 = 0.0
theta_2_0 = np.pi/2

v_theta_1_0 = 0.0
v_theta_2_0 = 0.0

p_theta_1_0 = float(
    (m1 + m2) * (l1 ** 2) * v_theta_1_0 + m2 * l1 * l2 * v_theta_2_0 * np.cos(theta_2_0 - theta_1_0)
)

p_theta_2_0 = float(
    m2 * (l2 ** 2) * v_theta_2_0 + m2 * l1 * l2 * v_theta_1_0 * np.cos(theta_2_0 - theta_1_0)
)

g = 9.81


class DoublePendulum:
    def doit(self, order):
        # Create an instance of our integrator
        integrator = GalerkinGaussLobattoIntegrator('t', ['theta_1', 'theta_2'], ['v_theta_1', 'v_theta_2'], order,
                                                    verbose=True)

        # Get symbols for use in Lagrangian
        theta_1, theta_2 = integrator.symbols['q']
        v_theta_1, v_theta_2 = integrator.symbols['v']

        # Define the Lagrangian for the system
        L = 0.5 * (m1 + m2) * l1 * l1 * v_theta_1 * v_theta_1 \
        + 0.5 * m2 * l2 * l2 * v_theta_2 * v_theta_2 \
        + m2 * l1 * l2 * v_theta_1 * v_theta_2 * sp_trig.cos(theta_1 - theta_2) \
        + (m1 + m2) * g * l1 * sp_trig.cos(theta_1) \
        + m2 * g * l2 * sp_trig.cos(theta_2)

        # Define discretization parameters
        integrator.discretise(L, 100, 0.0, 1.0)

        # Set the initial conditions for integration
        integrator.set_initial_conditions([theta_1_0, theta_2_0], [p_theta_1_0, p_theta_2_0])

        # Integrate the system
        start_time = timer()
        integrator.integrate()
        end_time = timer()

        # Display elapsed time
        elapsed_time = end_time - start_time
        print('Elapsed time is {0:.2f}'.format(elapsed_time))

        return integrator


if __name__ == "__main__":
    sios = DoublePendulum()
    doublependulum = sios.doit(2)
    doublependulum.plot_results()
    np.savez('data/double_pendulum.npz', q_list=doublependulum.q_list, t_list=doublependulum.t_list,
             q_solutions=doublependulum.q_solutions, p_solutions=doublependulum.p_solutions)
