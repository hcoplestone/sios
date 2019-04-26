from integrators import GalerkinGaussLobattoIntegrator, FirstOrderIntegrator
from timeit import default_timer as timer
import numpy as np

import sympy.functions.elementary.trigonometric as sp_trig

# Define our properties
lambda_1 = 0.01
lambda_3 = 0.025

g = 9.81
M = 5.0
R = 0.7

psi_0 = 0.0
phi_0 = 0.0
theta_0 = 1*np.pi/4

# v_psi_0 = 18.7 * 4
v_psi_0 = 18.7 * 4.0
# v_phi_0 = 0.5 * np.pi
v_phi_0 = 3.5 * np.pi
v_theta_0 = -3 * np.pi * 8

p_psi_0 = float(
    lambda_3 * (v_psi_0 + v_phi_0 * np.cos(theta_0))
)

p_phi_0 = float(
    lambda_1 * v_phi_0 * (np.sin(theta_0) * np.sin(theta_0)) + p_psi_0 * np.cos(theta_0)
)

p_theta_0 = float(
    lambda_1 * v_theta_0
)

print("p_psi = {}".format(p_psi_0))
print("p_tho = {}".format(p_phi_0))
print("p_theta = {}".format(p_theta_0))


# p_theta_0 = 0.0


class SpinningTop:
    def doit(self, order):
        # Create an instance of our integrator
        integrator = GalerkinGaussLobattoIntegrator('t', ['psi', 'phi', 'theta'], ['v_psi', 'v_phi', 'v_theta'], order,
                                                    verbose=True)

        # Get symbols for use in Lagrangian
        psi, phi, theta = integrator.symbols['q']
        v_psi, v_phi, v_theta = integrator.symbols['v']

        # Define the Lagrangian for the system
        L = 0.5 * lambda_1 * (v_phi * v_phi * sp_trig.sin(theta) * sp_trig.sin(theta) + v_theta * v_theta) \
            + 0.5 * lambda_3 * (v_psi + v_phi * sp_trig.cos(theta)) * (v_psi + v_phi * sp_trig.cos(theta)) \
            - M * g * R * sp_trig.cos(theta)

        # Define discretization parameters
        integrator.discretise(L, 4000, 0.0, 0.4)

        # Set the initial conditions for integration
        integrator.set_initial_conditions([psi_0, phi_0, theta_0], [p_psi_0, p_phi_0, p_theta_0])

        # Integrate the system
        start_time = timer()
        integrator.integrate()
        end_time = timer()

        # Display elapsed time
        elapsed_time = end_time - start_time
        print('Elapsed time is {0:.2f}'.format(elapsed_time))

        return integrator


if __name__ == "__main__":
    sios = SpinningTop()
    spinningtop = sios.doit(2)
    spinningtop.plot_results()
    np.savez('data/spinning_top_best.npz', q_list=spinningtop.q_list, t_list=spinningtop.t_list,
             q_solutions=spinningtop.q_solutions, p_solutions=spinningtop.p_solutions)
