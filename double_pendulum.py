from integrators import GalerkinGaussLobattoIntegrator, FirstOrderIntegrator
from timeit import default_timer as timer
import numpy as np

import sympy.functions.elementary.trigonometric as sp_trig

from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph import GlobbingFilter
from pycallgraph.output import GraphvizOutput

# Define our properties
m1 = 1.0
m2 = 5.0

l1 = 1.0
l2 = 1.0

theta_1_0 = 7 * np.pi / 8
theta_2_0 = 0.0

v_theta_1_0 = 0.5
v_theta_2_0 = 0.2

p_theta_1_0 = float(
    (m1 + m2) * (l1 ** 2) * v_theta_1_0 + m2 * l1 * l2 * v_theta_2_0 * np.cos(theta_2_0 - theta_1_0)
)

p_theta_2_0 = float(
    m2 * (l2 ** 2) * v_theta_2_0 + m2 * l1 * l2 * v_theta_1_0 * np.cos(theta_2_0 - theta_1_0)
)

print("p_theta_1_0 = {}".format(p_theta_1_0))
print("p_theta_2_0 = {}".format(p_theta_2_0))

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
        integrator.discretise(L, 5000, 0.0, 5.0)

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
    # config = Config()
    # config.trace_filter = GlobbingFilter(exclude=[
    #     'pycallgraph.*',
    #     '*.secret_function',
    # ])
    #
    # graphviz = GraphvizOutput(output_file='filter_exclude.png')
    #
    # with PyCallGraph(output=graphviz, config=config):
    sios = DoublePendulum()
    doublependulum = sios.doit(2)
    doublependulum.plot_results()

# np.savez('data/double_pendulum_order_comparison_4_limit.npz', ms=np.array([m1, m2]), ls=np.array([l1, l2]),
#          q_list=doublependulum.q_list, t_list=doublependulum.t_list,
#          q_solutions=doublependulum.q_solutions, p_solutions=doublependulum.p_solutions)
