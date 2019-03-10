from typing import List
from .Integrator import Integrator
import autograd.numpy as np
from scipy import optimize
from autograd import elementwise_grad as egrad

from .quadrature import GaussLobattoQuadrature


class GalerkinGaussLobattoIntegrator(Integrator):

    def __init__(self, t: str, q_list: List[str], v_list: List[str], order_of_integrator: int = 1,
                 verbose: bool = False) -> None:
        """
        We subclass from a base integrator class.
        This way we can abstract away common functionality between integrators.
        """

        Integrator.__init__(self, t, q_list, v_list, verbose, 'Galerkin Gauss Lobatto Integrator')
        self.order_of_integrator = order_of_integrator
        self.gauss_lobatto_quadrature = GaussLobattoQuadrature(self.order_of_integrator + 2, True)

    def action(self, q_n, q_n_plus_1, t, time_step):
        t_n_plus_1 = t + time_step
        self.gauss_lobatto_quadrature.scale_to_interval(t, t_n_plus_1)

        # Function returned accepts arguments (t, q, q1, q2..., v, v1, v2...)
        lagrangian_evaluator = self.get_expression_evaluator()

        action = 0.0

        for index, weight in enumerate(self.gauss_lobatto_quadrature.scaled_weights):
            scaled_t = self.gauss_lobatto_quadrature.scaled_points[index]

            if index == 0:
                qs = q_n
            elif index == len(self.gauss_lobatto_quadrature.scaled_weights):
                qs = q_n_plus_1
            else:
                qs = 0.5 * (q_n_plus_1 + q_n)

            vs = (q_n_plus_1 - q_n) / time_step

            action += weight * lagrangian_evaluator(scaled_t, *qs, *vs)

        return action

    def integrate(self):
        """
        Numerically integrate the system.
        """

        # Setup solutions with initial values
        self.setup_solutions()

        # Iterate
        if self.verbose:
            print("\nIterating...")

        for i in range(self.n - 1):
            t = self.t_list[i]
            t_next = self.t_list[i + 1]
            time_step = t_next - t

            if self.verbose:
                print('.', end='', flush=True)

            def new_position_from_nth_solution_equation(q_n_plus_1_trial_solutions):
                S = lambda q_n: self.action(q_n, q_n_plus_1_trial_solutions, t,
                                            time_step)
                partial_differential_of_action_wrt_q_n = egrad(S)
                equation = np.add(self.p_solutions[i], partial_differential_of_action_wrt_q_n(self.q_solutions[i]))
                return equation

            def determine_new_momentum_from_q_n_plus_1th_solution():
                S = lambda q_n_plus_1: self.action(self.q_solutions[i], q_n_plus_1,
                                                   t, time_step)
                partial_differential_of_action_wrt_q_n_plus_1 = egrad(S)
                return partial_differential_of_action_wrt_q_n_plus_1(self.q_solutions[i + 1])

            if (i > 1):
                q_nplus1_guess = self.q_solutions[i] + (self.q_solutions[i] - self.q_solutions[i - 1])
            else:
                q_nplus1_guess = self.q_solutions[i]

            q_nplus1_solution = optimize.root(new_position_from_nth_solution_equation, q_nplus1_guess, method='hybr')
            self.q_solutions[i + 1] = q_nplus1_solution.x

            self.p_solutions[i + 1] = determine_new_momentum_from_q_n_plus_1th_solution()

        if self.verbose:
            print("\nIntegration complete!")
