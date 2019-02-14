from typing import List
from .Integrator import Integrator
import autograd.numpy as np
from scipy import optimize
from autograd import elementwise_grad as egrad
from assertions import Assertions

from .quadrature import GaussLobattoQuadrature
from .quadrature import FirstOrderQuadrature


class GalerkinGaussLobattoIntegrator(Integrator):

    def __init__(self, t: str, q_list: List[str], v_list: List[str], order_of_integrator: int,
                 verbose: bool = False) -> None:
        """
        We subclass from a base integrator class.
        This way we can abstract away common functionality between integrators.
        """

        Integrator.__init__(self, t, q_list, v_list, verbose, 'Galerkin Gauss Lobatto Integrator')

        Assertions.assert_integer(order_of_integrator, 'Integrator order')
        self.order_of_integrator = order_of_integrator
        self.gauss_lobatto = GaussLobattoQuadrature(self.order_of_integrator + 1, verbose=False)

    def evaluate_path(self, q_n, q_n_plus_1, t_lower, t_upper, t):
        return q_n + (q_n_plus_1 - q_n) * (t - t_lower) / (t_upper - t_lower)

    def evaluate_path_derivative(self, q_n, q_n_plus_1, t_lower, t_upper, t):
        """
        Evaluate path above using autodiff. Could be made more efficient as evaluating path twice - but just a proof of
        concept for now.
        """
        number_of_generalised_coordinates = len(self.q_list)
        path_derivative = []

        for i in range(number_of_generalised_coordinates):
            path_of_ith_generalised_coordinate_as_func_of_t = lambda time: self.evaluate_path(q_n[i], q_n_plus_1[i],
                                                                                              t_lower, t_upper, time)
            deriv_of_ith_component_of_path_evaluated_at_t = egrad(path_of_ith_generalised_coordinate_as_func_of_t)(t)
            path_derivative.append(deriv_of_ith_component_of_path_evaluated_at_t)

        return path_derivative

    def action(self, q_n, q_n_plus_1, t, time_step):
        t_lower = t
        t_upper = t + time_step

        # Function returned accepts arguments (t, q, q1, q2..., v, v1, v2...)
        lagrangian_evaluator = self.get_expression_evaluator()

        action = 0.0

        # action = 0.0
        # for index, time in enumerate(self.gauss_lobatto.scaled_points):
        #     action += self.gauss_lobatto.scaled_weights[index] * lagrangian_evaluator(
        #         time,
        #         *self.evaluate_path(q_n, q_n_plus_1, t_lower, t_upper, time),
        #         *self.evaluate_path(q_n, q_n_plus_1, t_lower, t_upper, time)
        #     )

        # return FirstOrderQuadrature.trapezium_rule(
        #     lagrangian_evaluator(t_lower, *q_n, *((q_n_plus_1-q_n)/time_step)),
        #     lagrangian_evaluator(t_upper, *q_n_plus_1, *((q_n_plus_1-q_n)/time_step)),
        #     time_step
        # )

        path_at_t_lower = self.evaluate_path(q_n, q_n_plus_1, t_lower, t_upper, t_lower)
        deriv_of_path_at_t_lower = self.evaluate_path_derivative(q_n, q_n_plus_1, t_lower, t_upper, t_lower)

        path_at_t_upper = self.evaluate_path(q_n, q_n_plus_1, t_lower, t_upper, t_upper)
        deriv_of_path_at_t_upper = self.evaluate_path_derivative(q_n, q_n_plus_1, t_lower, t_upper, t_upper)

        return FirstOrderQuadrature.trapezium_rule(
            lagrangian_evaluator(t_lower, *path_at_t_lower, *deriv_of_path_at_t_lower),
            lagrangian_evaluator(t_upper, *path_at_t_upper, *deriv_of_path_at_t_upper),
            time_step
        )

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
            time_step = self.t_list[i + 1] - self.t_list[i]
            t = self.t_list[i]
            if self.verbose:
                print('.', end='', flush=True)

            self.gauss_lobatto.scale_to_interval(self.t_list[i], self.t_list[i + 1])

            def new_position_from_nth_solution_equation(q_n_plus_1_trial_solutions):
                S = lambda q_n: self.action(q_n, q_n_plus_1_trial_solutions, t, time_step)
                partial_differential_of_action_wrt_q_n = egrad(S)
                equation = np.add(self.p_solutions[i], partial_differential_of_action_wrt_q_n(self.q_solutions[i]))
                return equation

            def determine_new_momentum_from_q_n_plus_1th_solution():
                S = lambda q_n_plus_1: self.action(self.q_solutions[i], q_n_plus_1, t, time_step)
                partial_differential_of_action_wrt_q_n_plus_1 = egrad(S)
                return partial_differential_of_action_wrt_q_n_plus_1(self.q_solutions[i + 1])

            if (i > 1):
                q_nplus1_guess = self.q_solutions[i] + (self.q_solutions[i] - self.q_solutions[i - 1])
            else:
                q_nplus1_guess = self.q_solutions[i]

            q_nplus1_solution = optimize.root(new_position_from_nth_solution_equation, q_nplus1_guess,
                                              method='hybr')
            self.q_solutions[i + 1] = q_nplus1_solution.x

            self.p_solutions[i + 1] = determine_new_momentum_from_q_n_plus_1th_solution()

        if self.verbose:
            print("\nIntegration complete!")
