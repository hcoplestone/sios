from typing import List
from .Integrator import Integrator
import autograd.numpy as np
from scipy import optimize
from autograd import elementwise_grad as egrad
from assertions import Assertions

from .quadrature import FirstOrderQuadrature


class PolynomialIntegrator(Integrator):

    def __init__(self, t: str, q_list: List[str], v_list: List[str], order_of_integrator: int = 2,
                 verbose: bool = False) -> None:
        """
        We subclass from a base integrator class.
        This way we can abstract away common functionality between integrators.
        """

        Integrator.__init__(self, t, q_list, v_list, verbose, 'Polynomial Integrator')
        Assertions.assert_integer(order_of_integrator, "Order of integrator")
        self.order_of_integrator = order_of_integrator

    def interpolate_path(self, q_n, q_n_plus_1, t_lower, t_upper, t):
        return q_n + (q_n_plus_1 - q_n) * (t - t_lower) / (t_upper - t_lower)

    def action(self, q_n, q_n_plus_1, t, time_step):
        t_lower = t
        t_upper = t + time_step

        # Function returned accepts arguments (t, q, q1, q2..., v, v1, v2...)
        lagrangian_evaluator = self.get_expression_evaluator()

        path = lambda time: self.interpolate_path(q_n, q_n_plus_1, t_lower, t_upper, time)

        def time_derivative_of_path(time):
            components = []
            for i in range(len(self.q_list)):
                path_i = lambda time: self.interpolate_path(q_n[i], q_n_plus_1[i], t_lower, t_upper, time)
                components.append(egrad(path_i)(time))
            return components

        lagrangian_evaled_at_n = lagrangian_evaluator(t, *path(t_lower), *time_derivative_of_path(t_lower))

        lagrangian_evaled_at_n_plus_1 = lagrangian_evaluator(t_upper, *path(t_upper), *time_derivative_of_path(t_upper))

        action = FirstOrderQuadrature.trapezium_rule(lagrangian_evaled_at_n, lagrangian_evaled_at_n_plus_1, time_step)
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
            time_step = self.t_list[i + 1] - self.t_list[i]
            t = self.t_list[i]
            if self.verbose:
                print('.', end='', flush=True)

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

            q_nplus1_solution = optimize.root(new_position_from_nth_solution_equation, q_nplus1_guess, method='hybr')
            self.q_solutions[i + 1] = q_nplus1_solution.x

            self.p_solutions[i + 1] = determine_new_momentum_from_q_n_plus_1th_solution()

        if self.verbose:
            print("\nIntegration complete!")
