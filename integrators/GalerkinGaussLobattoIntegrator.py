from typing import List
from .Integrator import Integrator
from quadpy.line_segment.gauss_lobatto import GaussLobatto
from assertions import Assertions
from prettytable import PrettyTable
import autograd.numpy as np
from scipy import optimize
from autograd import elementwise_grad as egrad

from .quadrature import GaussLobattoQuadrature


class GalerkinGaussLobattoIntegrator(Integrator):

    def __init__(self, t: str, q_list: List[str], v_list: List[str], verbose: bool = False) -> None:
        """
        We subclass from a base integrator class.
        This way we can abstract away common functionality between integrators.
        """

        Integrator.__init__(self, t, q_list, v_list, verbose, 'Galerkin Gauss Lobatto Integrator')

    def discretise(self, expression, n: int, t_lim_lower: float, t_lim_upper: float) -> None:
        """
        Discretise the function that we provide on an interval [t_lim_lower, t_lim_upper].

        :param expression: Sympy expression for the function we want to discretise.
        :param n: The number of quadrature points to use.
        :param t_lim_lower: Lower time limit to sample our continuous function over.
        :param t_lim_upper: Upper time limit to sample our continuous function over.
        """

        Assertions.assert_integer(n, 'number of quadrature points')

        self.set_time_boundaries(t_lim_lower, t_lim_upper)
        self.n = n

        self.t_list = np.linspace(t_lim_lower, t_lim_upper, n)
        self.set_expression(expression)

    def action(self, q_n, q_n_plus_1, t, time_step):
        # Function returned accepts arguments (t, q, q1, q2..., v, v1, v2...)
        lagrangian_evaluator = self.get_expression_evaluator()

        v_n = (q_n_plus_1 - q_n) / time_step
        lagrangian_evaled_at_n = lagrangian_evaluator(t, *q_n, *v_n)
        # print('Lagrangian evaled at n:')
        # print(lagrangian_evaled_at_n)

        v_n_plus_1 = (q_n_plus_1 - q_n) / time_step
        t_n_plus_1 = t + time_step
        lagrangian_evaled_at_n_plus_1 = lagrangian_evaluator(t_n_plus_1, *q_n_plus_1, *v_n_plus_1)

        action = GaussLobattoQuadrature.trapezium_rule(lagrangian_evaled_at_n, lagrangian_evaled_at_n_plus_1, time_step)
        return action

    def integrate(self):
        """
        Numerically integrate the system.
        """

        # Setup solutions
        self.q_solutions = [np.zeros(len(self.q_list)) for i in range(self.n)]
        self.p_solutions = [np.zeros(len(self.q_list)) for i in range(self.n)]

        # Add the initial conditions to the solution
        self.q_solutions[0] = np.array(self.q_initial_value_list)
        self.p_solutions[0] = np.array(self.p_initial_value_list)

        # Iterate
        if self.verbose:
            print("\nIterating...")

        for i in range(self.n - 1):
            time_step = self.t_list[i + 1] - self.t_list[i]
            t = self.t_list[i + 1]
            print(f"\nSolving for n={i+2}")
            print(f"t={t}\n")

            def new_position_from_nth_solution_equation(q_n_plus_1_trial_solutions):
                S = lambda q_n: self.action(q_n, q_n_plus_1_trial_solutions, t, time_step)
                partial_differential_of_action_wrt_q_n = egrad(S)
                equation = self.p_solutions[i] + partial_differential_of_action_wrt_q_n(self.q_solutions[i])
                return equation

            def determine_new_momentum_from_q_n_plus_1th_solution():
                S = lambda q_n_plus_1: self.action(self.q_solutions[i], q_n_plus_1, t, time_step)
                partial_differential_of_action_wrt_q_n_plus_1 = egrad(S)
                return partial_differential_of_action_wrt_q_n_plus_1(self.q_solutions[i + 1])

            q_nplus1_guess = np.random.rand(len(self.q_list))
            print(f"Guessing q_n_plus_1 = {q_nplus1_guess}")

            q_nplus1_solution = optimize.root(new_position_from_nth_solution_equation, q_nplus1_guess, method='hybr')
            self.q_solutions[i + 1] = q_nplus1_solution.x
            print(f"Solved to be q_n_plus_1 = {q_nplus1_solution.x}")

            self.p_solutions[i + 1] = determine_new_momentum_from_q_n_plus_1th_solution()
            print(f"p_n_plus_1 = {self.p_solutions[i+1]}")

        # Display the solutions
        print()
        self.display_solutions()
