from typing import List
from .Integrator import Integrator
from quadpy.line_segment.gauss_lobatto import GaussLobatto
from assertions import Assertions
from prettytable import PrettyTable
import autograd.numpy as np
from scipy import optimize
from autograd import elementwise_grad as egrad

from .quadrature import FirstOrderQuadrature


class GaussLobattoScaled(GaussLobatto):

    def __init__(self, n: int, verbose: bool = False):
        """
        Calculate gauss lobatto weight and points on the interval [-1, 1].
        Then scale the distribution of points to our new interval [t_lim_lower, t_lim_upper].

        :param n:  Number of quadrature points
        :param verbose: Flag to determine if we should dump points and weights to stdio
        """
        GaussLobatto.__init__(self, n)

        self.verbose = verbose

        self.t_lim_lower = None
        self.t_lim_upper = None
        self.scaled_points = None
        self.scaled_weights = None

    def scale_to_interval(self, t_lim_lower: float, t_lim_upper: float) -> None:
        """
        Map weights and points on the interval [-1, 1] to the interval [t_lim_lower, t_lim_upper].

        :param t_lim_lower: Lower limit for t variable
        :param t_lim_upper: Upper limit for t variable
        """
        self.t_lim_upper = t_lim_upper
        self.t_lim_lower = t_lim_lower

        self.scaled_points = (self.points + 1) * 0.5 * (self.t_lim_upper - self.t_lim_lower) + self.t_lim_lower
        self.scaled_weights = self.weights * 0.5 * (self.t_lim_upper - self.t_lim_lower)

        if self.verbose:
            self.debug()

    def debug(self) -> None:
        """
        Print all necessary data for debugging
        """
        self.display_point_weight_table()

    def display_point_weight_table(self):
        """
        Display the unscaled and scaled points and weights.
        """
        table = PrettyTable()
        table.title = 'Gauss-Lobatto Quadrature'

        table.field_names = ["n", "t : [-1, 1]", "w : [-1, 1]", "t : [a, b]", "w: [a, b]"]
        for i in range(0, self.points.size):
            n = i + 1
            table.add_row([
                n,
                round(self.points[i], 2),
                round(self.weights[i], 2),
                round(self.scaled_points[i], 2),
                round(self.scaled_weights[i], 2)
            ])

        print(table)


class GalerkinGaussLobattoIntegrator(Integrator):

    def __init__(self, t: str, q_list: List[str], v_list: List[str], verbose: bool = False) -> None:
        """
        We subclass from a base integrator class.
        This way we can abstract away common functionality between integrators.
        """

        Integrator.__init__(self, t, q_list, v_list, verbose, 'Galerkin Gauss Lobatto Integrator')

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
            t = self.t_list[i + 1]
            if self.verbose:
                print('.', end='', flush=True)

            # print(f"\nSolving for n={i+2}")
            # print(f"t={t}\n")

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
            # print(f"Guessing q_n_plus_1 = {q_nplus1_guess}")

            q_nplus1_solution = optimize.root(new_position_from_nth_solution_equation, q_nplus1_guess, method='hybr')
            self.q_solutions[i + 1] = q_nplus1_solution.x
            # print(f"Solved to be q_n_plus_1 = {q_nplus1_solution.x}")

            self.p_solutions[i + 1] = determine_new_momentum_from_q_n_plus_1th_solution()
            # print(f"p_n_plus_1 = {self.p_solutions[i+1]}")

        if self.verbose:
            print("\nIntegration complete!")
