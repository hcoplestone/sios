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

        # if self.verbose:
        # self.debug()

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


class FirstOrderIntegrator(Integrator):

    def __init__(self, t: str, q_list: List[str], v_list: List[str], verbose: bool = False) -> None:
        """
        We subclass from a base integrator class.
        This way we can abstract away common functionality between integrators.
        """

        Integrator.__init__(self, t, q_list, v_list, verbose, 'First Order Integrator')

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

    def set_initial_conditions(self, q_initial_value_list: List[float], p_initial_value_list: List[float]):
        """
        Set the initial conditions for the integrator.
        :param q_initial_value_list:  List of initial q values.
        :param p_initial_value_list: List of initial v = $\dot{q}$ values.
        """

        Assertions.assert_list_of_floats(q_initial_value_list, 'Initial q values')
        Assertions.assert_dimensions_match(self.q_list, 'q variables', q_initial_value_list, 'Initial q values')

        Assertions.assert_list_of_floats(p_initial_value_list, 'Initial p values')
        Assertions.assert_dimensions_match(self.q_list, 'q variables', p_initial_value_list, 'Initial p values')

        self.q_initial_value_list = q_initial_value_list
        self.p_initial_value_list = p_initial_value_list

    def action(self, q_n, q_n_plus_1, t, time_step):
        # Function returned accepts arguments (t, q, q1, q2..., v, v1, v2...)
        lagrangian_evaluator = self.get_expression_evaluator()

        v_n = (q_n_plus_1 - q_n) / time_step
        lagrangian_evaled_at_n = lagrangian_evaluator(t, *q_n, *v_n)

        v_n_plus_1 = (q_n_plus_1 - q_n) / time_step
        lagrangian_evaled_at_n_plus_1 = lagrangian_evaluator(t, *q_n_plus_1, *v_n_plus_1)

        return FirstOrderQuadrature.trapezium_rule(lagrangian_evaled_at_n, lagrangian_evaled_at_n_plus_1, time_step)

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
            t = self.t_list[i+1]
            print(f"\nSolving for n={i+2}")
            print(f"t={t}\n")

            def new_position_from_nth_solution_equation(q_n_plus_1_trial_solutions):
                S = lambda q_n: self.action(q_n, q_n_plus_1_trial_solutions, t, time_step)
                partial_differential_of_action_wrt_q_n = egrad(S)
                return self.p_solutions[i] + partial_differential_of_action_wrt_q_n(self.p_solutions[i])

            def determine_new_momentum_from_q_n_plus_1th_solution():
                S = lambda q_n_plus_1: self.action(self.q_solutions[i], self.q_solutions[i+1], t, time_step)
                partial_differential_of_action_wrt_q_n_plus_1 = egrad(S)
                return partial_differential_of_action_wrt_q_n_plus_1(self.q_solutions[i+1])

            q_nplus1_guess = np.random.rand(len(self.q_list))
            print(f"Guessing q_n_plus_1 = {q_nplus1_guess}")

            q_nplus1_solution = optimize.root(new_position_from_nth_solution_equation, q_nplus1_guess, method='hybr')
            self.q_solutions[i + 1] = q_nplus1_solution.x
            print(f"Solved to be q_n_plus_1 = {q_nplus1_solution.x}")

            self.p_solutions[i+1] = determine_new_momentum_from_q_n_plus_1th_solution()
            print(f"p_n_plus_1 = {self.p_solutions[i+1]}")

        # Display the solutions
        print()
        self.display_solutions()
