from typing import List
from .Integrator import Integrator
from quadpy.line_segment.gauss_lobatto import GaussLobatto
from assertions import Assertions
from prettytable import PrettyTable
import autograd.numpy as np
from scipy import optimize

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

    def set_initial_conditions(self, q_initial_value_list: List[float], v_initial_value_list: List[float]):
        """
        Set the initial conditions for the integrator.
        :param q_initial_value_list:  List of initial q values.
        :param v_initial_value_list: List of initial v = $\dot{q}$ values.
        """

        Assertions.assert_list_of_floats(q_initial_value_list, 'Initial q values')
        Assertions.assert_dimensions_match(self.q_list, 'q variables', q_initial_value_list, 'Initial q values')

        Assertions.assert_list_of_floats(v_initial_value_list, 'Initial v values')
        Assertions.assert_dimensions_match(self.v_list, 'v variables', v_initial_value_list, 'Initial v values')

        self.q_initial_value_list = q_initial_value_list
        self.v_initial_value_list = v_initial_value_list

    def integrate(self):
        """
        Numerically integrate the system.
        """

        # Setup solutions
        self.q_solutions = [np.zeros(len(self.q_list)) for i in range(self.n)]
        self.v_solutions = [np.zeros(len(self.v_list)) for i in range(self.n)]

        # Add the initial conditions to the solution
        self.q_solutions[0] = np.array(self.q_initial_value_list)
        self.v_solutions[0] = np.array(self.v_initial_value_list)

        # Iterate
        if self.verbose:
            print("\nIterating...")

        for i in range(self.n - 1):
            print(f"\nSolving for n={i+2}")

            def new_positions_from_nth_solution_equation(q_n_plus_1_trial_solutions):
                time_step = self.t_list[i + 1] - self.t_list[i]
                return FirstOrderQuadrature.trapezium_rule_nd(self.q_solutions[i], q_n_plus_1_trial_solutions, time_step)

            q_nplus1_guess = np.random.rand(len(self.q_list))
            print(f"Guessing q_n_plus_1 = {q_nplus1_guess}")

            q_nplus1_solution = optimize.root(new_positions_from_nth_solution_equation, q_nplus1_guess, method='hybr')
            self.q_solutions[i + 1] = q_nplus1_solution.x
            print(f"Solved to be q_n_plus_1 = {q_nplus1_solution.x}")

        # Display the solutions
        print()
        self.display_solutions()