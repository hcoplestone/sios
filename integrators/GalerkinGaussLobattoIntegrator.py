from typing import List
from .Integrator import Integrator
from quadpy.line_segment.gauss_lobatto import GaussLobatto
from assertions import Assertions
from prettytable import PrettyTable


class GaussLobattoScaled(GaussLobatto):

    def __init__(self, n: int, t_lim_lower: float, t_lim_upper: float, verbose: bool = False):
        """
        Calculate gauss lobatto weight and points on the interval [-1, 1].
        Then scale the distribution of points to our new interval [t_lim_lower, t_lim_upper].

        :param n:  Number of quadrature points
        :param t_lim_lower: Lower limit for t variable
        :param t_lim_upper: Upper limit for t variable
        :param verbose: Flag to determine if we should dump points and weights to stdio
        """
        self.t_lim_upper = t_lim_upper
        self.t_lim_lower = t_lim_lower
        GaussLobatto.__init__(self, n)

        self.scaled_points = None
        self.scaled_weights = None
        self.scale_to_interval()

        if verbose:
            self.debug()

    def scale_to_interval(self) -> None:
        """
        Map weights and points on the interval [-1, 1] to the interval [t_lim_lower, t_lim_upper].
        """
        self.scaled_points = (self.points + 1)*0.5*(self.t_lim_upper-self.t_lim_lower) + self.t_lim_lower
        self.scaled_weights = self.weights*0.5*(self.t_lim_upper-self.t_lim_lower)

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

        Integrator.__init__(self, t, q_list, v_list, verbose, 'Galerkin Gauss Lobatto')

    def discretise(self, expression: str, n: int, t_lim_lower: float, t_lim_upper: float) -> None:
        """

        :param expression: String expression for the function we want to discretise.
        :param n: The number of quadrature points to use.
        :param t_lim_lower: Lower time limit to sample our continuous function over.
        :param t_lim_upper: Upper time limit to sample our continuous function over.
        """

        Assertions.assert_string(expression, 'function to discretise')
        Assertions.assert_integer(n, 'number of quadrature points')
        Assertions.assert_float(t_lim_lower, 't variable lower limit')
        Assertions.assert_float(t_lim_upper, 't variable upper limit')

        gl = GaussLobattoScaled(n, t_lim_lower, t_lim_upper, True)

    def integrate(self):
        print('Integrating...')
