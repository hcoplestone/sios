import numpy as np
from quadpy.line_segment.gauss_lobatto import GaussLobatto
from prettytable import PrettyTable


class GaussLobattoQuadrature(GaussLobatto):

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

    @staticmethod
    def approximate_integral(y0, y1, time_step):
        return 0.5 * time_step * np.add(y0, y1)
