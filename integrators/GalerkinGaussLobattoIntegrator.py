from typing import List
from .Integrator import Integrator
from quadpy.line_segment import GaussLobatto
from assertions import Assertions


class GalerkinGaussLobattoIntegrator(Integrator):

    def __init__(self, t: str, q_list: List[str], v_list: List[str], verbose: bool = False) -> None:
        """
        We subclass from a base integrator class.
        This way we can abstract away common functionality between integrators.
        """

        Integrator.__init__(self, t, q_list, v_list, verbose, 'Galerkin Gauss Lobatto')

    def discretise(self, n: int, t_lim_lower: float, t_lim_upper: float) -> None:
        """
        Discretise our continuous function
        """



    def integrate(self):
