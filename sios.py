from typing import List
from assertions import Assertions


class Integrator:

    def __init__(self, t: str, q_list: List[str], v_list: List[str]) -> None:
        """
        Implement common logic for all variational integrators we may want to build.

       :param t: What symbol to use as the independent time variable
       :param q_list: List of strings for generating symbols for $q$
       :param v_list: List of strings for generating symbols for $\dot{q}$
       """
        self.t = t
        self.q_list = q_list
        self.v_list = v_list

        self.validateIntegrator()

    def validateIntegrator(self) -> None:
        """
        Check that all the integrators parameters are as expected.
        """
        Assertions.assert_string(self.t, 'Symbol for time variable')
        Assertions.assert_list_of_strings(self.q_list, 'Generalised coordinates')


class GalerkinGaussLobatto(Integrator):

    def __init__(self, t: str, q_list: List[str], v_list: List[str]) -> None:
        """
        We subclass from a base integrator class.
        This way we can abstract away common functionality between integrators.
        """
        Integrator.__init__(self, t, q_list, v_list)

    def test(self):
        print(self.t)


def main():
    i = GalerkinGaussLobatto('t', ['1'], ['v'])
    i.test()


if __name__ == "__main__":
    main()
