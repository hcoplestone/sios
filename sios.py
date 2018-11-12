from typing import List


class Assertions:

    @staticmethod
    def assert_list_of_strings(lst: List, description: str) -> None:
        """
        Assert the provided object is (a) a list and (b) only contains strings.

        :param lst: The list that we want to assert only contains strings.
        :param description: A human readable description of the list we are validating.
        """

        assert type(lst) is list, f'{description} must be a list.'

        for i in lst:
            assert type(i) is str, f'List {description} must contain only strings.'


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
        assert type(self.t) is str, "Symbol for time variable must be a string."
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
