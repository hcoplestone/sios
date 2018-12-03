from typing import List


class Assertions:

    @staticmethod
    def assert_string(param: str, description: str) -> None:
        """
        Assert the provided object is a string.

        :param param: The object we want to assert is a string.
        :param description: A human readable description of the object we are validating.
        :return:
        """
        assert type(param) is str, f'"{description}" must be a string.'

    @staticmethod
    def assert_integer(param: int, description: str) -> None:
        """
        Assert the provided object is an integer.

        :param param: The object we want to assert is an integer.
        :param description: A human readable description of the object we are validating.
        :return:
        """
        assert type(param) is int, f'"{description}" must be an integer.'


    @staticmethod
    def assert_float(param: float, description: str) -> None:
        """
        Assert the provided object is a float.

        :param param: The object we want to assert is a float.
        :param description: A human readable description of the object we are validating.
        :return:
        """
        assert type(param) is float, f'"{description}" must be a float.'

    @staticmethod
    def assert_list_of_strings(param: List, description: str) -> None:
        """
        Assert the provided object is (a) a list and (b) only contains strings.

        :param param: The object that we want to assert is a list of strings.
        :param description: A human readable description of the object we are validating.
        """

        assert type(param) is list, f'"{description}" must be a list.'

        for i in param:
            assert type(i) is str, f'List "{description}" must contain only strings.'

    @staticmethod
    def assert_list_of_floats(param: List, description: str) -> None:
        """
        Assert the provided object is (a) a list and (b) only contains floats.

        :param param: The object that we want to assert is a list of floats.
        :param description: A human readable description of the object we are validating.
        """

        assert type(param) is list, f'"{description}" must be a list.'

        for i in param:
            assert type(i) is float, f'List "{description}" must contain only floats.'

    @staticmethod
    def assert_dimensions_match(list_i: List, description_i: str, list_j: List, description_j: str):
        """
        Assert the dimensions of two lists are equal.

        :param list_i: The first list.
        :param description_i: Human readable description of first list.
        :param list_j: The second list.
        :param description_j: Human readable description of second list.
        :return:
        """

        assert type(list_i) is list, f'"{description_i}" must be a list.'
        assert type(list_j) is list, f'"{description_j}" must be a list.'

        assert type(description_i) is str, "description_i must be a string"
        assert type(description_j) is str, "description_j must be a string"

        assert len(list_i) == len(list_j), f'Dimensions of "{description_i}" must match dimensions of "{description_j}""'
