from unittest import TestCase
from assertions import Assertions


class TestAssertions(TestCase):
    def test_assert_string(self):
        # Strings should pass
        Assertions.assert_string('this string should pass', 'a string')

        # Integers should not pass
        with self.assertRaises(AssertionError):
            Assertions.assert_string(1, 'an integer')

        # Lists should not pass
        with self.assertRaises(AssertionError):
            Assertions.assert_string([], 'a list')

    def test_assert_integer(self):
        # Integers should pass
        Assertions.assert_integer(1, 'an integer')

        # Strings should not pass
        with self.assertRaises(AssertionError):
            Assertions.assert_integer('yeet', 'a string')

        # Lists should not pass
        with self.assertRaises(AssertionError):
            Assertions.assert_integer([], 'a list')

    def test_assert_float(self):
        # Floats should pass
        Assertions.assert_float(1.5, 'an integer')

        # Integers should not pass
        with self.assertRaises(AssertionError):
            Assertions.assert_float(1, 'an integer')

        # Strings should not pass
        with self.assertRaises(AssertionError):
            Assertions.assert_float('yeet', 'a string')

        # Lists should not pass
        with self.assertRaises(AssertionError):
            Assertions.assert_float([], 'a list')

    def test_assert_list_of_strings(self):
        # Array of strings should pass
        Assertions.assert_list_of_strings(['a', 'b', 'c,'], 'list of strings')

        # Array including integers should fail
        with self.assertRaises(AssertionError):
            Assertions.assert_list_of_strings(['a', 1], 'list including integer')

        # Array including lists should fail
        with self.assertRaises(AssertionError):
            Assertions.assert_list_of_strings(['a', ['b']], 'list with nested list')

        # Passing any type other than a list to the assertion should throw an error
        with self.assertRaises(AssertionError):
            Assertions.assert_list_of_strings(1, 'an integer')

        with self.assertRaises(AssertionError):
            Assertions.assert_list_of_strings('yess', 'a string')

    def test_assert_list_of_floats(self):
        # Array of strings should pass
        Assertions.assert_list_of_floats([1.0, 2.0, 3.0], 'list of floats')

        # Array including integers should fail
        with self.assertRaises(AssertionError):
            Assertions.assert_list_of_floats(['a', 1], 'list including integer')

        # Array including lists should fail
        with self.assertRaises(AssertionError):
            Assertions.assert_list_of_floats([1.0, [2.0]], 'list with nested list')

        # Passing any type other than a list to the assertion should throw an error
        with self.assertRaises(AssertionError):
            Assertions.assert_list_of_floats('yeet', 'a string')

        with self.assertRaises(AssertionError):
            Assertions.assert_list_of_floats(1, 2)

    def test_assert_dimensions_match(self):
        # List with matching dimensions should pass
        Assertions.assert_dimensions_match(['a', 'n'], 'list 1', ['c', 'd'], 'list 2')

        # Arrays with different dimensions should fail
        with self.assertRaises(AssertionError):
          Assertions.assert_dimensions_match(['a'], 'list 1', ['b', 'c'], 'list 2')

        # Passing in objects that aren't arrays should throw errors
        with self.assertRaises(AssertionError):
            Assertions.assert_dimensions_match('yeet', 'not an array', ['b', 'c'], 'list 2')

        with self.assertRaises(AssertionError):
            Assertions.assert_dimensions_match(['a', 'b'], 'list 1', 'yeet', 'not an array')

        # Passing in non string descriptions should also throw an error
        with self.assertRaises(AssertionError):
            Assertions.assert_dimensions_match([1], None, [2], 'Description 2')

        with self.assertRaises(AssertionError):
            Assertions.assert_dimensions_match([1], 'Description 1', [2], None)
