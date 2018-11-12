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