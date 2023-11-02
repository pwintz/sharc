from .context import scarabizor

import unittest


class TestSum(unittest.TestCase):
    def test_list_int(self):
        """
        Test that it can sum a list of integers
        """
        self.assertEqual(scarabizor.fib(1), [0])
        self.assertEqual(scarabizor.fib(2), [0, 1, 1])
        self.assertEqual(scarabizor.fib(3), [0, 1, 1, 2])
        pass

# if __name__ == '__main__':
#     unittest.main()