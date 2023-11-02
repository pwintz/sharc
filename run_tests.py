import unittest

# test_suite = unittest.TestLoader().loadTestsFromModule("tests")
test_suite = unittest.TestLoader().discover("tests", pattern="test*.py")

runner = unittest.TextTestRunner()
runner.run(test_suite)