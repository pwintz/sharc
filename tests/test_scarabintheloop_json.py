#!/bin/env python3

import unittest
import copy
import scarabintheloop.utils as utils
import numpy as np


  def test_empty_dict(self):
    
    # Setup
    dictionary = {}
    expected_json_string = r"{\s*}"

    # Execute
    json_string = utils._create_json_string(dictionary)

    self.assertRegex(json_string, expected_json_string)


  def test_list_of_numbers(self):
    list_of_numbers = [1, 2, 3]
    expected_json_string = r"\[1, 2, 3\]"

    json_string = utils._create_json_string(list_of_numbers)

    self.assertRegex(json_string, expected_json_string)
    
  def test_numpy_1D_array(self):
    numpy_array = np.array([1, 2, 3])
    expected_json_string = r"\[1, 2, 3\]"

    json_string = utils._create_json_string(numpy_array)

    self.assertRegex(json_string, expected_json_string)

  def test_numpy_2D_array(self):
    numpy_array = np.array([[1], [2], [3]])
    expected_json_string = r"\[1, 2, 3\]"

    json_string = utils._create_json_string(numpy_array)

    self.assertRegex(json_string, expected_json_string)

#   def test_patch_error_if_patch_adds_new_entry(self):
#     """
#     Test that it can sum a list of integers
#     """
#     # Setup
#     base = {
#       "a": 1,
#       "b": 2
#     }
#     patch = {
#       "c": 3
#     }
# 
#     # Execute and check it raises an error
#     with self.assertRaises(ValueError):
#       patched = patch_dictionary(base, patch)


if __name__ == '__main__':
    unittest.main()