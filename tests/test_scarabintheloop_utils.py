#!/bin/env python3

import unittest
import copy
from scarabintheloop.utils import patch_dictionary

class TestPatchDictionary(unittest.TestCase):
  def test_empty_patch(self):
    
    # Setup
    base = {
      "a": 1,
      "b": 2
    }
    patch = {} # Empty dictionary
    expected_patched = copy.deepcopy(base)

    # Execute
    patched = patch_dictionary(base, patch)

    # Check
    self.assertEqual(patched, expected_patched)


  def test_patch_error_if_patch_adds_new_entry(self):
    """
    Test that it can sum a list of integers
    """
    # Setup
    base = {
      "a": 1,
      "b": 2
    }
    patch = {
      "c": 3
    }

    # Execute and check it raises an error
    with self.assertRaises(ValueError):
      patched = patch_dictionary(base, patch)

  def test_patch_updates_existing_entry(self):
    """
    Test that it can sum a list of integers
    """
    # Setup
    base = {
      "a": 1,
      "b": 2
    }
    patch = {
      "b": 222
    }
    expected_patched = {
      "a": 1,
      "b": 222
    }

    # Execute
    patched = patch_dictionary(base, patch)

    # Check
    self.assertEqual(patched, expected_patched)

  def test_patch_error_if_nondict_replaced_by_dict(self):
    """
    Test that it can sum a list of integers
    """
    # Setup
    base = {
      "a": 1,
      "b": 2
    }
    patch = {
      "b": {
        "this is a dict": True
      }
    }

    # Execute and check error is raised
    with self.assertRaises(ValueError):
      patched = patch_dictionary(base, patch)
    
  def test_patch_error_if_dict_replaced_by_nondict(self):
      """
      Test that it can sum a list of integers
      """
      # Setup
      base = {
        "a": 1,
        "b": {
          "this is a dict": True
        }
      }
      patch = {
        "b": 2
      }

      # Execute and check error is raised
      with self.assertRaises(ValueError):
        patched = patch_dictionary(base, patch)



  def test_patch_recursive(self):
    """
    Test that it can sum a list of integers
    """
    # Setup
    base = {
      "a": 1,
      "b": {
        "ba": 21,
        "bb": 22
      }
    }
    patch = {
      "b": {
        "bb": 222
      }
    }
    expected_patched = {
      "a": 1,
      "b": {
        "ba": 21,
        "bb": 222
      }
    }

    # Execute
    patched = patch_dictionary(base, patch)

    # Check
    self.assertEqual(patched, expected_patched)

if __name__ == '__main__':
    unittest.main()