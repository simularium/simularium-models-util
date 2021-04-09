#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple example of a test file using a function.
NOTE: All test file names must have one of the two forms.
- `test_<XYY>.py`
- '<XYZ>_test.py'

Docs: https://docs.pytest.org/en/latest/
      https://docs.pytest.org/en/latest/goodpractices.html#conventions-for-python-test-discovery
"""

import pytest


@pytest.mark.parametrize(
    "value, expected_value",
    [
        # (value, expected_value)
        (5, 5),
        # pytest.param(
        #     "hello",
        #     None,
        #     None,
        #     marks=pytest.mark.raises(
        #         exception=ValueError
        #     ),  # Init value isn't an integer
        # ),
    ],
)
def test_example(value, expected_value):
    assert value == expected_value
