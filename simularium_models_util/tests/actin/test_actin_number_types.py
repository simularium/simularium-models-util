#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from simularium_models_util.actin import ActinGenerator


@pytest.mark.parametrize(
    "actin_number, actin_number_types, offset, expected_value", # needs to match arguments to test_...()
    [
        ( # test 1
            3, # 1-3 or 1-5
            3, # 3 or 5
            1, # from -2 to +2
            1,
        ),
        ( # test 2
            3, # 1-3 or 1-5
            5, # 3 or 5
            1, # from -2 to +2
            4,
        ),
    ],
)
def test_get_actin_number(actin_number, actin_number_types, offset, expected_value):
    test_value = ActinGenerator.get_actin_number(actin_number, actin_number_types, offset)
    assert isinstance(test_value, int)
    assert test_value == expected_value
