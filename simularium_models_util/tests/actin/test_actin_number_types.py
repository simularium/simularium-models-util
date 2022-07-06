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
        ( # test 3; was failing bc expected_value input was wrong
            1,
            5,
            2,
            3,
        ),
        ( # test 4
            5,
            5,
            -2,
            3,      
        ),
        ( # test 5 
            1,
            3, 
            2,
            3,
        ),
        ( # test 6
            1,
            5,
            1,
            2,
        ),
        ( # test 7
            3,
            5,
            1,
            4,
        ),
        ( # test 8
            3,
            5,
            -1,
            2,
        ),
    ],
)
def test_get_actin_number(actin_number, actin_number_types, offset, expected_value):
    test_value = ActinGenerator.get_actin_number(actin_number_types, actin_number, offset)
    assert isinstance(test_value, int)
    assert test_value == expected_value


#run from simularium-models-util >> "make build"
