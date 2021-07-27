#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from simularium_models_util.actin import ActinGenerator, ActinTestData


@pytest.mark.parametrize(
    "fibers, expected_monomers",
    [
        (
            ActinTestData.linear_actin_fiber(),
            ActinTestData.linear_actin_monomers(),
        ),
        (
            ActinTestData.branched_actin_fiber(),
            ActinTestData.branched_actin_monomers(),
        ),
    ],
)
def test_generate_monomers(fibers, expected_monomers):
    monomers = ActinGenerator.get_monomers(fibers, 0)
    assert monomers["topologies"] == expected_monomers["topologies"]
