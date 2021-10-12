#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from simularium_models_util.actin import ActinGenerator, ActinTestData
from simularium_models_util.tests.conftest import assert_topologies_equal


@pytest.mark.parametrize(
    "fibers, expected_monomers",
    [
        (
            ActinTestData.linear_actin_fiber(),
            ActinTestData.linear_actin_monomers(),
        ),
        (
            ActinTestData.complex_branched_actin_fiber(),
            ActinTestData.complex_branched_actin_monomers(),
        ),
    ],
)
def test_generate_monomers(fibers, expected_monomers):
    monomers = ActinGenerator.get_monomers(fibers, use_uuids=False)
    assert_topologies_equal(monomers, expected_monomers)
