#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from simularium_models_util.actin import ActinGenerator
from simularium_models_util.tests import (
    linear_actin_fiber,
    branched_actin_fiber,
    linear_actin_monomers,
    branched_actin_monomers,
)


@pytest.mark.parametrize(
    "fibers, expected_monomers",
    [
        (
            linear_actin_fiber(),
            linear_actin_monomers(),
        ),
        (
            branched_actin_fiber(),
            branched_actin_monomers(),
        ),
    ],
)
def test_generate_monomers(fibers, expected_monomers):
    monomers = ActinGenerator.get_monomers(fibers, 0)
    assert monomers["topologies"] == expected_monomers["topologies"]
