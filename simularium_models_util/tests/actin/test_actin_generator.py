#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from simularium_models_util.actin import (
    ActinGenerator,
    # ActinTestData,
    FiberData,
)
from simularium_models_util.tests.conftest import (
    # assert_monomers_equal,
    assert_fibers_equal,
)


# @pytest.mark.parametrize(
#     "fibers, expected_monomers",
#     [
#         (
#             ActinTestData.linear_actin_fiber(),
#             ActinTestData.linear_actin_monomers(),
#         ),
#         (
#             ActinTestData.complex_branched_actin_fiber(),
#             ActinTestData.complex_branched_actin_monomers(),
#         ),
#     ],
# )
# def test_generate_monomers(fibers, expected_monomers):
#     monomers = ActinGenerator.get_monomers(fibers, use_uuids=False)
#     assert_monomers_equal(monomers, expected_monomers)


@pytest.mark.parametrize(
    "fibers, child_box_center, child_box_size, expected_fibers",
    [
        (
            [
                FiberData(
                    fiber_id=1,
                    points=[
                        np.array([1000.0, 2000.0, 2000.0]),
                        np.array([3160.0, 2000.0, 2000.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=1,
                    points=[
                        np.array([500.0, 2000.0, 2000.0]),
                        np.array([1000.0, 2000.0, 2000.0]),
                        np.array([3160.0, 2000.0, 2000.0]),
                        np.array([4000.0, 2000.0, 2000.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=1,
                    points=[
                        np.array([1000.0, 2000.0, 2000.0]),
                        np.array([2200.0, 2000.0, 2000.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=1,
                    points=[
                        np.array([1700.0, 2000.0, 2000.0]),
                        np.array([3160.0, 2000.0, 2000.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=1,
                    points=[
                        np.array([1000.0, 3000.0, 2000.0]),
                        np.array([2000.0, 2000.0, 2000.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=1,
                    points=[
                        np.array([2000.0, 2000.0, 2000.0]),
                        np.array([3000.0, 3000.0, 2000.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=1,
                    points=[
                        np.array([1000.0, 3000.0, 2000.0]),
                        np.array([3000.0, 1000.0, 2000.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=1,
                    points=[
                        np.array([1000.0, 3000.0, 2000.0]),
                        np.array([2000.0, 4000.0, 2000.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=1,
                    points=[
                        np.array([1700.0, 2200.0, 2000.0]),
                        np.array([2700.0, 2200.0, 2000.0]),
                        np.array([2700.0, 1800.0, 2000.0]),
                        np.array([2000.0, 1800.0, 2000.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
            ],
            0.5 * np.array(3 * [4000.0]),
            1000.0,
            [
                FiberData(
                    fiber_id=1,
                    points=[
                        np.array([-500.0, 0.0, 0.0]),
                        np.array([500.0, 0.0, 0.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=1,
                    points=[
                        np.array([-500.0, 0.0, 0.0]),
                        np.array([500.0, 0.0, 0.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=1,
                    points=[
                        np.array([-500.0, 0.0, 0.0]),
                        np.array([200.0, 0.0, 0.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=1,
                    points=[
                        np.array([-300.0, 0.0, 0.0]),
                        np.array([500.0, 0.0, 0.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=1,
                    points=[
                        np.array([-500.0, 500.0, 0.0]),
                        np.array([0.0, 0.0, 0.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=1,
                    points=[
                        np.array([0.0, 0.0, 0.0]),
                        np.array([500.0, 500.0, 0.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=1,
                    points=[
                        np.array([-500.0, 500.0, 0.0]),
                        np.array([500.0, -500.0, 0.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=1,
                    points=[
                        np.array([-300.0, 200.0, 0.0]),
                        np.array([500.0, 200.0, 0.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
                FiberData(
                    fiber_id=1,
                    points=[
                        np.array([500.0, -200.0, 0.0]),
                        np.array([0.0, -200.0, 0.0]),
                    ],
                    type_name="Actin-Polymer",
                ),
            ],
        ),
    ],
)
def test_crop_fibers(fibers, child_box_center, child_box_size, expected_fibers):
    cropped_fibers = ActinGenerator.get_cropped_fibers(
        fibers, child_box_center, child_box_size
    )
    assert_fibers_equal(cropped_fibers, expected_fibers)
