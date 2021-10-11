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
            ActinTestData.complex_branched_actin_fiber(),
            ActinTestData.complex_branched_actin_monomers(),
        ),
    ],
)
def test_generate_monomers(fibers, expected_monomers):
    monomers = ActinGenerator.get_monomers(fibers, use_uuids=False)
    # check topology has the correct type_name
    # and contains the correct particle_ids (in any order)
    expected_top_id = list(expected_monomers["topologies"].keys())[0]
    top_id = list(monomers["topologies"].keys())[0]
    assert len(monomers["topologies"][top_id]["type_name"]) == len(
        expected_monomers["topologies"][expected_top_id]["type_name"]
    )
    assert len(monomers["topologies"][top_id]["particle_ids"]) == len(
        expected_monomers["topologies"][expected_top_id]["particle_ids"]
    )
    for particle_id in monomers["topologies"][top_id]["particle_ids"]:
        assert (
            particle_id
            in expected_monomers["topologies"][expected_top_id]["particle_ids"]
        )
    for particle_id in expected_monomers["topologies"][expected_top_id]["particle_ids"]:
        assert particle_id in monomers["topologies"][top_id]["particle_ids"]
    # check the particle types and neighbors
    for particle_id in monomers["particles"]:
        expected_particle = expected_monomers["particles"][particle_id]
        particle = monomers["particles"][particle_id]
        assert particle["type_name"] == expected_particle["type_name"]
        assert particle["neighbor_ids"] == expected_particle["neighbor_ids"]
