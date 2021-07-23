#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class ParticleData:
    """
    Particle data for a monomer
    """
    unique_id = ""
    type_name = ""
    position = np.zeros(3)
    neighbor_ids = []

    def __init__(self, unique_id, position, type_name = "", neighbor_ids = None):
        self.unique_id = unique_id
        self.type_name = type_name
        self.position = position
        if neighbor_ids is None:
            self.neighbor_ids = []
        else:
            self.neighbor_ids = neighbor_ids

    def __iter__(self):
        yield "unique_id", self.unique_id
        yield "type_name", self.type_name
        yield "position", self.position
        yield "neighbor_ids", self.neighbor_ids
