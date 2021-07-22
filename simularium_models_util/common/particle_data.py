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

    def __init__(self, unique_id, position, type_name = "", neighbor_ids = []):
        self.unique_id = unique_id
        self.type_name = type_name
        self.position = position
        self.neighbor_ids = neighbor_ids
