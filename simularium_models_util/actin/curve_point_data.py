#!/usr/bin/env python
# -*- coding: utf-8 -*-


class CurvePointData:
    """
    structure to store data for a point on a curve
    """
    position = None
    tangent = None
    arc_length = 0

    def __init__(self, position, tangent, arc_length):
        self.position = position
        self.tangent = tangent
        self.arc_length = arc_length
