#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

from ..common import ReaddyUtil


class FiberData:
    """
    Fiber network data
    """

    fiber_id = -1
    points = []
    points_reversed = None
    mother_arp = None
    nucleated_arps = []
    bound_arps = []

    def __init__(self, fiber_id, points):
        if len(points) < 2:
            raise Exception("Fiber has less than 2 points!")
        self.fiber_id = fiber_id
        self.points = points
        self.mother_arp = None
        self.nucleated_arps = []
        self.bound_arps = []

    def pointed_point(self):
        """
        get the current first point, the pointed end
        """
        return self.points[0]

    def barbed_point(self):
        """
        get the current last point, the barbed end
        """
        return self.points[len(self.points) - 1]

    def direction(self):
        """
        get a normalized direction vector along the entire fiber
        """
        return ReaddyUtil.normalize(
            self.barbed_point().position - self.pointed_point().position
        )

    def reversed_points(self):
        """
        create the points_reversed list if not already created
        and return it
        """
        if self.points_reversed is None:
            self.points_reversed = self.points.copy()
            self.points_reversed.reverse()
        return self.points_reversed

    def get_index_of_curve_start_point(self, start_position, reverse=False):
        """
        get the index of the first fiber point that defines
        a curve from the nearest position to the start_position
        optionally starting from the barbed end of the fiber
        """
        if len(self.points) < 2:
            raise Exception("Fiber has less than 2 points!")
        if len(self.points) == 2:
            return 0
        fiber_points = self.reversed_points() if reverse else self.points
        for p in range(len(fiber_points) - 2):
            d = np.linalg.norm(start_position - fiber_points[p].position)
            arc_length = np.linalg.norm(
                fiber_points[p + 1].position - fiber_points[p].position
            )
            if d < arc_length:
                return p
        return None

    def get_index_of_closest_point(self, position):
        """
        get the index of the closest fiber point to the given position
        """
        if len(self.points) < 2:
            raise Exception("Fiber has less than 2 points!")
        closest_index = 0
        min_distance = math.inf
        for p in range(len(self.points) - 1):
            d = np.linalg.norm(position - self.points[p].position)
            if d < min_distance:
                closest_index = p
                min_distance = d
        return closest_index

    def get_indices_of_closest_points(self, position):
        """
        get the indices of the closest and next closest
        fiber points to the given position
        """
        closest_index = self.get_index_of_closest_point(position)
        if len(self.points) == 2:
            return [closest_index, 1 - closest_index]
        if closest_index == 0:
            next_closest_index = 1
        elif closest_index == len(self.points) - 1:
            next_closest_index = len(self.points) - 2
        else:
            if np.linalg.norm(
                position - self.points[closest_index + 1].position
            ) < np.linalg.norm(position - self.points[closest_index - 1].position):
                next_closest_index = closest_index + 1
            else:
                next_closest_index = closest_index - 1
        return [closest_index, next_closest_index]

    def get_nearest_position(self, position):
        """
        get the nearest position on the fiber line to a given position
        """
        closest_ix = self.get_indices_of_closest_points(position)
        fiber_dir = ReaddyUtil.normalize(
            self.points[closest_ix[1]].position - self.points[closest_ix[0]].position
        )
        v = position - self.points[closest_ix[0]].position
        d = np.dot(v, fiber_dir)
        return self.points[closest_ix[0]].position + fiber_dir * d

    def get_nearest_segment_direction(self, position):
        """
        get the direction vector of the nearest segment of the fiber
        """
        start_index = self.get_index_of_curve_start_point(
            self.get_nearest_position(position)
        )
        return ReaddyUtil.normalize(
            self.points[start_index + 1].position - self.points[start_index].position
        )

    def get_first_segment_direction(self):
        """
        get the direction vector of the first segment of the fiber at the pointed end
        """
        return ReaddyUtil.normalize(self.points[1].position - self.points[0].position)
