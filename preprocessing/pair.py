# Class representing a pair of linked points in the droplet (typically on opposite sides of the droplet)
import numpy as np


class PointPair:
    def __init__(self, rname, lname, rind, lind):
        self.r_name = rname
        self.l_name = lname
        self.r_index = rind
        self.l_index = lind
        self.r_values = []
        self.l_values = []

    def pair_mean(self):
        """
        Computes elementwise mean of each value present in the pairs.

        :return:
        """
        return [sum(pair)/2 for pair in zip(self.r_values, self.l_values)]

    def merged_title(self):
        """
        Constructs merged title of the pair (for use with the mean)

        :return:
        """
        return self.l_name + "_to_" + self.r_name
