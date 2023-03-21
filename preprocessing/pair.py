# Class representing a pair of linked points in the droplet (typically on opposite sides of the droplet)
import numpy as np


class PointPair:
    """
    A PointPair represents two linked points on either side of a droplet image which occur at the same percentile away
    from the centre.
    """
    def __init__(self, rname, lname, rind, lind):
        self.r_name = rname
        self.r_index = rind
        self.r_values = []

        self.l_name = lname
        self.l_index = lind
        self.l_values = []

    def pair_mean(self):
        """
        Computes elementwise mean of each value present in the pairs.

        :return:
        """
        return [sum(pair)/2 if None not in pair else None for pair in zip(self.r_values, self.l_values)]

    def merged_title(self):
        """
        Constructs merged title of the pair (for use with the mean)

        :return:
        """
        return self.l_name + "_to_" + self.r_name
