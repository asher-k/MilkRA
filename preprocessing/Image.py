import numpy as np


class Constants:
    REF_DROP_PXL_BORDER = 152  # Maximum value of a BW pixel for it to be considered the droplet (when finding height)
    REF_NONDROP = 225  # Minimum value of a BW pixel for it to be considered not part of the droplet (when finding sides)
    REF_RADIUS = 10  # Radius of the search area when attempting to find the reflection line of an image
    REF_THRESH = 2.0  # Maximum difference between each side of the radius for a row to be considered reflected
    MIN_DROP_WIDTH = 100  # Pixel width of a dark region within an image in order to be considered a droplet (removes false-positives)
    REF_LB = 700  # Lower Bound where pixels below are guaranteed to not be part of the Droplet (ie only reflection)
    HEIGHT_RADIUS = 10  # Number of columns to each side of a given pixel to average over (enables smoother estimations)


CONSTS = Constants()


class Droplet:
    """
    A Droplet is composed of various measurements computed from an original image.
    """
    def __init__(self, image, height_mode, refl=None):
        self.rfl = refl
        self.hgts = []
        self.wid = None
        self.mid = None
        self.l = None
        self.r = None

        self.img = image
        self.height_mode = height_mode

    def setup(self, complete=False):
        """
        Performs calculations on the droplet to obtain quantitative features.

        :param complete: when true, heights for each column within the image are pre-computed; computationally expensive
        :return: self
        """
        self.rfl = self._reflection() if self.rfl is None else self.rfl
        _ = self._bounds(self.rfl)
        self.l = _[0]
        self.r = _[1]
        self.wid = self.r - self.l
        self.mid = self.l + self.wid//2
        if complete:
            self.hgts = [0]*self.l + [self._height(c, mode=self.height_mode) for c in range(self.l, self.r)] \
                        + [0]*(len(self.img[0])-self.l-self.wid)  # 0 padding for non-droplet columns
        else:
            self.hgts = [None]*len(self.img[0])
            self.height_average(self.mid, radius=1)
        return self

    def summary(self):
        """
        Prints a formatted string of the current values of the droplet.
        """
        print("Droplet Values: \nReflection Line\t{}".format(self.rfl),
              "\nWidth\t\t{}".format(self.wid),
              "\nMidpoint\t{}".format(self.mid),
              "\nMidpoint Height\t{}".format(self.hgts[self.mid]),
              "\nLeft side\t{}".format(self.l),
              "\nRight side\t{}".format(self.r))

    def _height(self, column, mode="top"):
        """
        Computes the height of the droplet according to the selected mode.
        The height is defined as the maximum pixel distance between the top of the droplet and the reference line.

        :param column: column of the image
        :param mode: "top"(-down) or "bottom"(-up); "top" recommended
        :return: height of the image at column
        """
        height = 0
        if mode == "top":
            while self.img[height][column] > 250:
                height += 1
                if height >= len(self.img):  # check measurement is within the bounds of the droplet, otherwise null
                    return None
            height = self.rfl - height
        elif mode == "bottom":
            while self.img[self.rfl - height][column] < CONSTS.REF_DROP_PXL_BORDER:
                height += 1
                if height >= len(self.img):
                    return None
        else:
            raise ValueError("Unknown height calculation mode: {}".format(mode))
        return height

    def _reflection(self):
        """
        Determines the reflection line of the droplet.
        The 'reference line' is defined as where the base of the droplet meets its reflection.
        """
        image = self.img
        lb = len(image)
        ref = lb - CONSTS.REF_RADIUS
        while ref > CONSTS.REF_RADIUS:  # for each row, check widths of lines above & below the current
            pre = [self._bounds(i)[1] - self._bounds(i)[0] for i in range(ref - CONSTS.REF_RADIUS, ref)]
            post = [self._bounds(i)[1] - self._bounds(i)[0] for i in range(ref, ref + CONSTS.REF_RADIUS)]
            if np.abs(np.mean(np.subtract(pre, post))) <= CONSTS.REF_THRESH:  # if within threshold, we have our reflection!
                return ref
            ref -= 1
        raise Exception("Unable to find reflection line")

    def _bounds(self, row):
        """
        Finds the left & rightmost points of the droplet at the provided row

        :param row: the row (index) to calculate bounds at
        :return: tuple consisting of (left, right) points
        """
        left, right = None, None
        row_px = self.img[row]
        for f_ind, forw in enumerate(row_px):  # left side of droplet
            if forw < CONSTS.REF_NONDROP:
                if not any([row_px[f_ind + i] >= CONSTS.REF_NONDROP for i in range(0, CONSTS.MIN_DROP_WIDTH, CONSTS.MIN_DROP_WIDTH//10)]):
                    left = f_ind
                    break
        for r_ind, rev in enumerate(reversed(row_px)):  # right side of droplet
            r_ind = len(row_px)-r_ind-1  # adjust index for reverse
            if rev < CONSTS.REF_NONDROP:
                if not any([row_px[r_ind - i] >= CONSTS.REF_NONDROP for i in range(0, CONSTS.MIN_DROP_WIDTH, CONSTS.MIN_DROP_WIDTH//10)]):
                    right = r_ind
                    break

        if left is None or right is None:
            raise Exception("Unable to find bounds of droplet")
        return left, right

    def height_average(self, column, radius=5):
        """
        Obtains the mean height from the heights of columns according to a radius, supporting both in-line and
        pre-computed height computations.

        :param column: column to obtain height at
        :param radius: radius around column to average height across
        :return:
        """
        if None in self.hgts[column-radius:column+radius+1]:  # no height pre-computation
            range_heights = [self._height(c, mode=self.height_mode) for c in range(column-radius, column+radius+1)]
            self.hgts[column-radius:column+radius+1] = range_heights  # save heights after computation
        else:
            range_heights = self.hgts[column-radius:column+radius+1]

        return None if None in range_heights else int(np.mean(range_heights))
