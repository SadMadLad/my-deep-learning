"""Collection of Functions to use for manipulation of images"""

import numpy as np
from matplotlib import pyplot

class ImageHelper:
    def ConvertTo2D(self, flattened_image, dimension_1, dimension_2):
        return np.reshape(flattened_image, shape=(dimension_1, dimension_2))
    def ConvertToFlatten(self, two_dimensional_image):
        return two_dimensional_image.flatten()
    def AddThirdDimension(self, two_dimensional_image):
        shape = two_dimensional_image.shape
        dim_1, dim_2 = shape
        shape = (dim_1, dim_2, 1)
        return np.reshape(two_dimensional_image, shape)
