import numpy
import math


__D8_DIRECTIONS__ = {
    1: (1, 2),
    2: (2, 2),
    4: (2, 1),
    8: (2, 0),
    16: (1, 0),
    32: (0, 0),
    64: (0, 1),
    128: (0, 2)
}


def __get_neighbors__(array, row, col):
    '''
    Retrieves a 3x3 block of cells from "array", with array[row][col] at
    the center.
    '''
    top_cut = row - 1
    bottom_cut = row + 2
    left_cut = col - 1
    right_cut = col + 2
    if top_cut >= 0 and left_cut >= 0 and bottom_cut <= array.shape[0] \
        and right_cut <= array.shape[1]:

        subset = array[top_cut:bottom_cut, left_cut:right_cut]
        mask = numpy.zeros(subset.shape, bool)
    else:
        subset = numpy.zeros((3, 3), array.dtype)
        mask = numpy.zeros(subset.shape, bool)
        for row in range(top_cut, bottom_cut):
            for col in range(left_cut, right_cut):
                subset_row = row - top_cut
                subset_col = col - left_cut
                if row >= 0 and col >= 0:
                    try:
                        value = array[row][col]
                        subset[subset_row][subset_col] = value
                    except:
                        mask[subset_row][subset_col] = True
                else:
                    mask[subset_row][subset_col] = True
    result = numpy.ma.array(subset, mask=mask)
    return result


def flowdir_d8(array):
    '''
    Returns a d8 flow direction grid based on a DEM. Directions are
    coded with the following values:
    32 64 128
    16  #   1
     8  4   2
    '''
    result = numpy.zeros(array.shape, 'uint8')
    directions_list = __D8_DIRECTIONS__.keys()
    directions_list.sort()
    for row in range(0, array.shape[0]):
        for col in range(0, array.shape[1]):
            value = array[row][col]
            neighbors = __get_neighbors__(array, row, col)
            comparison_slope = 0.0
            for direction in directions_list:
                coord = __D8_DIRECTIONS__[direction]
                neighbor_value = neighbors[coord[0]][coord[1]]
                if direction in (2, 8, 32, 128):
                    # Account for longer distance along diagonals
                    horizontal_dist = math.sqrt(2.0)
                else:
                    horizontal_dist = 1.0
                drop = value - neighbor_value
                slope = drop / horizontal_dist
                if slope > comparison_slope:
                    comparison_slope = slope
                    result[row][col] = direction
    return result