import numpy
import math


__D8_DIRECTIONS__ = {
    1: (1, 2),  # east 
    2: (2, 2),  # southeast
    4: (2, 1),  # south
    8: (2, 0),  # southwest
    16: (1, 0), # west
    32: (0, 0), # northwest
    64: (0, 1), # north
    128: (0, 2) # northeast
}


def __get_3x3_edges__(row, col):
    '''
    For cell (row, col) in a 2D array, return a dictionary of "edges" to query
    the 3x3 block surrounding the cell.
    '''
    top_edge = row - 1
    bottom_edge = row + 2
    left_edge = col - 1
    right_edge = col + 2
    return {'top': top_edge,
            'bottom': bottom_edge,
            'left': left_edge,
            'right': right_edge}


def __in_middle__(array, row, col, edges=None):
    '''
    Return True if (row, col) falls in the middle of 2D array.
    Return False if (row, col) falls along the edge of 2D array.
    '''
    if edges == None:
        edges = __get_3x3_edges__(row, col)
    if edges['top'] >= 0 and edges['left'] >= 0 \
        and edges['bottom'] <= array.shape[0] \
        and edges['right'] <= array.shape[1]:
        return True
    else:
        return False


def __get_neighbors__(array, row, col):
    '''
    Retrieves a 3x3 block of cells from "array", with array[row][col] at
    the center.
    '''
    edges = __get_3x3_edges__(row, col)
    if __in_middle__(array, row, col, edges):
        subset = array[edges['top']:edges['bottom'], 
                       edges['left']:edges['right']]
        mask = numpy.zeros(subset.shape, bool)
    else:
        subset = numpy.zeros((3, 3), array.dtype)
        mask = numpy.zeros(subset.shape, bool)
        for i in range(edges['top'], edges['bottom']):
            for j in range(edges['left'], edges['right']):
                subset_row = i - edges['top']
                subset_col = j - edges['left']
                if i >= 0 and j >= 0:
                    try:
                        value = array[i][j]
                        subset[subset_row][subset_col] = value
                    except:
                        mask[subset_row][subset_col] = True
                else:
                    mask[subset_row][subset_col] = True
    result = numpy.ma.array(subset, mask=mask)
    return result


def __pick_direction__(directions):
    '''
    Given a list of flow directions, select the middle-most direction.
    For an even number of directions, err counterclockwise.
    '''
    if len(directions) % 2 == 1:
        return numpy.median(directions)
    else:
        return directions[(len(directions) / 2) - 1]


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
    for i, row in enumerate(array):
        for j, value in enumerate(row):
            neighbors = __get_neighbors__(array, i, j)
            masked_neighbors = []
            comparison_slope = 0.0
            for direction in directions_list:
                coord = __D8_DIRECTIONS__[direction]
                neighbor_value = neighbors[coord[0]][coord[1]]
                if neighbor_value is numpy.ma.masked:
                    masked_neighbors.append(direction)
                else:
                    if direction in (2, 8, 32, 128):
                        # Account for longer distance along diagonals
                        horizontal_dist = math.sqrt(2.0)
                    else:
                        horizontal_dist = 1.0
                    drop = value - neighbor_value
                    slope = drop / horizontal_dist
                    if slope > comparison_slope:
                        comparison_slope = slope
                        result[i][j] = direction
            # Let outlet point flow away from other cells. Select the
            # middle-most direction.
            if result[i][j] == 0 and len(masked_neighbors) > 0:
                result[i][j] = __pick_direction__(masked_neighbors)
    return result
