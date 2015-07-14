import numpy
import pydrology


TEST_ARRAY = numpy.array([[78, 72, 69, 71, 58, 49],
                          [74, 67, 56, 49, 46, 50],
                          [69, 53, 44, 37, 38, 48],
                          [64, 58, 55, 22, 31, 24],
                          [68, 61, 47, 21, 16, 19],
                          [74, 53, 34, 12, 11, 12]])

TEST_ARRAY_D8 = numpy.array([[2, 2, 2, 4, 4, 8],
                             [2, 2, 2, 4, 4, 8],
                             [1, 1, 2, 4, 8, 4],
                             [128, 128, 1, 2, 4, 8],
                             [2, 2, 1, 4, 4, 4],
                             [1, 1, 1, 1, 4, 16]])


def test_get_neighbors_middle():
    result = numpy.array([[78, 72, 69],
                          [74, 67, 56],
                          [69, 53, 44]])
    neighbors = pydrology.__get_neighbors__(TEST_ARRAY, 1, 1)
    numpy.testing.assert_equal(result, neighbors)


def test_get_neighbors_edge():
    mask = numpy.array([[True, False, False],
                        [True, False, False],
                        [True, False, False]])
    result = numpy.ma.array([[None, 78, 72],
                             [None, 74, 67],
                             [None, 69, 53]], mask = mask)
    neighbors = pydrology.__get_neighbors__(TEST_ARRAY, 1, 0)
    numpy.testing.assert_equal(result, neighbors)


def test_get_neighbors_corner():
    mask = numpy.array([[True, True, True],
                        [True, False, False],
                        [True, False, False]])
    result = numpy.ma.array([[None, None, None],
                             [None, 78, 72],
                             [None, 74, 67]], mask = mask)
    neighbors = pydrology.__get_neighbors__(TEST_ARRAY, 0, 0)
    numpy.testing.assert_equal(result, neighbors)


def test_flowdir_d8():
    result = pydrology.flowdir_d8(TEST_ARRAY)
    numpy.testing.assert_equal(result, TEST_ARRAY_D8)


if __name__ == '__main__':
    test_get_neighbors_middle()
    test_get_neighbors_edge()
    test_get_neighbors_corner()
    test_flowdir_d8()
