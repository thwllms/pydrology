import numpy
import pydrology


TEST_ARRAY = numpy.array([[1, 2, 3, 4, 5],
                          [6, 7, 8, 9, 0],
                          [1, 2, 3, 4, 5],
                          [6, 7, 8, 9, 0]])

TEST_ARRAY_D8 = numpy.array([[0, 16, 16, 2, 4],
                             [4, 4, 4, 1, 0],
                             [0, 16, 16, 2, 4],
                             [64, 64, 64, 1, 0]])


def test_get_neighbors_middle():
    result = numpy.array([[1, 2, 3],
                          [6, 7, 8],
                          [1, 2, 3]])
    neighbors = pydrology.__get_neighbors__(TEST_ARRAY, 1, 1)
    numpy.testing.assert_equal(result, neighbors)


def test_get_neighbors_edge():
    mask = numpy.array([[True, False, False],
                        [True, False, False],
                        [True, False, False]])
    result = numpy.ma.array([[None, 1, 2],
                             [None, 6, 7],
                             [None, 1, 2]], mask = mask)
    neighbors = pydrology.__get_neighbors__(TEST_ARRAY, 1, 0)
    numpy.testing.assert_equal(result, neighbors)


def test_get_neighbors_corner():
    mask = numpy.array([[True, True, True],
                        [True, False, False],
                        [True, False, False]])
    result = numpy.ma.array([[None, None, None],
                             [None, 1, 2],
                             [None, 6, 7]], mask = mask)
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
