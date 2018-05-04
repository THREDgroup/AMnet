import numpy
import pkg_resources


def load_data():

    geometric_data = numpy.load(pkg_resources.resource_filename('AMnet', 'data/data_geometry.npz'))
    geometry = geometric_data['geometry']
    flattened_geometry = geometric_data['flattened_geometry']
    volume = geometric_data['volume']
    constants = numpy.load(pkg_resources.resource_filename('AMnet', 'data/constants.npz'))
    N = constants['N']
    G = constants['G']

    return geometry, volume, flattened_geometry, N, G
