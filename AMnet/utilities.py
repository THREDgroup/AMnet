import numpy
import pkg_resources


def load_data():

    geometric_data = numpy.load(pkg_resources.resource_filename('AMnet', 'data/data_geometry.npz'))
    geometry = geometric_data['geometry']
    flattened_geometry = geometric_data['flattened_geometry']
    mass = geometric_data['mass'].flatten()
    support_material = geometric_data['support_material'].flatten()
    print_time = geometric_data['print_time'].flatten()
    constants = numpy.load(pkg_resources.resource_filename('AMnet', 'data/constants.npz'))
    N = constants['N']
    G = constants['G']

    return geometry, mass, support_material, print_time, flattened_geometry, N, G
