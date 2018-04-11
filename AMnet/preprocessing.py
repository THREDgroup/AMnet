import numpy
import scipy.spatial
import pkg_resources
import os
import scipy.io


def extract_data(path_to_data):

    file_list = [f for f in os.listdir(path_to_data) if os.path.isfile(os.path.join(path_to_data, f))]
    geometry = []
    flattened_geometry = []
    volume = []
    sumsum = []
    for file in file_list:
        data = scipy.io.loadmat(os.path.join(path_to_data, file))
        v = sum(data['Voxelized_GE_file_10_'].flatten()/pow(len(data['Voxelized_GE_file_10_']), 3))
        if v > 0.005:
            geometry.append(data['Voxelized_GE_file_10_'])
            flattened_geometry.append(data['Voxelized_GE_file_10_'].flatten())
            volume.append(v)

    N = len(geometry)
    print(N)
    G = len(geometry[0])

    numpy.savez(pkg_resources.resource_filename('AMnet', 'data/data_geometry.npz'), geometry=geometry, flattened_geometry=flattened_geometry, volume=volume)
    numpy.savez(pkg_resources.resource_filename('AMnet', 'data/constants.npz'), N=N, G=G)

    return True
