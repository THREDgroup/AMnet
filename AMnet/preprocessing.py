import numpy
import scipy.spatial
import pkg_resources
import os
import scipy.io
import AMnet.utilities
import random


def extract_data(path_to_data):

    file_list = [f for f in os.listdir(path_to_data) if os.path.isfile(os.path.join(path_to_data, f))]
    geometry = []
    flattened_geometry = []
    volume = []
    sumsum = []
    for file in file_list:
        data = scipy.io.loadmat(os.path.join(path_to_data, file))
        print(sum(data['Voxelized_GE_file_10_'].flatten()))
        v = sum(data['Voxelized_GE_file_10_'].flatten()/pow(len(data['Voxelized_GE_file_10_']), 3))
        if v > 0.005:
            geometry.append(data['Voxelized_GE_file_10_'])
            flattened_geometry.append(data['Voxelized_GE_file_10_'].flatten())
            volume.append(v)
        elif v == 0:
            print(file)

    N = len(geometry)
    print(N)
    G = len(geometry[0])

    numpy.savez(pkg_resources.resource_filename('AMnet', 'data/data_geometry.npz'),
                geometry=geometry,
                flattened_geometry=flattened_geometry,
                volume=volume)
    numpy.savez(pkg_resources.resource_filename('AMnet', 'data/constants.npz'), N=N, G=G)

    return True


def augment_data():
    # Load the data
    geometry, volume, _, N, G = AMnet.utilities.load_data()

    # Define some variables
    augmented_geometry = []
    augmented_flattened_geometry = []
    augmented_volume = []

    # Make some rotation options
    faces = []
    faces.append([])

    for i, part in enumerate(geometry):
        for face in faces:

            vol = volume[i]
            temp = part
            for quadrant in range(4):
                temp_rotated = numpy.rot90(temp, quadrant+1)
                augmented_geometry.append(temp_rotated)
                augmented_flattened_geometry.append(temp_rotated.flatten())
                augmented_volume.append(vol)

    # Shuffle the data
    x = list(range(len(augmented_volume)))
    random.shuffle(x)

    numpy.savez(pkg_resources.resource_filename('AMnet', 'data/data_geometry.npz'),
                geometry=[augmented_geometry[idx] for idx in x],
                flattened_geometry=[augmented_flattened_geometry[idx] for idx in x],
                volume=[augmented_volume[idx] for idx in x])
    numpy.savez(pkg_resources.resource_filename('AMnet', 'data/constants.npz'), N=len(volume), G=G)

    print(len(augmented_volume))

    return True