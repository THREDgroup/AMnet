import numpy
import scipy.spatial
import pkg_resources
import os
import scipy.io
import AMnet.utilities
import random


def extract_data(size):

    path_to_data = pkg_resources.resource_filename("AMnet", "data/Voxelized_GE_Files_"+size+"/")

    file_list = [f for f in os.listdir(path_to_data) if os.path.isfile(os.path.join(path_to_data, f))]
    geometry = []
    flattened_geometry = []
    mass = []
    support_material = []
    print_time =[]
    sumsum = []
    for file in file_list:
        data = scipy.io.loadmat(os.path.join(path_to_data, file))
        # print(sum(data['c'].flatten()))
        v = sum(data['c'].flatten()/pow(len(data['c']), 3))
        if v > 0:
            geometry.append(data['c'])
            flattened_geometry.append(data['c'].flatten())
            mass.append(data['mass'])
            print_time.append(data['print_time'])
            support_material.append(data['support_material'])

    N = len(geometry)
    print(N)
    G = len(geometry[0])

    numpy.savez(pkg_resources.resource_filename('AMnet', 'data/data_geometry.npz'),
                geometry=geometry,
                flattened_geometry=flattened_geometry,
                mass=mass,
                support_material=support_material,
                print_time=print_time)
    numpy.savez(pkg_resources.resource_filename('AMnet', 'data/constants.npz'), N=N, G=G)

    return True


def augment_data():
    # Load the data
    geometry, mass, support_material, print_time, _, N, G = AMnet.utilities.load_data()

    # Define some variables
    augmented_geometry = []
    augmented_flattened_geometry = []
    augmented_mass = []
    augmented_print_time = []
    augmented_support_material = []

    # Make some rotation options
    faces = []
    faces.append([1, (1, 2)])
    faces.append([2, (1, 2)])
    faces.append([3, (1, 2)])
    faces.append([4, (1, 2)])
    faces.append([1, (0, 2)])
    faces.append([3, (0, 2)])

    for i, part in enumerate(geometry):
        for face in faces:
            m = mass[i]
            sm = support_material[i]
            pt = print_time[i]
            temp = part
            temp = numpy.rot90(temp, face[0], face[1])
            for quadrant in range(4):
                temp_rotated = numpy.rot90(temp, quadrant+1, (0, 1))
                augmented_geometry.append(temp_rotated)
                augmented_flattened_geometry.append(temp_rotated.flatten())
                augmented_mass.append(m)
                augmented_print_time.append(pt)
                augmented_support_material.append(sm)

    # Shuffle the data
    x = list(range(len(augmented_mass)))
    random.shuffle(x)

    numpy.savez(pkg_resources.resource_filename('AMnet', 'data/data_geometry.npz'),
                geometry=[augmented_geometry[idx] for idx in x],
                flattened_geometry=[augmented_flattened_geometry[idx] for idx in x],
                mass=[augmented_mass[idx] for idx in x],
                support_material=[augmented_support_material[idx] for idx in x],
                print_time=[augmented_print_time[idx] for idx in x])
    numpy.savez(pkg_resources.resource_filename('AMnet', 'data/constants.npz'), N=len(augmented_mass), G=G)

    print(len(augmented_mass))

    return True
