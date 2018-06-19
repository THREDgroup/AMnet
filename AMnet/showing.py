from mpl_toolkits.mplot3d import Axes3D
import AMnet.application
import matplotlib.pyplot
import numpy
import pkg_resources
import keras


def _cuboid_data(pos, size=(1, 1, 1)):
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(pos, size)]
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1], o[1], o[1]],
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]
    z = [[o[2], o[2], o[2], o[2], o[2]],
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]
    return x, y, z


def _plot_cube(position=(0, 0, 0), ax=None, color='b', size=1):
    x, y, z = _cuboid_data(position, size=(size, size, size))
    ax.plot_surface(x, y, z, color=color, rstride=1, cstride=1)


def _plot_dot(position=(0, 0, 0), ax=None, color='b'):
    ax.scatter(position[0], position[1], position[2], color=color)


def plot_voxels(ax, matrix, color, quick, axes_off, xyz=None):
    # plot a Matrix
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                if matrix[i, j, k] > 0.5:
                    if quick:
                        _plot_dot(position=(i-0.5, j-0.5, k-0.5), ax=ax, color=color)
                    else:
                        if xyz is not None:
                            x = xyz[0]
                            y = xyz[1]
                            z = xyz[2]
                            ex = abs(x[0]-x[1])
                            _plot_cube(position=(x[i]-0.5*ex, y[j]-0.5*ex, z[k]-0.5*ex), ax=ax, color=color, size=ex)
                        else:
                            _plot_cube(position=(i-0.5, j-0.5, k-0.5), ax=ax, color=color)

    if axes_off:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    if axes_off:
        g = len(matrix[0, 0, :])
        ax.set_xlim((0, g-1))
        ax.set_ylim((0, g-1))
        ax.set_zlim((0, g-1))
    else:
        ax.set_xlim((min(xyz[0]), max(xyz[0])))
        ax.set_ylim((min(xyz[1]), max(xyz[1])))
        ax.set_zlim((min(xyz[2]), max(xyz[2])))
    ax.set_aspect('equal')


def plot_random_autoencoder_examples(case, nx, ny, quick=True):
    # Load network
    structure = pkg_resources.resource_filename("AMnet", "trained_models/"+case+"_structure.yml")
    weights = pkg_resources.resource_filename("AMnet", "trained_models/"+case+"_weights.h5")
    nw = AMnet.application.Network(structure, weights)

    mult = 2

    for i in range(1, nx*ny*mult, mult):
        [_, ip, ot, op] = nw.prediction()
        idx = i

        # Plot input
        ax = matplotlib.pyplot.subplot(ny, nx * mult, idx, projection='3d')
        plot_voxels(ax, ip, 'b', quick, True)

        # Plot predicted output
        idx = idx + 1
        ax = matplotlib.pyplot.subplot(ny, nx * mult, idx, projection='3d')
        plot_voxels(ax, op, 'g', quick, True)

    matplotlib.pyplot.savefig(pkg_resources.resource_filename("AMnet", "figures/"+case+"_examples.png"), dpi=1000)


def plot_autoencoder_examples_along_axis(case, dimensions, instances, quick=True):

    # Load network
    structure = pkg_resources.resource_filename("AMnet", "trained_models/"+case+"_structure.yml")
    weights = pkg_resources.resource_filename("AMnet", "trained_models/"+case+"_weights.h5")
    nw = AMnet.application.Network(structure, weights)
    nx = nw.network.layers[0].input_shape[1]

    r = numpy.linspace(-1, 1, instances)

    counter = 0
    for i in range(instances):
        for j in range(dimensions):
            vector = numpy.zeros([1, nx])

            # Plot input
            vector[0][j] = r[i]
            ax = matplotlib.pyplot.subplot(dimensions, instances, counter+1, projection='3d')
            output = nw.prediction_from_encoded(vector)
            plot_voxels(ax, output, 'g', quick, True)
            counter += 1

    matplotlib.pyplot.savefig(pkg_resources.resource_filename("AMnet", "figures/"+case+"_examples.png"), dpi=1000)

def plot_delta_along_axes(case, dimensions, instances, delta):
    # Load the autoencoder
    structure = pkg_resources.resource_filename("AMnet", "trained_models/"+case+"_autoencoder_structure.yml")
    weights = pkg_resources.resource_filename("AMnet", "trained_models/"+case+"_autoencoder_weights.h5")
    autoencoder_network = AMnet.application.Network(structure, weights)

    # Load data
    geometry, volume, flattened_geometry, N, G = AMnet.utilities.load_data()

    # Find best example design
    output = autoencoder_network.predict_raw(flattened_geometry)
    mse = keras.backend.mean(keras.losses.binary_crossentropy(flattened_geometry, output)).eval()
    vec = mse
    best_index = numpy.argmin(vec)

    # Load the encoder
    structure = pkg_resources.resource_filename("AMnet", "trained_models/"+case+"_encoder_structure.yml")
    weights = pkg_resources.resource_filename("AMnet", "trained_models/"+case+"_encoder_weights.h5")
    encoder_network = AMnet.application.Network(structure, weights)

    # Load the encoder
    structure = pkg_resources.resource_filename("AMnet", "trained_models/"+case+"_decoder_structure.yml")
    weights = pkg_resources.resource_filename("AMnet", "trained_models/"+case+"_decoder_weights.h5")
    decoder_network = AMnet.application.Network(structure, weights)
    nx = decoder_network.network.layers[0].input_shape[1]

    # Encode the best index
    output = encoder_network.predict_raw(flattened_geometry[best_index:(best_index + 1), :])
    print(output)

    counter = 0

    deltas = numpy.linspace(-delta, delta, instances)

    for j in range(dimensions):
        for i in range(instances):
            vector = output

            # Plot input
            vector[0][j] += deltas[i]
            ax = matplotlib.pyplot.subplot(dimensions, instances, counter+1, projection='3d')
            plot_voxels(ax, decoder_network.prediction_from_encoded(vector), 'g', False, True)
            counter += 1

    matplotlib.pyplot.savefig(pkg_resources.resource_filename("AMnet", "figures/"+case+"_examples.png"), dpi=1000)


def plot_examples_for_data_augmentation(idx):
    geometry, _, _, N, _ = AMnet.utilities.load_data()

    # Make some rotation options
    faces = []
    faces.append([1, (1, 2)])
    faces.append([2, (1, 2)])
    faces.append([3, (1, 2)])
    faces.append([4, (1, 2)])
    faces.append([1, (0, 2)])
    faces.append([3, (0, 2)])

    for i, part in enumerate(geometry):
        for j, face in enumerate(faces):
            temp = part
            temp = numpy.rot90(temp, face[0], face[1])
            for quadrant in range(4):
                temp_rotated = numpy.rot90(temp, quadrant+1, (0, 1))

    return True
