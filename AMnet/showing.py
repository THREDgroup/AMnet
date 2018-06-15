from mpl_toolkits.mplot3d import Axes3D
import AMnet.application
import matplotlib.pyplot
import numpy
import pkg_resources


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


def plot_curves(ax, true, line_style, topper, axes_off, x=None):
    if x is not None:
        f = x
    else:
        f = range(len(true[0, :]))

    ax.plot(f, true[0, :], 'r'+line_style,
            f, true[1, :], 'b'+line_style,
            f, true[2, :], 'g'+line_style,
            )

    if axes_off:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Get and reset ylim
    if topper is not 0:
        y0, y1 = ax.get_ylim()
        ax.set_ylim([y0, topper])

    # Scale to square
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect((x1 - x0) / (y1 - y0))


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


def plot_autoencoder_examples_along_axis(case, index, quick=True):

    # Load network
    structure = pkg_resources.resource_filename("AMnet", "trained_models/"+case+"_structure.yml")
    weights = pkg_resources.resource_filename("AMnet", "trained_models/"+case+"_weights.h5")
    nw = AMnet.application.Network(structure, weights)
    nx = nw.network.layers[0].input_shape[1]

    r = numpy.linspace(-1, 1, nx)

    counter = 0
    for i in range(nx):
        for j in range(nx):
            vector = numpy.zeros([1, nx])

            # Plot input
            vector[0][j] = r[i]
            ax = matplotlib.pyplot.subplot(nx, nx, counter+1, projection='3d')
            output = nw.prediction_from_encoded(vector)
            plot_voxels(ax, output, 'g', quick, True)
            counter += 1

    matplotlib.pyplot.savefig(pkg_resources.resource_filename("AMnet", "figures/"+case+"_examples.png"), dpi=1000)