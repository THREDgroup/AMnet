from mpl_toolkits.mplot3d import Axes3D
import WAnet.application
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
                if matrix[i, j, k] > 0.51:
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


def plot_examples(case, nx, ny, quick=True):
    # Load network
    structure = pkg_resources.resource_filename("WAnet", "trained_models/"+case+"_structure.yml")
    weights = pkg_resources.resource_filename("WAnet", "trained_models/"+case+"_weights.h5")
    nw = WAnet.application.Network(structure, weights)

    # Find out if its an autoencoder or a predictor
    if nw.network.layers[0].input_shape[1] == nw.network.layers[-1].output_shape[1]:
        mult = 2
    else:
        mult = 3

    for i in range(1, nx*ny*mult, mult):
        [_, ip, ot, op] = nw.prediction()
        idx = i

        y_top = 0
        if len(ip.shape) == 2:
            y_top = 1.1*numpy.max([y_top, numpy.max(ip.flatten())])
        if len(ot.shape) == 2:
            y_top = 1.1*numpy.max([y_top, numpy.max(ot.flatten())])
        if len(op.shape) == 2:
            y_top = 1.1*numpy.max([y_top, numpy.max(op.flatten())])

        print(y_top)

        # Plot input
        if len(ip.flatten()) == pow(nw.G, 3):
            ax = matplotlib.pyplot.subplot(ny, nx * mult, idx, projection='3d')
            plot_voxels(ax, ip, 'b', quick, True)
        else:
            ax = matplotlib.pyplot.subplot(ny, nx * mult, idx)
            plot_curves(ax, ip, '-', y_top, True)

        # Plot true output
        if mult == 3:
            idx = idx + 1
            if len(op.flatten()) == pow(nw.G, 3):
                ax = matplotlib.pyplot.subplot(ny, nx * mult, idx, projection='3d')
                plot_voxels(ax, ot, 'b', quick, True)
            else:
                ax = matplotlib.pyplot.subplot(ny, nx*mult, idx)
                plot_curves(ax, ot, '-', y_top, True)

        # Plot predicted output
        idx = idx + 1
        if len(op.flatten()) == pow(nw.G, 3):
            ax = matplotlib.pyplot.subplot(ny, nx * mult, idx, projection='3d')
            plot_voxels(ax, op, 'g', quick, True)
        else:
            ax = matplotlib.pyplot.subplot(ny, nx*mult, idx)
            plot_curves(ax, op, '--', y_top, True)

    matplotlib.pyplot.savefig(pkg_resources.resource_filename("WAnet", "figures/"+case+"_examples.png"), dpi=1000)


def plot_BIEM_example():
    curves, geometry, S, N, D, F, G, new_curves, new_geometry = WAnet.training.load_data()
    idx = numpy.random.randint(1, S*N)

    ax = matplotlib.pyplot.subplot(1, 2, 1)
    x = numpy.linspace(0.05, 2, F)
    plot_curves(ax, new_curves[idx].reshape((D, F))*1000000, '-', 0, False, x)
    ax.set_xlabel('Frequency ($rad/s$)')
    ax.set_ylabel('Spectral Force ($N$ or $Nm$)')
    matplotlib.pyplot.legend(['Surge', 'Heave', 'Pitch'])

    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(pkg_resources.resource_filename("WAnet", "figures/BIEM_example_curve.png"), dpi=1000)

    ax = matplotlib.pyplot.subplot(1, 2, 1, projection='3d')
    ex = 5 - 5 / G
    xyz = (numpy.linspace(-ex, ex, G), numpy.linspace(-ex, ex, G), numpy.linspace(-(9.5 - 5 / G), 0.5 - 5 / G, G))
    plot_voxels(ax, new_geometry[idx].reshape((G, G, G), order='F'), 'b', False, False, xyz)
    ax.set_xlabel('x ($m$)')
    ax.set_ylabel('y ($m$)')
    ax.set_zlabel('z ($m$)')

    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(pkg_resources.resource_filename("WAnet", "figures/BIEM_example_geometry.png"), dpi=1000)


