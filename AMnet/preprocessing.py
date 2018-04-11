import shutil
import WAnet.openwec
import numpy
import scipy.spatial
import sklearn.utils
import pkg_resources
import os


def generate_data():
    data_dir = pkg_resources.resource_filename('WAnet', 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save the current directory for later use
    save_dir = os.path.join(os.getcwd(), data_dir)

    # Define info for running the simulations
    minimum_frequency = 0.05
    maximum_frequency = 2.0
    frequency_steps = 64
    waterDepth = 100
    nPanels = 200
    number_of_random_draws = 1000
    rhoW = 1000.0
    zG = 0
    geometries = {
        "box": {
            "vars": {
                "length": [3, 10],
                "width":  [3, 10],
                "height": [3, 10]
            }
        },
        "cone": {
            "vars": {
                "diameter": [3, 10],
                "height":   [3, 10]
            }
        },
        "cylinder": {
            "vars": {
                "diameter": [3, 10],
                "height":   [3, 10]
            }
        },
        "sphere": {
            "vars": {
                "diameter": [3, 10]
            }
        },
        "wedge": {
            "vars": {
                "length": [3, 10],
                "width":  [3, 10],
                "height": [3, 7.5]
            }
        },
    }

    for shape in geometries:
        print(shape)

    # Run the simulations
    for shape_index, shape in enumerate(geometries):
        for i in range(number_of_random_draws):
            # Make the project directory and a directory to save things in
            temp_dir = os.path.join(save_dir, shape + str(i).zfill(3))
            if not os.path.exists(temp_dir):
                os.mkdir(temp_dir)
            WAnet.openwec.helpers.make_project_directory()

            # Dynamically define some variables, and save to file
            for var, limits in geometries[shape]["vars"].items():
                print(var + " = " + str(numpy.random.uniform(limits[0], limits[1])))
                exec(var + " = " + str(numpy.random.uniform(limits[0], limits[1])))
            numpy.savetxt(temp_dir + '/geometry.txt',
                          numpy.array(eval("[" + str(shape_index) + "," +
                                           ", ".join(geometries[shape]["vars"].keys()) + "]")))

            # Make the mesh
            print("meshTypes."+shape+"("+", ".join(geometries[shape]["vars"].keys())+", [0, 0, 0])")
            msh = eval("WAnet.openwec.meshTypes."+shape+"("+", ".join(geometries[shape]["vars"].keys())+", [0, 0, 0])")
            msh.panelize()
            WAnet.openwec.meshTypes.writeMesh(msh,
                                             os.path.join(os.path.join(os.path.expanduser('~'), 'openWEC'),
                                                          'Calculation', 'mesh', 'axisym'))
            WAnet.openwec.nemoh.createMeshOpt([msh.xC, msh.yC, zG], nPanels, int(0), rhoW)

            # Run Nemoh on the mesh
            advOps = {
                'dirCheck': False,
                'irfCheck': False,
                'kochCheck': False,
                'fsCheck': False,
                'parkCheck': False
            }
            nbody = WAnet.openwec.nemoh.writeCalFile(rhoW, waterDepth,
                                                    [frequency_steps, minimum_frequency, maximum_frequency],
                                                    zG, [1, 0, 1, 0, 1, 0], aO=advOps)
            WAnet.openwec.nemoh.runNemoh(nbody)

            # Copy out what is needed
            shutil.copy(os.path.join(os.path.expanduser("~"), 'openWEC/Calculation/axisym.dat'), temp_dir)
            shutil.copy(os.path.join(os.path.expanduser("~"), 'openWEC/Calculation/Nemoh.cal'), temp_dir)
            shutil.copy(os.path.join(os.path.expanduser("~"), 'openWEC/Calculation/results/RadiationCoefficients.tec'), temp_dir)
            shutil.copy(os.path.join(os.path.expanduser("~"), 'openWEC/Calculation/results/DiffractionForce.tec'), temp_dir)
            shutil.copy(os.path.join(os.path.expanduser("~"), 'openWEC/Calculation/results/ExcitationForce.tec'), temp_dir)

            # Cleanup the project directory
            WAnet.openwec.helpers.clean_directory()

    return True


def extract_data(N=1000):
    # Define constants
    S = 5
    D = 3
    F = 64
    G = 32

    # Initialize some huge vectors
    curves = numpy.empty([S * N, F, 3])
    geometry = numpy.zeros([S * N, G, G, G, 1])

    # Set up test points
    ex = 5 - 5 / G
    x, y, z = numpy.meshgrid(numpy.linspace(-ex, ex, G),
                             numpy.linspace(-ex, ex, G),
                             numpy.linspace(-(9.5 - 5 / G), 0.5 - 5 / G, G))
    test_points = numpy.vstack((x.ravel(), y.ravel(), z.ravel())).T

    # Step through data
    current = 0
    nemoh_dir = pkg_resources.resource_filename('WAnet', 'data/NEMOH_data/')
    data = sklearn.utils.shuffle(os.listdir(pkg_resources.resource_filename('WAnet', 'data/NEMOH_data/')))
    for i in range(S * N):
        dd = data[i]
        print(dd)
        dir_path = os.path.join(nemoh_dir + dd)
        if os.path.isdir(dir_path):

            # Read in the hydrodynamic coefficients
            with open(dir_path + '/ExcitationForce.tec') as fid:
                current_f = 0
                for line in fid:
                    if line.find('"') is -1:
                        str_list = line.split(' ')
                        str_list = filter(None, str_list)
                        new_array = numpy.array([float(elem) for elem in str_list])
                        # curves[i, current_f, :] = new_array[1:]
                        curves[i, current_f, 0] = new_array[1]
                        curves[i, current_f, 1] = new_array[3]
                        curves[i, current_f, 2] = new_array[5]
                        current_f += 1

            # Read in existing vertices
            vertices = numpy.empty([0, 3])
            with open(dir_path + '/axisym.dat') as fid:
                for line in fid:
                    vert = numpy.array([float(elem) for elem in filter(None, line.split(' '))])
                    if sum(vert) == 0:
                        break
                    if len(vert) == 4:
                        vertices = numpy.vstack([vertices, vert[1:4]])

            # Jiggle to avoid memory issues with Delaunay below.
            vertices += 0.001 * numpy.random.random(vertices.shape)

            # Check points in hull of vertices
            hull = scipy.spatial.Delaunay(vertices)
            within = hull.find_simplex(test_points) >= 0

            # Stop if resolution is too low
            if sum(within) == 0:
                print("Bad!")
                break

            # maxer = np.max(vertices, axis=0)
            # miner = np.min(vertices, axis=0)
            # if maxer[2] > 0.25:
            #     print(dd, maxer[2])
            # if miner[2] < -5.25:
            #     print(dd, miner[2])

            # Reshape and save
            geometry[i, :, :, :, 0] = within.reshape((G, G, G))

    # Check that compiled_data exists
    sd = pkg_resources.resource_filename('WAnet', 'data/compiled_data')
    if not os.path.exists(sd):
        os.makedirs(sd)

    numpy.savez(pkg_resources.resource_filename('WAnet', 'data/compiled_data/data_geometry.npz'), geometry=geometry)
    numpy.savez(pkg_resources.resource_filename('WAnet', 'data/compiled_data/data_curves.npz'), curves=curves)
    numpy.savez(pkg_resources.resource_filename('WAnet', 'data/compiled_data/constants.npz'), S=S, N=N, D=D, F=F, G=G)

    return True
