import AMnet
import pkg_resources
import keras
import sklearn
import numpy

variables = ['mass', 'support_material', 'print_time']

for var in variables:

    temp = open(pkg_resources.resource_filename('AMnet', 'trained_models/8_'+var+'_forward_structure.yml'), 'r')
    geo = keras.models.model_from_yaml(temp.read())
    geo.load_weights(pkg_resources.resource_filename('AMnet', 'trained_models/8_'+var+'_forward_weights.h5'))

    geometry, mass, support_material, print_time, flattened_geometry, N, G = AMnet.utilities.load_data()

    variable_of_interest = eval(var)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(flattened_geometry, variable_of_interest, shuffle=False)

    y_pred = geo.predict(x_test)

    x = []
    for i in range(len(y_pred)):
        x.append([y_pred[i], y_test[i]])

    numpy.savetxt(var+'.csv', numpy.array(x), delimiter=',')