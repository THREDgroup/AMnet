import keras
import AMnet.training
import numpy


class Network(object):

    def __init__(self, structure, weights):
        # Instantiate variables
        self.curves = 0
        self.new_curves = 0
        self.geometry = 0
        self.flattened_geometry = 0
        self.N = 0
        self.D = 0
        self.F = 0
        self.G = 0

        # Load network
        with open(structure, 'r') as file:
            self.network = keras.models.model_from_yaml(file.read())
            self.network.load_weights(weights)

        # Load data
        self._load_data()

    def _load_data(self):
        self.geometry, self.volume, self.flattened_geometry, self.N, self.G = AMnet.training.load_data()

    def prediction(self, idx=None):

        if idx is None:
            idx = numpy.random.randint(1, self.N)
            print(idx)

        # Get the input
        if self.network.layers[0].input_shape[1] == pow(self.G, 3):
            data_input = self.flattened_geometry[idx:(idx+1), :]
            other_data_input = data_input.reshape((self.G, self.G, self.G), order='F')
        else:
            data_input = self.new_curves[idx:(idx+1), :]
            other_data_input = data_input.reshape((3, self.F))

        # Get the outputs
        predicted_output = self.network.predict(data_input)
        if self.network.layers[-1].output_shape[1] == pow(self.G, 3):
            true_output = self.flattened_geometry[idx].reshape((self.G, self.G, self.G), order='F')
            predicted_output = predicted_output.reshape((self.G, self.G, self.G), order='F')
        else:
            true_output = self.new_curves[idx].reshape((3, self.F))
            predicted_output = predicted_output.reshape((3, self.F))

        return idx, other_data_input, true_output, predicted_output
