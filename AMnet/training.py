import keras
import sklearn.model_selection
import numpy
import pkg_resources
import os
import scipy.io

VERBOSE = 0

def load_data():

    path_to_data = os.path.join('Users', 'ccm' 'Box Sync', 'ML+AM', 'Voxelized GE Files', 'Voxelized_GE_Files_10')

    file_list = [f for f in os.listdir(path_to_data) if os.path.isfile(os.path.join(path_to_data, f))]
    for file in file_list:
        dat = scipy.io.loadmat(os.path.join(path_to_data, file))

    return geometry, volume, flattened_geometry, N, G


def train_geometry_autoencoder(epochs, latent_dim, save_results, print_network):
    geometry, volume, flattened_geometry, N, G = load_data()

    batch_size = 100
    original_dim = G*G*G
    intermediate_dim = 256
    epsilon_std = 1.0

    x = keras.layers.Input(shape=(original_dim,))
    h = keras.layers.Dense(intermediate_dim, activation='relu')(x)
    z_mean = keras.layers.Dense(latent_dim)(h)
    z_log_var = keras.layers.Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = keras.backend.random_normal(shape=(keras.backend.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + keras.backend.exp(z_log_var / 2) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = keras.layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_h = keras.layers.Dense(intermediate_dim, activation='relu')
    decoder_mean = keras.layers.Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # Custom loss layer
    class CustomVariationalLayer(keras.layers.Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)

        def vae_loss(self, x, x_decoded_mean):
            xent_loss = original_dim * keras.metrics.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * keras.backend.sum(1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var), axis=-1)
            return keras.backend.mean(xent_loss + kl_loss)

        def call(self, inputs, **kwargs):
            x = inputs[0]
            x_decoded_mean = inputs[1]
            loss = self.vae_loss(x, x_decoded_mean)
            self.add_loss(loss, inputs=inputs)
            # We won't actually use the output.
            return x

    y = CustomVariationalLayer()([x, x_decoded_mean])
    vae = keras.models.Model(x, y)
    vae.compile(optimizer='rmsprop', loss=None)

    x_train, x_test = sklearn.model_selection.train_test_split(flattened_geometry, shuffle=False)

    weights = pkg_resources.resource_filename('WAnet', 'trained_models/'+str(latent_dim)+'temp_vae_weights.h5')
    logger = pkg_resources.resource_filename('WAnet', 'trained_models/'+str(latent_dim)+'geometry_vae_training.csv')
    vae.fit(x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None),
            verbose=VERBOSE,
            callbacks=[keras.callbacks.ModelCheckpoint(filepath=weights, verbose=VERBOSE, save_best_only=True),
                       keras.callbacks.CSVLogger(logger, separator=',', append=False)])

    vae.load_weights(weights)
    os.remove(weights)

    # build a model to project inputs on the latent space
    encoder = keras.models.Model(x, z_mean)

    # build a digit generator that can sample from the learned distribution
    decoder_input = keras.layers.Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = keras.models.Model(decoder_input, _x_decoded_mean)

    # Build and save the autoencoder
    _h_decoded2 = decoder_h(z_mean)
    _x_decoded_mean2 = decoder_mean(_h_decoded2)
    autoencoder = keras.models.Model(x, _x_decoded_mean2)

    structure = []
    weights = []
    if save_results:
        # Save encoder structure and weights
        temp = open(pkg_resources.resource_filename('WAnet', 'trained_models/'+str(latent_dim)+'geometry_encoder_structure.yml'), 'w')
        temp.write(encoder.to_yaml())
        temp.close()
        encoder.save_weights(pkg_resources.resource_filename('WAnet', 'trained_models/'+str(latent_dim)+'geometry_encoder_weights.h5'))

        # Save decoder structure and weights
        temp = open(pkg_resources.resource_filename('WAnet', 'trained_models/'+str(latent_dim)+'geometry_decoder_structure.yml'), 'w')
        temp.write(generator.to_yaml())
        temp.close()
        generator.save_weights(pkg_resources.resource_filename('WAnet', 'trained_models/'+str(latent_dim)+'geometry_decoder_weights.h5'))

        # Save full autoencoder structure and weights
        structure = pkg_resources.resource_filename('WAnet', 'trained_models/'+str(latent_dim)+'geometry_autoencoder_structure.yml')
        weights = pkg_resources.resource_filename('WAnet', 'trained_models/'+str(latent_dim)+'geometry_autoencoder_weights.h5')
        temp = open(structure, 'w')
        temp.write(autoencoder.to_yaml())
        temp.close()
        autoencoder.save_weights(weights)

    if print_network:
        keras.utils.plot_model(generator, to_file=pkg_resources.resource_filename('WAnet', 'figures/'+str(latent_dim)+'geometry_decoder.eps'), show_shapes=True)
        keras.utils.plot_model(encoder, to_file=pkg_resources.resource_filename('WAnet', 'figures/'+str(latent_dim)+'geometry_encoder.eps'), show_shapes=True)
        keras.utils.plot_model(autoencoder, to_file=pkg_resources.resource_filename('WAnet', 'figures/'+str(latent_dim)+'geometry_autoencoder.eps'), show_shapes=True)

    # Final check on metrics
    x_pred = autoencoder.predict(x_test)
    mse = keras.backend.mean(keras.losses.binary_crossentropy(x_pred, x_test)).eval()
    x_pred.fill(numpy.mean(x_test.flatten()))
    s2 = keras.backend.mean(keras.losses.binary_crossentropy(x_pred, x_test)).eval()
    r2 = 1-mse/s2
    print("Final BCE: "+str(mse))
    print("Final S2: "+str(s2))
    print("Final R2: "+str(r2))
    return r2

def train_forward_network(epochs, latent_dim, save_results, print_network):
    curves, geometry, S, N, D, F, G, flattened_curves, flattened_geometry = load_data()

    # Define model
    x   = keras.layers.Input(shape=(32768,))
    de1 = keras.layers.Dense(256, activation='relu')(x)
    de2 = keras.layers.Dense(latent_dim, activation='relu')(de1)
    con = keras.layers.Dense(latent_dim, activation='relu')(de2)
    dd2 = keras.layers.Dense(64, activation='relu')(con)
    y   = keras.layers.Dense(192, activation='sigmoid')(dd2)

    # Build and compile ,model
    mdl = keras.models.Model(x, y)
    mdl.compile(optimizer='rmsprop', loss='mse')

    # # Instantiate and freeze layers if possible
    temp = open(pkg_resources.resource_filename('WAnet', 'trained_models/'+str(latent_dim)+'geometry_encoder_structure.yml'), 'r')
    geo = keras.models.model_from_yaml(temp.read())
    geo.load_weights(pkg_resources.resource_filename('WAnet', 'trained_models/'+str(latent_dim)+'geometry_encoder_weights.h5'))

    # Load curve autoencoder
    temp = open(pkg_resources.resource_filename('WAnet', 'trained_models/'+str(latent_dim)+'curve_decoder_structure.yml'), 'r')
    curve = keras.models.model_from_yaml(temp.read())
    curve.load_weights(pkg_resources.resource_filename('WAnet', 'trained_models/'+str(latent_dim)+'curve_decoder_weights.h5'))

    mdl.layers[1].set_weights(geo.layers[1].get_weights())
    mdl.layers[1].trainable = False
    mdl.layers[2].set_weights(geo.layers[2].get_weights())
    mdl.layers[2].trainable = False

    mdl.layers[4].set_weights(curve.layers[1].get_weights())
    mdl.layers[4].trainable = False
    mdl.layers[5].set_weights(curve.layers[2].get_weights())
    mdl.layers[5].trainable = False

    # Make file names
    weights = pkg_resources.resource_filename('WAnet', 'trained_models/'+str(latent_dim)+'forward_weights.h5')
    structure = pkg_resources.resource_filename('WAnet', 'trained_models/'+str(latent_dim)+'forward_structure.yml')
    plot = pkg_resources.resource_filename('WAnet', 'figures/'+str(latent_dim)+'forward.eps')

    # Save model structure and start training
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(flattened_geometry, flattened_curves, shuffle=False)
    if save_results:
        mdl.fit(x_train, y_train, verbose=VERBOSE, epochs=epochs, shuffle=False, validation_data=(x_test, y_test),
                callbacks=[keras.callbacks.ModelCheckpoint(filepath=weights, verbose=VERBOSE, save_best_only=True)])

        # Save decoder structure and weights
        temp = open(structure, 'w')
        temp.write(mdl.to_yaml())
        temp.close()
    else:
        mdl.fit(flattened_geometry, flattened_curves, verbose=VERBOSE, epochs=epochs, shuffle=False, validation_data=(x_test, y_test))

    if print_network:
        keras.utils.plot_model(mdl, to_file=plot, show_shapes=True)

    #
    mdl.load_weights(weights)
    y_pred = mdl.predict(x_test)
    s2 = numpy.mean(numpy.power(numpy.mean(y_test.flatten()) - y_test.flatten(), 2))
    mse = keras.backend.mean(keras.losses.mean_squared_error(y_pred, y_test)).eval()
    r2 = 1-mse/s2
    print("Final MSE: "+str(mse))
    print("Final S2: "+str(s2))
    print("Final R2: "+str(r2))

    return r2