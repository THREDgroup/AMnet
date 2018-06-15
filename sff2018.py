import AMnet

PRE_PROCESS = False
TRAIN = True
EXAMPLES = True
QUICK = False

# Set variables
latent_dims = [4]
example_size = (2, 3)

if PRE_PROCESS:
    AMnet.preprocessing.extract_data('050')
    AMnet.preprocessing.augment_data()

# Train all the models
if TRAIN:
    for latent_dim in latent_dims:
        r2_encoding = AMnet.training.variational_autoencoder(10, latent_dim, True, True)
        # r2_prediction = AMnet.training.train_forward_network(20, latent_dim, True, True, load_previous=False)

if EXAMPLES:
    # AMnet.showing.plot_random_autoencoder_examples(str(latent_dims[0])+"geometry_autoencoder", example_size[0], example_size[1], quick=QUICK)
    AMnet.showing.plot_autoencoder_examples_along_axis(str(latent_dims[0])+"geometry_decoder", 3, quick=QUICK)