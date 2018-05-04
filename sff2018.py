import AMnet

PRE_PROCESS = False
TRAIN = True
EXAMPLES = False
QUICK = False

# Set variables
latent_dims = [32]
example_size = (2, 3)

if PRE_PROCESS:
    AMnet.preprocessing.extract_data('/Users/ccm/Box Sync/ML + AM/Voxelized GE Files/Voxelized_GE_Files_50/')
    AMnet.preprocessing.augment_data()

# Train all the models
if TRAIN:
    for latent_dim in latent_dims:
        # r2_encoding = AMnet.training.variational_autoencoder(2, latent_dim, True, True)
        # r2_prediction = AMnet.training.train_forward_network(25, latent_dim, True, True, load_previous=True)
        # r2_encoding = AMnet.training.convolutional_autoencoder(20, True)
        r2_prediction = AMnet.training.train_convolutional_forward_network(10, latent_dim, True, True, load_previous=True)

if EXAMPLES:
    AMnet.showing.plot_autoencoder_examples(str(latent_dims[0])+"geometry_autoencoder", example_size[0], example_size[1], quick=QUICK)