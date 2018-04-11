import AMnet

PRE_PROCESS = True
TRAIN = not True
EXAMPLES = not True
QUICK = False

# Set variables
latent_dims = [32]
example_size = (2, 3)

if PRE_PROCESS:
    AMnet.preprocessing.extract_data('/Users/ccm/Box Sync/ML + AM/Voxelized GE Files/Voxelized_GE_Files_50/')

# Train all the models
if TRAIN:
    for latent_dim in latent_dims:
        # r2_encoding = AMnet.training.variational_autoencoder(50, latent_dim, True, True)
        r2_prediction = AMnet.training.train_forward_network(10, latent_dim, True, True)

if EXAMPLES:
    AMnet.showing.plot_examples(str(latent_dims[0])+"geometry_autoencoder", example_size[0], example_size[1], quick=QUICK)
    # AMnet.showing.plot_examples("forward", example_size[0], example_size[1], quick=QUICK)
