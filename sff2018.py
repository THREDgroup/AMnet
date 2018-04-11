import AMnet

TRAIN = True
EXAMPLES = False
QUICK = False

# Set variables
latent_dim = 16
example_size = (2, 3)

# Train all the models
if TRAIN:
    r2_encoding = AMnet.training.train_geometry_autoencoder(40, latent_dim, True, True))
    r2_prediction = AMnet.training.train_forward_network(25, latent_dim, True, True))

print(r2_r, r2_g, r2_f, r2_i)

if EXAMPLES:
    AMnet.showing.plot_examples("geometry_autoencoder", example_size[0], example_size[1], quick=QUICK)
    AMnet.showing.plot_examples("forward", example_size[0], example_size[1], quick=QUICK)
