import WAnet

PRE_PROCESS = False
TRAIN = True
EXAMPLES = False
QUICK = False

# Set variables
latent_dims = [2, 4, 8, 16, 32]
example_size = (2, 3)

# Preprocess to extract the data
if PRE_PROCESS:
    WAnet.preprocessing.extract_data(1000)

# Train all the models
r2_i = []
r2_g = []
r2_f = []
r2_r = []

if TRAIN:
    for latent_dim in latent_dims:
        r2_r.append(WAnet.training.train_response_autoencoder(100, latent_dim, True, True))
        r2_g.append(WAnet.training.train_geometry_autoencoder(40, latent_dim, True, True))
        r2_f.append(WAnet.training.train_forward_network(25, latent_dim, True, True))
        r2_i.append(WAnet.training.train_inverse_network(25, latent_dim, True, True))

print(r2_r, r2_g, r2_f, r2_i)

if EXAMPLES:
    WAnet.showing.plot_examples("geometry_autoencoder", example_size[0], example_size[1], quick=QUICK)
    WAnet.showing.plot_examples("curve_autoencoder", example_size[0], example_size[1], quick=QUICK)
    WAnet.showing.plot_examples("forward", example_size[0], example_size[1], quick=QUICK)
    WAnet.showing.plot_examples("inverse", example_size[0], example_size[1], quick=QUICK)
    WAnet.showing.plot_BIEM_example()
