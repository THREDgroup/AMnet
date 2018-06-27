import AMnet

PRE_PROCESS = not True
TRAIN = not True
EXAMPLES = True
QUICK = False

# Set variables
# latent_dims = [pow(2, y) for y in range(1, 8)]
latent_dims = [8]
example_size = (3, 3)

if PRE_PROCESS:
    AMnet.preprocessing.extract_data('050')
    AMnet.preprocessing.augment_data()

# Train all the models
if TRAIN:
    r2_autoencoding = []
    r2_support_material = []
    r2_mass = []
    r2_print_time = []
    for latent_dim in latent_dims:
        print(latent_dim)
        # r2_autoencoding.append(AMnet.training.variational_autoencoder(10, latent_dim, True, True))
        r2_mass.append(AMnet.training.train_forward_network(20, latent_dim, True, True, 'mass', load_previous=True))
        r2_support_material.append(AMnet.training.train_forward_network(20, latent_dim, True, True, 'support_material', load_previous=True))
        r2_print_time.append(AMnet.training.train_forward_network(20, latent_dim, True, True, 'print_time', load_previous=True))

    print(r2_autoencoding)
    print(r2_mass)
    print(r2_support_material)
    print(r2_print_time)

if EXAMPLES:
    AMnet.showing.plot_random_autoencoder_examples(str(latent_dims[0])+"geometry_autoencoder", example_size[0], example_size[1], quick=QUICK)
    # AMnet.showing.plot_examples_for_data_augmentation(72)
