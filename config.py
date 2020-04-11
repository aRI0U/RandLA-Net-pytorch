class cfg:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 40960  # Number of input points
    num_classes = 14  # Number of valid classes
    sub_grid_size = 0.04  # preprocess_parameter

    batch_size = 1  # batch_size during training
    val_batch_size = 1  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter
    max_epoch = 60  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None

    class_weights = [1938651, 1242339, 608870, 1699694, 2794560, 195000, 115990, 549838, 531470, 292971, 196633, 59032, 209046, 39321]
