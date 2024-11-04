from ThermoPINNs import *
import torch

def load_setup(filename='config.yaml'):
    from datetime import datetime
    ghp_save_folder = 'network_%s' % (datetime.now().strftime("%Y%m%d_%H%M%S"))

    parser = parse_args()
    para_arg = read_yaml_file(filename)
    parser.set_defaults(**para_arg)
    arg = parser.parse_args()

    sampling_seed_ = arg.sampling_seed

    # Number of training+validation points
    n_coll_ = arg.n_coll
    n_u_ = arg.n_u
    n_int_ = arg.n_int  # internal points = data support

    # Additional Info
    folder_path_ = ghp_save_folder
    point_ = arg.point_type  # define what point distribution to use
    validation_size_ = arg.n_valid
    network_properties_ = {
        "dict_name": "Network properties",
        "hidden_layers": arg.n_nn_lay,
        "neurons": arg.n_nn_neu,
        "residual_parameter": arg.icbc_w,  # weight of the init+boundary function in Loss function
        "kernel_regularizer": arg.reg_exp,  # what kind of regularization L#
        "regularization_parameter": arg.reg_w,  # how strong is regularization
        "batch_size": (n_coll_ + n_u_ + n_int_), # use full batch for training, else the performance is pretty bad
        "epochs": arg.epoch,
        "max_iter": arg.max_iter,
        "activation": arg.act,  # tanh,#sin... needs to be 2 times differentiable
        "optimizer": arg.optim  # ADAM
    }
    retrain_ = arg.retrain_seed

    shuffle_ = arg.shuffle_set

    if arg.restart_folder == '':
        restart_folder = []
    else:
        restart_folder = arg.restart_folder
    device_type = arg.device
    verbose = arg.verbose_set
    return sampling_seed_, n_coll_, n_u_, n_int_, folder_path_, point_, validation_size_, network_properties_, retrain_, shuffle_, ghp_save_folder, device_type, restart_folder, verbose


def pinns_run():
    sampling_seed, N_coll, N_u, N_int, folder_path, point, validation_size, network_properties, retrain, shuffle, ghp_save_folder, device_type, restart_folder, verbose = load_setup()

    Ec = EquationClass()
    if device_type == 'cuda':
        Ec.device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    elif device_type == 'cpu':
        Ec.device = torch.device("cpu")

    Ec.type_of_points=point
    if Ec.extrema_values is not None:
        extrema = Ec.extrema_values
        space_dimensions = Ec.space_dimensions
        time_dimension = Ec.time_dimensions
        parameter_dimensions = Ec.parameter_dimensions

        print(space_dimensions, time_dimension, parameter_dimensions)
    else:
        print("Using free shape. Make sure you have the functions:")
        print("     - add_boundary(n_samples)")
        print("     - add_collocation(n_samples)")
        print("in the Equation file")

        extrema = None
        space_dimensions = Ec.space_dimensions
        time_dimension = Ec.time_dimensions
    try:
        parameters_values = Ec.parameters_values
        parameter_dimensions = parameters_values.shape[0]
    except AttributeError:
        print("No additional parameter found")
        parameters_values = None
        parameter_dimensions = 0

    input_dimensions = parameter_dimensions + time_dimension + space_dimensions
    output_dimension = Ec.output_dimension
    mode = "none"
    if network_properties["epochs"] != 1:
        max_iter = 1
    else:
        max_iter = network_properties["max_iter"]
    print(max_iter)
    N_u_train = int(N_u * (1 - validation_size))
    N_coll_train = int(N_coll * (1 - validation_size))
    N_int_train = int(N_int * (1 - validation_size))
    N_train = N_u_train + N_coll_train + N_int_train

    if space_dimensions > 0:
        N_b_train = int(N_u_train / (4 * space_dimensions))
        # N_b_train = int(N_u_train / (1 + 2 * space_dimensions))
    else:
        N_b_train = 0
    if time_dimension == 1:
        N_i_train = N_u_train - 2 * space_dimensions * N_b_train
        # N_i_train = N_u_train - N_b_train*(2 * space_dimensions)
    elif time_dimension == 0:
        N_b_train = int(N_u_train / (2 * space_dimensions))
        N_i_train = 0
    else:
        raise ValueError()

    print("\n######################################")
    print("*******Domain Properties********")
    print(extrema)

    print("\n######################################")
    print("*******Info Training Points********")
    print("Number of train collocation points: ", N_coll_train)
    print("Number of initial and boundary points: ", N_u_train, N_i_train, N_b_train)
    print("Number of internal points: ", N_int_train)
    print("Total number of training points: ", N_train)

    print("\n######################################")
    print("*******Network Properties********")
    pprint.pprint(network_properties)
    batch_dim = network_properties["batch_size"]

    print("\n######################################")
    print("*******Dimensions********")
    print("Space Dimensions", space_dimensions)
    print("Time Dimension", time_dimension)
    print("Parameter Dimensions", parameter_dimensions)
    print("\n######################################")

    if network_properties["optimizer"] == "LBFGS" and network_properties["epochs"] != 1 and network_properties["max_iter"] == 1 and (batch_dim == "full" or batch_dim == N_train):
        print(bcolors.WARNING + "WARNING: you set max_iter=1 and epochs=" + str(network_properties["epochs"]) + " with a LBFGS optimizer.\n"
                                                                                                                "This will work but it is not efficient in full batch mode. Set max_iter = " + str(network_properties["epochs"]) + " and epochs=1. instead" + bcolors.ENDC)

    if batch_dim == "full":
        batch_dim = N_train

    # #############################################################################################################################################################
    # Dataset Creation
    training_set_class = DefineDataset(Ec, N_coll_train, N_b_train, N_i_train, N_int_train, batches=batch_dim, random_seed=sampling_seed, shuffle=shuffle)
    training_set_class.assemble_dataset()

    # #############################################################################################################################################################
    # Model Creation
    additional_models = None
    model = Pinns(input_dimension=input_dimensions, output_dimension=output_dimension, network_properties=network_properties)

    # #############################################################################################################################################################
    # Weights Initialization
    torch.manual_seed(retrain)
    init_xavier(model)
    print(torch.cuda.is_available(),Ec.device)
    if restart_folder:
        if restart_folder=='last':
            network_folders = [x for x in os.listdir('./') if 'network' in x]
            folder_num = [int(x[-15:-7]+x[-6:]) for x in network_folders]
            max_num = np.array(folder_num).max()
            old_folder = 'network_'+str(max_num)[:-6]+'_'+str(max_num)[-6:]
            old_model_state_dict = torch.load(os.path.join(old_folder, 'model_state.pth'))
            print('Loaded network from last training attempt.')
        else:
            old_folder = restart_folder
            print(old_folder)
            old_model_state_dict = torch.load(os.path.join(old_folder, 'model_state.pth'))#, map_location=torch.device('cpu')).to(Ec.device)
            print('Loaded network from folder %s.'%(old_folder))
        model.load_state_dict(old_model_state_dict, strict=False)
    else:
        old_folder = ''
        print('Loaded random weights and biases for the model.')


    # #############################################################################################################################################################
    # Model Training
    start = time.time()
    print("Fitting Model")
    model.to(Ec.device)
    model.train()
    optimizer_LBFGS = optim.LBFGS(model.parameters(), lr=0.8, max_iter=max_iter, max_eval=50000, history_size=100,
                                  line_search_fn="strong_wolfe",
                                  tolerance_change=1.0 * np.finfo(float).eps)
    optimizer_ADAM = optim.Adam(model.parameters(), lr=0.0005)

    if network_properties["optimizer"] == "LBFGS":
        model.optimizer = optimizer_LBFGS
    elif network_properties["optimizer"] == "ADAM":
        model.optimizer = optimizer_ADAM
    else:
        raise ValueError()

    if N_coll_train != 0:
        errors = fit(Ec, model, training_set_class, verbose)
    else:
        errors = StandardFit(Ec, model, training_set_class, verbose)
    end = time.time() - start
    print("\nTraining Time: ", end)

    model = model.eval()
    final_error_train = float(((10 ** errors[0]) ** 0.5).detach().cpu().numpy())
    error_vars = float((errors[1]).detach().cpu().numpy())
    error_pde = float((errors[2]).detach().cpu().numpy())
    print("\n################################################")
    print("Final Training Loss:", final_error_train)
    print("################################################")

    # #############################################################################################################################################################
    # Plotting ang Assessing Performance
    model_path = folder_path

    if(not os.path.exists(folder_path)):
        os.mkdir(folder_path)
    L2_test=0
    rel_L2_test=0

    dump_to_file(model, model_path, old_folder, N_u_train, N_coll_train, N_int_train, validation_size, end, final_error_train, error_vars, error_pde, network_properties, ghp_save_folder)


if __name__ == "__main__":
    parser = parse_args()
    arg = parser.parse_args()
    write_yaml_file('../test_run/config.yaml', arg)
    #pinns_run()
