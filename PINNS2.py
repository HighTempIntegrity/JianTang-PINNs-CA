from ImportFile import *

torch.manual_seed(42)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#################### Pooriya mods # start ###########################
# Folder to restart
restart_folder = ['network20230530_023140']
device_type = 'cuda' # cuda|cpu; if you choose cuda the code will check for gpu availability
# Assign [] to start training from scratch
# Assign ['base'] to restart training from Ole's base model
# Assign ['last'] to restart training from last training attempt
# Assign ['network_date_time'network_20220409_124347] to restart training from the chosen folder
'''
with open('Paper_Equation_tempDep.py', 'r') as file:
    data = file.readlines()
  
import re

def use_regex(input_text):
    pattern = re.compile(r"                                               \[([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?, ([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?\],  # conductivity/1000", re.IGNORECASE)
    return pattern.match(input_text)
krange = use_regex(data[19])
print(krange.group(1),krange.group(2),krange.group(3))
replace_text = "                                               [{minn}, {maxn}],  # conductivity/1000\n".format(minn=str(round(float(krange.group(1))-0.005,3)),maxn=str(round(float(krange.group(3))+0.01,3)))

data[19] = replace_text
  
with open('Paper_Equation_tempDep.py', 'w') as file:
    file.writelines(data)

'''
from datetime import datetime
ghp_save_folder = 'network%s' % (datetime.now().strftime("%Y%m%d_%H%M%S"))

# Write overview of current run
def write_overview(dict_list, overview_file):
    with open(overview_file,'w') as file:
        for dict in dict_list:
            file.write(dict['dict_name']+'\n')
            for key, value in dict.items():
                if key != 'dict_name':
                    file.write('\t%s: %s\n'%(key,value.__str__()))
            file.write('\n')
#################### Pooriya mods # end #############################

def dump_to_file():
    torch.save(model, os.path.join(model_path,'model_full.pkl'))
    torch.save(model.state_dict(), os.path.join(model_path,'model_state.pth'))
    #torch.save(Ec.heat, 'heat.pt')
    train_info_dict = {
        "dict_name": "Training Information",
        "restart_folder":old_folder,
        "Nu_train": N_u_train,
        "Nf_train": N_coll_train,
        "Nint_train": N_int_train,
        "validation_size": validation_size,
        "train_time": end,
        "L2_norm_test": L2_test,
        "rel_L2_norm": rel_L2_test,
        "error_train": final_error_train,
        "error_vars": error_vars,
        "error_pde": error_pde,
    }
    parameter_info_dict = {
        "dict_name": "Parameter values",
        "parameter_values" : Ec.parameters_values,
    }
    write_overview([network_properties,train_info_dict,parameter_info_dict], os.path.join(ghp_save_folder,'overview.txt'))

def initialize_inputs(len_sys_argv):
    
    if len_sys_argv == 1:

        # Random Seed for sampling the dataset
        sampling_seed_ = 128

        # Number of training+validation points
        n_coll_ = 2**19
        n_u_ = n_coll_/2
        n_int_ =0 # internal points = data support

        # Additional Info
        folder_path_ = ghp_save_folder
     
        point_ = "moving_center"  # define what point distribution to use
        validation_size_ = 0.0  # useless
        network_properties_ = {
            "dict_name": "Network properties",
            "hidden_layers": 10,
            "neurons": 24,
            "residual_parameter": 1, # weight of the init+boundary function in Loss function
            "kernel_regularizer": 2,   # what kind of regularization L#
            "regularization_parameter": 0, # how strong is regularization
            "batch_size": (n_coll_ + n_u_ + n_int_),
            "epochs": 1,
            "max_iter": 50000,
            "activation": "sin", #tanh,#sin... needs to be 2 times differentiable
            "optimizer": "LBFGS"  # ADAM
        }
        retrain_ = 32

        shuffle_ = False

    elif len_sys_argv == 13:
        print(sys.argv)
        # Random Seed for sampling the dataset
        sampling_seed_ = int(sys.argv[1])

        # Number of training+validation points
        n_coll_ = int(sys.argv[2])
        n_u_ = int(sys.argv[3])
        n_int_ = int(sys.argv[4])

        # Additional Info
        folder_path_ = sys.argv[7]
        point_ = sys.argv[8]
        print(sys.argv[9])
        validation_size_ = float(sys.argv[9])

        json_string = str(sys.argv[10])
        if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
            print()
        else:
            json_string = json_string.replace(', ', '\", \"')
            json_string = json_string.replace(': ', '\" :\"')
            json_string = json_string.replace('{', '{\"')
            json_string = json_string.replace('}', '\"}')
            json_string = json_string.replace('\'', '')
        network_properties_ = json.loads(json_string)
        network_properties_["hidden_layers"]=int(network_properties_["hidden_layers"])
        network_properties_["neurons"]=int(network_properties_["neurons"])
        network_properties_["residual_parameter"]=float(network_properties_["residual_parameter"])
        network_properties_["kernel_regularizer"]=int(network_properties_["kernel_regularizer"])
        network_properties_["regularization_parameter"]=float(network_properties_["regularization_parameter"])
        network_properties_["batch_size"]=int(network_properties_["batch_size"])
        network_properties_["epochs"]=int(network_properties_["epochs"])
        network_properties_["max_iter"]=int(network_properties_["max_iter"])

        retrain_ = sys.argv[11]
        if sys.argv[12] == "false":
            shuffle_ = False
        else:
            shuffle_ = True
    else:
        raise ValueError("One input is missing, I only have ", len_sys_argv )
    print(network_properties_)
    return sampling_seed_, n_coll_, n_u_, n_int_, folder_path_, point_, validation_size_, network_properties_, retrain_, shuffle_


sampling_seed, N_coll, N_u, N_int, folder_path, point, validation_size, network_properties, retrain, shuffle = initialize_inputs(len(sys.argv))

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
    max_iter = network_properties["max_iter"]
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

if restart_folder:
    if restart_folder[0]=='last':
        network_folders = [x for x in os.listdir('./') if 'network' in x]
        old_folder = network_folders[-1]
        old_model_state_dict = torch.load(os.path.join(old_folder, 'model_state.pth'))
        print('Loaded network from last training attempt.')
    elif restart_folder[0]=='base':
        old_folder = 'oles_base'
        old_model_state_dict = torch.load(os.path.join(old_folder, 'model_base.pkl')).state_dict()
    else:
        old_folder = restart_folder[0]
        old_model_state_dict = torch.load(os.path.join(old_folder, 'model_state.pth'))
        print('Loaded network from folder %s.'%(old_folder))
    model.load_state_dict(old_model_state_dict, strict=False)
else:
    old_folder = ''
    print('Loaded random weights and biases for the model.')

# print(model.coeff_list)
# #############################################################################################################################################################
# Model Training
start = time.time()
print("Fitting Model")
model.to(Ec.device)
model.train()
optimizer_LBFGS = optim.LBFGS(model.parameters(), lr=0.5, max_iter=max_iter, max_eval=50000, history_size=100,
                              line_search_fn="strong_wolfe",
                              tolerance_change=1.0 * np.finfo(float).eps)

if network_properties["optimizer"] == "LBFGS":
    model.optimizer = optimizer_LBFGS

else:
    raise ValueError()


errors = fit(Ec, model, training_set_class, verbose=True)

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
model_path = folder_path

if(not os.path.exists(folder_path)):
    os.mkdir(folder_path)
L2_test=0
rel_L2_test=0


dump_to_file()
