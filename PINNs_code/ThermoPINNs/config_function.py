import yaml
import argparse
import torch
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restart_folder', type=str, default='',
                        help='Initial PINNs model from scratch or previous model:'
                             'Assign [] to start training from scratch'
                            'Assign [last] to restart training from last training attempt'
                            'Assign [network_20220409_124347] to restart training from the chosen folder')
    parser.add_argument('--device', type=str, default='cuda',
                        help='training device (cuda or cpu)')

    parser.add_argument('--n_coll', type=int, default=1000,
                        help='Number of collocation points')
    parser.add_argument('--n_u', type=int, default=500,
                        help='Number of boundary and initial conditions')
    parser.add_argument('--n_int', type=int, default=0,
                        help='Number of support data point')
    parser.add_argument('--point_type', type=str, default="moving_center",
                        help='collocation point strategy')
    parser.add_argument('--n_valid', type=int, default=0,
                        help='Number of validation point')

    parser.add_argument('--n_nn_lay', type=int, default=8,
                        help='Number of validation point')
    parser.add_argument('--n_nn_neu', type=int, default=32,
                        help='Number of validation point')
    parser.add_argument('--icbc_w', type=float, default=5000.,
                        help='Weight factor for the initial and boundary conditions')
    parser.add_argument('--reg_exp', type=int, default=2,
                        help='Type of regularization')
    parser.add_argument('--reg_w', type=float, default=0.0,
                        help='Weight factor for the regularization term')
    parser.add_argument('--epoch', type=int, default=1,
                        help='Epoch number')
    parser.add_argument('--max_iter', type=int, default=50000,
                        help='max_iteration for LBFGs optim')
    parser.add_argument('--act', type=str, default="sin",
                        help='activation function')
    parser.add_argument('--optim', type=str, default="LBFGS",
                        help='Optimizer')

    parser.add_argument('--sampling_seed', type=int, default=128,
                        help='Random seed for sampling')
    parser.add_argument('--retrain_seed', type=int, default=32,
                        help='Random seed for training')
    parser.add_argument('--shuffle_set', type=bool, default=False,
                        help='Shuffle the dataset or not')
    parser.add_argument('--verbose_set', type=bool, default=False,
                        help='Print out loss during training or not')
    return parser


def write_yaml_file(file_name: str, data: dict):
    # Write the dictionary to a YAML file
    with open(file_name, 'w') as file:
        yaml.dump(data, file)


def read_yaml_file(file_name: str) -> dict:
    # Read the YAML file
    with open(file_name, 'r') as file:
        loaded_data = yaml.safe_load(file)
    return loaded_data

def dump_to_file(model, model_path, old_folder, N_u_train, N_coll_train, N_int_train, validation_size, end, final_error_train, error_vars, error_pde, network_properties, ghp_save_folder):
    torch.save(model, os.path.join(model_path,'model_full.pkl'))
    torch.save(model.state_dict(), os.path.join(model_path,'model_state.pth'))
    train_info_dict = {
        "dict_name": "Training Information",
        "restart_folder":old_folder,
        "Nu_train": N_u_train,
        "Nf_train": N_coll_train,
        "Nint_train": N_int_train,
        "validation_size": validation_size,
        "train_time": end,
        "error_train": final_error_train,
        "error_vars": error_vars,
        "error_pde": error_pde,
    }
    write_overview([network_properties,train_info_dict], os.path.join(ghp_save_folder, 'overview.txt'))


# Write overview of current run
def write_overview(dict_list, overview_file):
    with open(overview_file,'w') as file:
        for dict in dict_list:
            file.write(dict['dict_name']+'\n')
            for key, value in dict.items():
                if key != 'dict_name':
                    file.write('\t%s: %s\n'%(key,value.__str__()))
            file.write('\n')