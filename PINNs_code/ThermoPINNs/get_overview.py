from genericpath import isdir
import os
import csv

# path to network files
exp_path = '..'  #r"C:\Users\Philip Wolf\polybox\Semster_Thesis\PINN_new\Networks\network_20221227\Exp65"
#exp_path = r"C:\Users\woph\polybox\PINN_new\Networks\network_20221128\Exp33"
os.path.normpath(exp_path)

# path to output file
out_path = exp_path + '/net_overview.txt'

# data lists
net_name = ['Network name']
err_train = ['Error train']
err_vars = ['Error vars']
err_pde = ['Error PDE']
parameters = 'initial'

count = 0
if len(next(os.walk(exp_path))) > 0:

    for item in os.listdir(exp_path):
        net_path = exp_path + '/' + item
        if os.path.isdir(net_path):
            if 'network' in item:
                net_name.append(item)

                with open(net_path+'/overview.txt', 'r') as f:
                    content = f.read()
                    
                    for line in content.split('\n'):
                        if 'error_train:' in line:
                            line = line.replace("\t", '')
                            err_train.append(float(line.replace("error_train: ", '')))
                        
                        elif 'error_vars:' in line:
                            line = line.replace("\t", '')
                            err_vars.append(float(line.replace("error_vars:", '')))
                        
                        elif 'error_pde:' in line:
                            line = line.replace("\t",'')
                            err_pde.append(float(line.replace("error_pde:", '')))

                    if count == 0:
                        for paragraph in content.split('\n\n'):
                            if 'Parameter values' in paragraph:
                                parameters = paragraph.replace('parameter_values: tensor', '')
                        
                        count = 1

                

    id = list(range(0, len(net_name)-1))
    id.insert(0,'ID')

    with open(out_path, 'w') as out_file:
        for x in zip(id,net_name, err_train, err_vars, err_pde):
            out_file.write("{0} {1}\t{2}\t{3}\t{4}\n".format(*x))
        
        out_file.write('\n\n')
        out_file.write(parameters)

    print('Network overview successfuly written.')

else:
    raise('Error: no networks found')
