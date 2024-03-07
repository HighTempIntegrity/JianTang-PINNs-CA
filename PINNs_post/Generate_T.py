import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata
from ImportFile import *
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def generate_mesh(length=0.001, file_name='_tmp'):
    # create grid in the horizontal cross section at x=0 and vertical cross-section at y=0
    domain = Ec.extrema_values[1:4]
    domain[0,0] = 0.348#-0.2
    domain[0,1] = 1.35#1.2
    domain[1,0] = -0.102#-0.3
    domain[1,1] = 0.102#0.3
    domain[2,0] = -0.174#-0.5
    print("THe domain is define as : ", domain.detach().numpy())
    nx = int((domain[0, 1] - domain[0, 0]).item() // length)+1
    ny = int((domain[1, 1] - domain[1, 0]).item() // length)+1
    nz = int((domain[2, 1] - domain[2, 0]).item() // length)+1

    x = torch.from_numpy(np.array(range(int(nx))) * length) + domain[0, 0]
    y = torch.from_numpy(np.array(range(int(ny))) * length) + domain[1, 0]
    z = torch.from_numpy(np.array(range(int(nz))) * length) + domain[2, 1] - nz*length

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
    grid_x = grid_x.reshape(-1, 1)
    grid_y = grid_y.reshape(-1, 1)
    grid_z = grid_z.reshape(-1, 1)
    grid_mesh = torch.cat([grid_x, grid_y, grid_z], axis=1)
    np.savetxt('./mesh_node_'+str(file_name)+'.csv',grid_mesh.numpy(), delimiter=',')
    return grid_mesh

def T_pred(Ec, model, grid_mesh, laser_power, scan_speed, goldak_opt, frame = 50, batch_size=1e5, file_name='_tmp', kfa=0.0, rra=0.015, rrb=0.005):

        T_norm=Ec.umax

        grid_size = grid_mesh.shape[0]
        print("The mesh contains %.1f nodes"%(grid_size))
        batch_size = int(batch_size)
        batch_num = int(grid_size//batch_size + 1) # to save the GPU memory, we seperate the data into batches
        duration = (torch.max(grid_mesh[:,0]).item() )/(scan_speed*1000)   # unit: s

        print("%.1f time frame with time increment %.6f ms will be output"%(frame, duration*1000/frame))

        # create grid in the horizontal cross section at x=0 and vertical cross-section at y=0
        goldak_opt = goldak_opt
        absorptivityP = laser_power * goldak_opt[-1]
        lp = torch.full(size=(grid_size, 1), fill_value=1.0) * absorptivityP
        ss = torch.full(size=(grid_size, 1), fill_value=1.0) * scan_speed
        kf = torch.full(size=(grid_size, 1),
                        fill_value=1.0) * kfa  # randn(size=(grid_size,1)).to(Ec.device) * 0.019 + 0.001  #
        rr = torch.randn(size=(grid_size, 1)) * rra + rrb  # full(size=(grid_size,1), fill_value=1.0).to(Ec.device) * 0.1

        goldak = goldak_opt * torch.full(
            size=(grid_size, goldak_opt.shape[0]), fill_value=1.0)


        tot_rec = []

        for t in np.arange(0.,1.+1./frame,1./frame):
            t_rec = np.array([t*torch.max(grid_mesh[:,0]).item()])
            grid_t = torch.full(size=(grid_size,1), fill_value=t)

            inputs = torch.cat([grid_t, grid_mesh, rr, kf, lp, ss, goldak[:, :-1]], axis=1).type(torch.FloatTensor)

            for batch_i in range(batch_num):
                if batch_i != (batch_num-1):
                    batch_inp = inputs[batch_i*batch_size:batch_i*batch_size+batch_size].to(Ec.device)
                    T = model(batch_inp)[:, 0].reshape(-1, ) * T_norm + 273.
                    t_rec = np.append(t_rec, T.cpu().detach().numpy())
                else:
                    batch_inp = inputs[batch_i*batch_size:].to(Ec.device)
                    T = model(batch_inp)[:, 0].reshape(-1, ) * T_norm + 273.
                    t_rec = np.append(t_rec, T.cpu().detach().numpy())
            #print(t_rec)
            #t_rec=np.array(t_rec).reshape(-1,1)
            tot_rec.append(t_rec)
        tot_rec = np.array(tot_rec)
        #print(tot_rec.shape)
        tot_rec = tot_rec.T[1:,8]

        np.savetxt('./Temperature_pred/Temperature_'+str(file_name)+'.dat', tot_rec, fmt='%.02f')
        print("Prediction for the following setting is done : ")
        print("P: %.3f W   v: %.3f mm/s Absorptivty:%.6f  " %(laser_power*1000,scan_speed*1000,goldak_opt[-1]))
        print("Goldak_parameters: ", list(goldak_opt.detach().numpy()))


if __name__ == "__main__":

    Ec = EquationClass()
    model_path = 'network_20230826_082323/'#network20230514_233329/'#network20230412_105024/'#network20230412_105024/'#
    model = torch.load(os.path.join(model_path, 'model_full.pkl'))
    
    if not os.path.exists('./Temperature_pred'):
        os.makedirs('./Temperature_pred')
    mesh_size = 0.003 # mesh size: mm
    frame = 10 #time frame
    batch_size = 1e6 # batch size for prediction

    grid_mesh = generate_mesh(mesh_size, file_name='tmp')

    Tm = 1307.
    laser_power = torch.Tensor([200,175,200,175,125])/1000.
    scan_speed = torch.Tensor([900,800,1100,900,700])/1000.

    absa = 2.0350
    absb = -0.1200
    aa = 1.003
    ab = -0.0410
    goldak_opt = torch.Tensor([0.0275,0.16813, 0.0236,0.1072, 0.65145])
    rra = 0.0
    rrb = 0.01
    kfa = 0.0035



    for i in range(5):
        goldak_opt[1] = aa * laser_power[i]/scan_speed[i] + ab
        goldak_opt[-1] = torch.sigmoid(absa * laser_power[i]/scan_speed[i] + absb)
        T_pred(Ec, model, grid_mesh, laser_power[i], scan_speed[i], goldak_opt, frame, batch_size, str(i), kfa, rra, rrb)


