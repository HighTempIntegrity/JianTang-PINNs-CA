import random
from scipy.stats import qmc
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata
from ImportFile import *
import csv
import matplotlib as mpl
import mpld3
import os
#mpl.use('TkAgg') # if want to show interactive plot
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def FEM_compare(Ec,model,laser_power, scan_speed, Tm,absorptivity, FEM_t, FEM_mesh, FEM_temp, goldak_opt, kfa=0.0, rra=0.015, rrb=0.005, plot_com=False):

        # create grid in the horizontal cross section at x=0 and vertical cross-section at y=0
        grid_size = FEM_mesh.shape[0]
        goldak_opt = goldak_opt.to(Ec.device)
        T_data = []
        FEM_data = []
        absorptivityP = laser_power * absorptivity
        lp = torch.full(size=(grid_size,1), fill_value=1.0).to(Ec.device)*absorptivityP
        ss = torch.full(size=(grid_size,1), fill_value=1.0).to(Ec.device) * scan_speed
        kf = torch.full(size=(grid_size,1), fill_value=1.0).to(Ec.device) * kfa
        rr = torch.randn(size=(grid_size,1)).to(Ec.device)*rra+rrb

        goldak = goldak_opt * torch.full(
            size=(grid_size, goldak_opt.shape[0]), fill_value=1.0).to(Ec.device)


        T_norm = Ec.umax
        mp_fem = []
        mp_pinns_in = []
        mp_pinns = []
        t_norm = []
        for j in range(FEM_t.shape[0]//2, FEM_t.shape[0]):
            FEM_tt = torch.full(size=(grid_size, 1), fill_value=1.0) * FEM_t[j] /FEM_t[-1]
            print("###############################################################################")
            print("Time: ",str(FEM_t[j] / FEM_t[-1]))
            t_norm.append(FEM_t[j] / FEM_t[-1])
            inputs = torch.cat(
                [FEM_tt.to(Ec.device), torch.from_numpy(FEM_mesh).to(Ec.device), rr, kf, lp, ss, goldak[:, :-1]],
                axis=1).type(torch.FloatTensor).to(Ec.device)

            # get PINNs prediction and residual for this time frame based on FEM mesh
            T, residual, dtt, dtxx, q = cal_t_res(inputs, T_norm, model)

            del inputs, FEM_tt

            # measure the melt pool dimension from PINNs and FEM from interpolation based on the FEM mesh
            mp_d,mp_w,mp_l, mp_d_pinns, mp_w_pinns, mp_l_pinns = visual_diff(T[...,None], FEM_temp[:, j:j + 1], FEM_mesh, Tm, residual, dtt, dtxx, q, plot_com)

            # measure the melt pool dimension from PINNs directly from a fine mesh with mesh size of 1 um
            depth, width, length = melt_pool_measure(Ec, model, FEM_t[j] / FEM_t[-1], laser_power, scan_speed,
                                                     goldak_opt, Tm, T_norm, kfa, rra, rrb)

            print("PINNs: depth:",depth, " width: ",width, " length: ",length)
            mp_fem.append([mp_d,mp_w,mp_l])
            mp_pinns.append([depth, width, length])
            mp_pinns_in.append([mp_d_pinns, mp_w_pinns, mp_l_pinns])

            T_data.append(T)
            FEM_data.append(FEM_temp[:, j:j + 1])

        # save and print melt pool dimension for the current conditions at different time frame
        T_data = np.array(T_data)#.reshape(-1,1)
        FEM_data = np.array(FEM_data)#.reshape(-1, 1)
        t_norm = np.array(t_norm)[..., None]
        mp_fem = np.concatenate([t_norm, np.array(mp_fem)], axis=1)[None, ...]
        mp_pinns = np.concatenate([t_norm, np.array(mp_pinns)], axis=1)[None, ...]
        mp_pinns_in = np.concatenate([t_norm, np.array(mp_pinns_in)], axis=1)[None, ...]
        mp = np.concatenate([mp_fem, mp_pinns_in, mp_pinns], axis=0)

        print("###############################################################################")
        print("Process Parameters:  Laser Power ", laser_power.item(), " W   |   Scan Speed ", scan_speed.item(), " mm/s ")
        print('Normalized t  |  FEM D  | PINNS D_in | PINNS D |  FEM W |  PINNS W_in |  PINNS W |  FEM L |  PINNS L_in |  PINNS L ')
        for j in range(mp.shape[1]):
            print(mp[0,j,0], mp[0,j,1], mp[1,j,1], mp[2,j,1], mp[0,j,2], mp[1,j,2], mp[2,j,2], mp[0,j,3], mp[1,j,3], mp[2,j,3])

        return T_data, FEM_data, mp

def cal_t_res(inputs, T_norm, model):
    inputs.requires_grad = True
    u = model(inputs)[:, 0].reshape(-1, )
    # calculate the residual of PINNs

    grad_u = torch.autograd.grad(u, inputs, grad_outputs=torch.ones_like(u).to(Ec.device), create_graph=True)[
        0]
    grad_u_t = grad_u[:, 0]
    grad_u_x = grad_u[:, 1]
    grad_u_y = grad_u[:, 2]
    grad_u_z = grad_u[:, 3]

    rhocp = Ec.heat_capacity_density(u)  # 0.0068#
    c = Ec.conductivity(u)#, ls_factor=inputs[:, 5])

    grad_u_xx = torch.autograd.grad(c * grad_u_x, inputs, grad_outputs=torch.ones_like(u).to(Ec.device),
                                    create_graph=True)[0][:, 1]
    grad_u_yy = torch.autograd.grad(c * grad_u_y, inputs, grad_outputs=torch.ones_like(u).to(Ec.device),
                                    create_graph=True)[0][:, 2]
    grad_u_zz = torch.autograd.grad(c * grad_u_z, inputs, grad_outputs=torch.ones_like(u).to(Ec.device),
                                    create_graph=True)[0][:, 3]
    time = Ec.vm / (inputs[:, 7] * 1000)

    q = Ec.source(inputs) * time / (rhocp * T_norm)#goldak_

    res2 = (grad_u_xx.reshape(-1, ) + grad_u_yy.reshape(-1, ) + grad_u_zz.reshape(-1, )) * time / (rhocp)
    residual = (grad_u_t.reshape(-1, ) - res2 - q)

    # enforce zero-gradient in the region before laser gradient
    mask_init = torch.le(inputs[:, 0], 0)
    residual[mask_init] = grad_u_t.reshape(-1, )[mask_init]
    # enforce temperatures above 0
    value = 25
    mask_temp = torch.le(u * T_norm, value)
    residual[mask_temp] = abs(u[mask_temp] * T_norm - value) * residual[mask_temp] / abs(residual[mask_temp]) + \
                          residual[mask_temp]
    return u.cpu().detach().numpy()*T_norm, residual.cpu().detach().numpy(), grad_u_t.reshape(-1, ).cpu().detach().numpy(), res2.cpu().detach().numpy(), q.cpu().detach().numpy(),

def visual_diff(tp, tt, tmesh, Tm, residual, dtt, dtxx, q, plot_com=False):

    diff = (tt-tp)[:,0]

    x = tmesh[:, 0]
    y = tmesh[:, 1]
    z = tmesh[:, 2]

    # plot the point distribution where the temperature difference larger than 100 K
    if plot_com:
        plot_diff_point(diff, x, y, z)

    # the horizontal plane on the top
    xh = x[z==z.max()]
    yh = y[z==z.max()]
    diffh = diff[z==z.max()]

    # calculate the melt pool length and width from the horizontal plane
    grid_x, grid_y = np.mgrid[-1:1.8:0.001, -1:1:0.001] # unit: um
    grid_z0 = griddata(np.concatenate([xh[..., None], yh[..., None]], axis=1), tt[z==z.max()], (grid_x, grid_y), method='cubic') # interpolation of FEM results
    mp = (grid_z0 >= Tm) * 1.0

    # length and width from FEM after interpolation
    mp_l = mp.sum(axis=0).max()
    mp_w = mp.sum(axis=1).max()


    grid_zp = griddata(np.concatenate([xh[..., None], yh[..., None]], axis=1), tp[z == z.max()], (grid_x, grid_y),
                       method='cubic') # interpolation of PINNs results
    mp_pinns = (grid_zp >= Tm) * 1.0

    # length and width from FEM after interpolation
    mp_l_pinns = mp_pinns.sum(axis=0).max()
    mp_w_pinns = mp_pinns.sum(axis=1).max()
    '''
    xhm = xh[yh==0]
    tth = tt[z==z.max(),0]
    tthm=tth[yh==0]
    tph = tp[z==z.max(),0]
    tphm=tph[yh==0]
    diffhm=diffh[yh==0]
    plt.figure()
    plt.scatter(xhm,tthm,label='FEM')
    plt.scatter(xhm, tphm, label='PINNs')
    #plt.scatter(xhm,diffhm)
    plt.xlim([0.0,1.5])
    plt.ylim([1000,1750])
    plt.show()
    '''
    # plot the temperature and residual at the middle line in y direction and cross line along y direction on the top surface
    if plot_com:
        plot_res_T(xh,yh,residual,grid_x,grid_y,z,grid_z0,grid_zp,mp_pinns,mp,dtxx,dtt,q)

        plot_diff(xh, yh, diffh, grid_x, grid_y,'top')  # plot the difference between FEM and PINNs on the top surface

    # the vertical plane at y=0
    xv = x[y==0.]
    zv = z[y==0.]
    diffv = diff[y==0.]

    # calculate the melt pool depth and width from the vertical plane for FEM
    grid_x, grid_z = np.mgrid[-1:1.8:0.001, -1:0.03:0.001]  # unit: um
    grid_y0 = griddata(np.concatenate([xv[..., None], zv[..., None]], axis=1), tt[y==0.0], (grid_x, grid_z),
                       method='cubic')
    mp = (grid_y0 >= Tm) * 1.0
    mp_d = mp.sum(axis=1).max() # measure the depth from FEM after interpolation

    # calculate the melt pool depth and width from the vertical plane for PINNS
    grid_yp = griddata(np.concatenate([xv[..., None], zv[..., None]], axis=1), tp[y==0.0], (grid_x, grid_z),
                       method='cubic')
    mp_pinns = (grid_yp >= Tm) * 1.0
    mp_d_pinns = mp_pinns.sum(axis=1).max() # measure the depth from PINNS after interpolation

    if plot_com:
        plot_diff(xv, zv, diffv, grid_x, grid_z, 'cross') # plot the difference between FEM and PINNs in the middle plane of SD-BD direction

    print("FEM(interpolation): depth:", mp_d, " width: ", mp_w, " length: ", mp_l)
    print("PINNs (interpolation): depth:", mp_d_pinns, " width: ", mp_w_pinns, " length: ", mp_l_pinns)
    return mp_d,mp_w,mp_l, mp_d_pinns, mp_w_pinns, mp_l_pinns

def plot_diff_point(diff,x,y,z):
    # find where is the big difference
    diff_loc = np.where(np.abs(diff)>100)
    dx = x[diff_loc[0]]
    dy = y[diff_loc[0]]
    dz = z[diff_loc[0]]
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(dx, dy, dz)
    plt.show()

def plot_res_T(xh,yh,residual,grid_x,grid_y,z,grid_z0,grid_zp,mp_pinns,mp,dtxx,dtt,q):
    # find the middle line in y direction on the top surface
    mid_line = np.where(grid_y==0.0)
    tt_mid = grid_z0[mid_line[0],mid_line[1]].reshape(-1,1)

    tp_mid = grid_zp[mid_line[0], mid_line[1]].reshape(-1,1)

    # plot the residual and temperature distribution in the middle line
    fig, ax1 = plt.subplots()
    #ax2 = ax1.twinx()    # create second axis
    grid_res = griddata(np.concatenate([xh[..., None], yh[..., None]], axis=1), residual[z == z.max()], (grid_x, grid_y),
                       method='cubic') # interpolation of residual
    res_mid = grid_res[mid_line[0], mid_line[1]].reshape(-1, 1)
    ax1.plot(np.linspace(-1.0,1.8,len(tp_mid)),tp_mid,label='PINNs')
    ax1.plot(np.linspace(-1.0, 1.8, len(tt_mid)), tt_mid, label='FEM')
    #ax2.plot(np.linspace(-1.0, 1.8, len(res_mid)), np.abs(res_mid), label='abs(residual)', color='g')
    ax1.set_xlabel('x')
    ax1.set_ylabel('T')
    ax1.tick_params(axis="x", direction="in")
    ax1.tick_params(axis="y", direction="in")
    #ax2.set_ylabel('Residual')
    ax1.set_ylim([0,2100])
    #ax2.set_ylim([-0.001,1])
    ax1.set_xlim([-0.1,1.5])
    ax1.legend(loc='best')
    #ax2.legend(loc='best')
    plt.show()

    # plot the residual in the cross section where we measure width in PINNs
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()# create second axis
    tran_line = np.where(grid_x == grid_x[np.argmax(mp_pinns.sum(axis=1)),0]) # the location to measure width in PINNs
    tran_line_fem = np.where(grid_x == grid_x[np.argmax(mp_pinns.sum(axis=1)), 0])  # the location to measure width in FEM
    print("THe width of FEM is measured at x = ", grid_x[np.argmax(mp.sum(axis=1)), 0])
    print("THe width of PINNs is measured at x = ", grid_x[np.argmax(mp_pinns.sum(axis=1)),0])
    tt_tran = grid_z0[tran_line[0],tran_line[1]].reshape(-1,1)
    tp_tran = grid_zp[tran_line[0], tran_line[1]].reshape(-1,1)
    res_tran = grid_res[tran_line[0], tran_line[1]].reshape(-1, 1)
    ax1.plot(np.linspace(-1.0,1.0,len(tp_tran)),tp_tran,label='PINNs')
    ax1.plot(np.linspace(-1.0, 1.0, len(tt_tran)), tt_tran, label='FEM')
    ax2.plot(np.linspace(-1.0, 1.0, len(res_tran)), np.abs(res_tran), label='abs(residual)', color='g')
    ax1.tick_params(axis="x", direction="in")
    ax1.tick_params(axis="y", direction="in")
    ax1.set_xlabel('x')
    ax1.set_ylabel('T')
    ax2.set_ylabel('Residual')
    ax1.set_ylim([1000,2100])
    ax2.set_ylim([-0.001,1])
    ax1.set_xlim([-0.12,0.12])
    ax1.legend(loc='best')
    #ax2.legend(loc='best')
    plt.show()

    # calculate the contribudtion of each residual terms
    grid_dtt = griddata(np.concatenate([xh[..., None], yh[..., None]], axis=1), dtt[z == z.max()], (grid_x, grid_y),
                       method='cubic')
    grid_dtxx = griddata(np.concatenate([xh[..., None], yh[..., None]], axis=1), dtxx[z == z.max()], (grid_x, grid_y),
                       method='cubic')
    grid_q = griddata(np.concatenate([xh[..., None], yh[..., None]], axis=1), q[z == z.max()], (grid_x, grid_y),
                       method='cubic')
    dtt_tran = grid_dtt[tran_line[0], tran_line[1]].reshape(-1, 1)
    dtxx_tran = grid_dtxx[tran_line[0], tran_line[1]].reshape(-1, 1)
    q_tran = grid_q[tran_line[0], tran_line[1]].reshape(-1, 1)
    # plot the contribution of each residual term
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()    # create second axis
    ax1.plot(np.linspace(-1.0, 1.0, len(dtt_tran)), dtt_tran, label='dT/dt')
    ax2.plot(np.linspace(-1.0, 1.0, len(dtxx_tran)), dtxx_tran, label='d2T/dx2', color='g')
    ax2.plot(np.linspace(-1.0, 1.0, len(q_tran)), q_tran, label='q', color='r')
    ax1.set_ylim([-2,2])
    ax2.set_ylim([-2,2])
    ax1.set_xlim([-0.12,0.12])
    ax1.legend(loc='best')
    #ax2.legend(loc='best')
    plt.show()

def plot_diff(x, y, diff, grid_x, grid_y,plane='top'):
    if plane=='top':
        # visualization of the temperature difference of FEM and PINNs
        grid_zd = griddata(np.concatenate([x[..., None], y[..., None]], axis=1), diff, (grid_x, grid_y),
                           method='cubic')
        plt.imshow(np.array(grid_zd.T, dtype=float), extent=(-1, 1.8, -1, 1), origin='lower')
        plt.colorbar()
        plt.gcf().set_size_inches(6, 6)
        plt.show()
    else:
        # visualization of the temperature difference of FEM and PINNs
        grid_yd = griddata(np.concatenate([x[..., None], y[..., None]], axis=1), diff, (grid_x, grid_y),
                           method='cubic')
        plt.imshow(np.array(grid_yd.T, dtype=float), extent=(-1, 1.8, -1, 0.03), origin='lower')
        plt.colorbar()
        plt.gcf().set_size_inches(6, 6)
        plt.show()

def melt_pool_measure(Ec,model, t, laser_power, scan_speed, goldak_opt, Tm, T_norm, kfa=0.0, rra=0.015, rrb=0.005):

        # create grid in the horizontal cross section at x=0 and vertical cross-section at y=0
        domain = Ec.extrema_values[1:4]
        length = 0.001 # the gap between two point is 0.001 mm
        nx = int((domain[0, 1] - domain[0, 0]).item() // length)+1
        ny = int((domain[1, 1] - domain[1, 0]).item() // length)+1
        nz = int((domain[2, 1] - domain[2, 0]).item() // length)+1

        x = torch.from_numpy(np.array(range(int(nx)))*length) + domain[0, 1] - nx*length
        y = torch.from_numpy(np.array(range(int(ny)))*length) + domain[1, 1] - ny*length
        z = torch.from_numpy(np.array(range(int(nz)))*length) + domain[2, 1] - nz*length

        grid_x_h, grid_y_h = torch.meshgrid(x, y)
        grid_x_h = grid_x_h.reshape(-1, 1)
        grid_y_h = grid_y_h.reshape(-1, 1)
        grid_z_h = torch.full(size=grid_x_h.shape, fill_value=0.03)  # JT: need to change here if not measuring at z=0.0
        hor_size = grid_x_h.shape[0]
        hor_grid = torch.cat([grid_x_h, grid_y_h, grid_z_h], axis=1)

        grid_x_v, grid_z_v = torch.meshgrid(x, z)
        grid_x_v = grid_x_v.reshape(-1, 1)
        grid_z_v = grid_z_v.reshape(-1, 1)
        grid_y_v = torch.full(size=grid_x_v.shape, fill_value=0.0)
        ver_grid = torch.cat([grid_x_v, grid_y_v, grid_z_v], axis=1)

        grid_mesh = torch.cat([hor_grid, ver_grid], axis=0)
        grid_size = grid_mesh.shape[0]
        grid_t = torch.full(size=(grid_size,1), fill_value=t)

        # create grid in the horizontal cross section at x=0 and vertical cross-section at y=0
        goldak_opt = goldak_opt.to(Ec.device)
        absorptivityP = laser_power * goldak_opt[-1]
        lp = torch.full(size=(grid_size,1), fill_value=1.0).to(Ec.device)*absorptivityP
        ss = torch.full(size=(grid_size,1), fill_value=1.0).to(Ec.device) * scan_speed
        kf = torch.full(size=(grid_size,1), fill_value=1.0).to(Ec.device) * kfa
        rr = torch.randn(size=(grid_size,1)).to(Ec.device) * rra + rrb

        goldak = goldak_opt * torch.full(
            size=(grid_size, goldak_opt.shape[0]), fill_value=1.0).to(Ec.device)

        inputs = torch.cat(
                [grid_t.to(Ec.device), grid_mesh.to(Ec.device), rr, kf, lp, ss, goldak[:, :-1]],axis=1).type(torch.FloatTensor).to(Ec.device)

        T = model(inputs)[:, 0].reshape(-1, )
        T = T*T_norm

        # measure melt pool depth
        T_hor = torch.relu(T[:hor_size] - Tm) / (T[:hor_size] - Tm + 1e-10)  # For T> Tm, T->1; otherwise T->0
        w_i = torch.sum(T_hor.reshape(nx, ny), axis=1)

        mp_l = torch.max(torch.sum(T_hor.reshape(nx, ny), axis=0)).item() * length * 1e3  # the melt pool length (um)

        # measure melt pool depth
        T_ver = torch.relu(T[hor_size:] - Tm) / (
                    T[hor_size:] - Tm + 1e-10)  # / (T[hor_size:] - Tm + 1e-10)  # For T> Tm, T->1; otherwise T->0
        d_i = torch.sum(T_ver.reshape(nx, nz), axis=1)

        mp_w = torch.max(w_i).item() * length * 1e3  # the melt pool width(um)
        mp_d = torch.max(d_i).item() * length * 1e3  # the melt pool depth(um)


        w_i = torch.sum(T_hor.reshape(nx, ny), axis=1).cpu().detach().numpy()
        tran_line = np.where(grid_x_h == x[np.argmax(w_i)].item())
        print("PINNs width measured at x = :", np.unique(grid_x_h[tran_line[0], tran_line[1]]))
        return mp_d,mp_w,mp_l #, torch.max(T.cpu().detach()).item()

def random_evaluation(Ec, model):
    laser_power = 0.2

    trained_range = torch.cat([torch.Tensor([[0.5, 1.5]]), torch.Tensor([[0.02, 0.08]]),
                               torch.Tensor([[0.02, 0.2]]), torch.Tensor([[0.02, 0.08]]),
                               torch.Tensor([[0.02, 0.25]]), torch.Tensor([[0.3, 0.8]]),
                               torch.Tensor([[0.002, 0.005]]), torch.Tensor([[0.005, 0.02]])], axis=0)
    Tm = 1307

    sampler = qmc.LatinHypercube(d=8, optimization="random-cd")
    sample_tmp = torch.from_numpy(sampler.random(n=1500)) * (trained_range[:, 1] - trained_range[:, 0]) + trained_range[:, 0]
    sample = sample_tmp[sample_tmp[:,4] > sample_tmp[:,3]]


    mp_rec=[]

    for i in range(len(sample)):
        scan_speed = sample[i,0]
        goldak_opt = sample[i, 1:6]
        kfa = sample[i, 6]
        rra = 0.0
        rrb = sample[i, 7]
        depth, width, length = melt_pool_measure(Ec, model, 0.9, laser_power, scan_speed, goldak_opt, Tm,
                                                    Ec.umax, kfa, rra, rrb)
        mp_rec.append([depth, width, length])
    mp_rec = np.array(mp_rec)
    mask =  (mp_rec[:, 1] > 0) & (mp_rec[:, 2] > 0) & (mp_rec[:, 0] > 0)
    mp_rec = mp_rec[mask]
    sample = sample.numpy()
    sample = sample[mask]
    tot_rec = np.concatenate([sample, mp_rec], axis=1)
    if len(tot_rec)>1000:
        tot_rec=tot_rec[:1000,:]
    np.savetxt("./random_evaluate_sen.txt",tot_rec)
    return tot_rec

def mp_opt_measure(sample,Tm=1307.):
    goldak_o = torch.Tensor([1.0301, 0.0745, 0.0233, 0.0231, 2.7704, 0.0320, 0.0089])

    sample=np.array(sample)

    lp = torch.Tensor([0.2, 0.175, 0.2, 0.175, 0.125])
    ss = torch.Tensor([0.9, 0.8, 1.1, 0.9, 0.7])
    kfa = 0.003
    rra = 0.0
    rrb = 0.01

    mp_rec=[]
    for j in range(len(sample)):
        mpi=[]
        for i in range(5):
            goldak_o = torch.from_numpy(sample[j,:])
            goldak_opt = torch.Tensor(
                [goldak_o[1], goldak_o[0] * lp[i] / ss[i] + goldak_o[-1], goldak_o[2], goldak_o[3],
                 goldak_o[-3] * lp[i] / ss[i] + goldak_o[-2]])
            depth, width, length = melt_pool_measure(Ec, model, 0.9, lp[i], ss[i], goldak_opt, Tm,
                                                     Ec.umax, kfa, rra, rrb)
            mpi.append([depth, width, length])
        mp_rec.append(mpi)


if __name__ == "__main__":
    # load the model
    Ec = EquationClass()
    model_path = 'network_20230826_082323/'#network_20230727_030849/'#network20230604_224415/'#network20230530_150029/'#network20230602_020625/'#network20230530_023140/'#network20230528_234823/'#network20230412_105024/'#network20230601_211936/'#
    model = torch.load(os.path.join(model_path, 'model_full.pkl'))

    # using hypercube sampling get random input settings and measure the corresponding melt pool dimensions
    #random_evaluation(Ec, model)

    # check whether the result folder exists
    res_folder = './comp_result'
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
        
    Tm = 1307.  # define melting temperature
    pinn_sum=[]
    fem_sum=[]
    err=[]
    MP=[]
    plot_com = False #True #
    # read the goldak settings from the file
    selected_input = np.loadtxt('./FEM_data/random_set.csv',delimiter=',')
    scan_speed = selected_input[:,0].T
    laser_power = torch.Tensor([0.2,0.175,0.2,0.175,0.125]) # scan_speed* 0.0 + 0.2 #  we assume laser power are all 200 W
    FEM_mesh = pd.read_csv('./FEM_data/Node.csv', sep=',',header=0).to_numpy()[:,1:] # read the FEM mesh
    for i in range(5):
        #if i in [0,2,4,5,6]:
        #    continue
        FEM_temp1 = pd.read_csv('./FEM_data/FEM_temp'+str(i+1)+'.csv', sep=',',header=None).to_numpy()[:,1:] # read the FEM temperature data
        FEM_t = FEM_temp1[0,:]
        FEM_temp = FEM_temp1[1:,:]

        goldak_opt = torch.from_numpy(selected_input[i,1:6]) # read the goldak setting
        kfa = selected_input[i,6]  *0+0.003#0.0015 # channel for thermal conductivity.  BUt for this model we don't consider it.
        rrb = selected_input[i,7] *0+0.01  # free channel
        rra = 0.0
        #print(goldak_opt,laser_power,scan_speed,kfa,rrb)
        #a = goldak_opt[1]
        #goldak_opt[1] = goldak_opt[2]
        #goldak_opt[2] = a
        print(laser_power[i], scan_speed[i],goldak_opt, kfa, rra, rrb)
        # subroutine of comparison between FEM and PINNs
        pinn, fem, mp = FEM_compare(Ec, model, laser_power[i], scan_speed[i], Tm, goldak_opt[-1], FEM_t, FEM_mesh, FEM_temp,
                    goldak_opt, kfa, rra, rrb, plot_com)

        pinn_sum.append((pinn.T)[:,-3])#.reshape(-1,mp.shape[1])
        print(fem.shape)
        fem_sum.append((fem[:,:,0].T)[:,-3])#.reshape(-1,mp.shape[1])
        MP.append(mp)
        np.savetxt('./comp_result/PINNs_temp'+str(i+1)+'.csv', pinn.reshape(-1,mp.shape[1]), delimiter=",")
        #err_i = (abs(pinn - fem)/fem).mean()

    fem_sum=np.array(fem_sum)
    pinn_sum=np.array(pinn_sum)
    print(fem_sum.shape,pinn_sum.shape)
    # plot the temperature comparsion between FEM and PINNS
    color = {1:'#0a3d62',2:'#0c2461', 3:'#78e08f', 4:'#079992', 10:'#e58e26', 6:'#82a3ac', 7:'#a9c7c5', 8:'#e2cfc9', 9:'#bfd8d2', 5:'#d1a7a5', 11:'#d1aca5'}
    T_err_sum = []
    for i in range(len(pinn_sum)):
        T_err = np.abs(np.average(np.abs(fem_sum[i]- pinn_sum[i]))) * 100.0 / np.average(fem_sum[i])
        T_err_sum.append(T_err)
        plt.scatter(fem_sum[i], pinn_sum[i], facecolors='none', edgecolors=color[i+1], s=18, label="\#"+str(i+1))#str(round(laser_power[i].item()*1000))+' W, '+str(round(scan_speed[i].item()*1000))+' mm/s')#+' '+str(round(err_i*100))+'%')
    plt.legend(loc='best')
    plt.plot([0, 2500], [0, 2500])
    plt.xlim([0, 2500])
    plt.ylim([0, 2500])
    plt.xlabel('FEM calculated temperature ($^\circ$C)')
    plt.ylabel('PINNs calculated temperature ($^\circ$C)')
    plt.savefig('./comp_result/model_comp.jpg',dpi=600)
    plt.show(block=True)

    T_err_sum = np.array(T_err_sum)
    print("The temperature error is ", T_err_sum,' % ')
    print(" in average: ", np.mean(T_err_sum),' pm ', np.std(T_err_sum),' % ')

    # plot the temperature distribution along the centre line on the top surface

    x = FEM_mesh[:, 0]
    y = FEM_mesh[:, 1]
    z = FEM_mesh[:, 2]
    surface_filter = (z==z.max()) & (y==0.0)

    for i in range(len(pinn_sum)):
        pinns_tmp = (pinn_sum[i])[surface_filter]
        x_tmp = x[surface_filter]
        fem_tmp = (fem_sum[i])[surface_filter]
        plt.scatter(x_tmp,fem_tmp, facecolors='none', edgecolors=color[i+1], s=18)#str(round(laser_power[i].item()*1000))+' W, '+str(round(scan_speed[i].item()*1000))+' mm/s')#+' '+str(round(err_i*100))+'%')
        ord = np.argsort(x_tmp)
        pinns_tmp = pinns_tmp[ord]
        x_tmp = x_tmp[ord]
        plt.plot(x_tmp, pinns_tmp, color=color[i + 1], label="\#"+str(i+1))
    plt.legend(loc='best')

    plt.xlim([0, 1.5])
    plt.ylim([0, 2500])
    plt.ylabel('Temperature ($^\circ$C)')
    plt.xlabel('x (mm)')
    plt.savefig('./comp_result/model_comp_T.jpg',dpi=600)
    plt.show(block=True)


    # print and save the melt pool dimensions measured from FEM, PINNS+interpolation, PINNs+ fine mesh
    with open('./comp_result/meltpool_summary.csv', 'w') as f:
        writer = csv.writer(f, delimiter =' ')
        writer.writerow([c.strip() for c in " Melt pool summary".split(' ')])
        for i in range(len(MP)):
            writer.writerow("####################################################################################")
            writer.writerow([c.strip() for c in ("Benchmark case "+ str(i+1)).split(' ')])
            writer.writerow([c.strip() for c in (str("Process Parameters:  Laser Power " + "".join(str(laser_power[i].item())) + " W   |   Scan Speed " + "".join(str(scan_speed[i].item())) + " mm/s ")).split(' ')])
            mp = MP[i]
            writer.writerow([c.strip() for c in str('Normalized t  |  FEM D  | PINNS D_in | PINNS D |  FEM W |  PINNS W_in |  PINNS W |  FEM L |  PINNS L_in |  PINNS L ').split(' ')])
            for j in range(mp.shape[1]):
                writer.writerow([mp[0,j,0], mp[0,j,1], mp[1,j,1], mp[2,j,1], mp[0,j,2], mp[1,j,2], mp[2,j,2], mp[0,j,3], mp[1,j,3], mp[2,j,3]])
    MP_FEM = []
    MP_PINNS_in = []
    MP_PINNS = []
    for i in range(len(MP)):
        mp = MP[i]
        MP_FEM.append(np.median(mp[0,...],axis=0))
        MP_PINNS_in.append(np.median(mp[1, ...], axis=0))
        MP_PINNS.append(np.median(mp[2, ...], axis=0))
    MP_FEM = np.array(MP_FEM)[:,1:]
    MP_PINNS_in = np.array(MP_PINNS_in)[:,1:]
    MP_PINNS = np.array(MP_PINNS)[:,1:]

    print("################################################################")
    print('FEM D  | PINNS D_in | PINNS D |  FEM W |  PINNS W_in |  PINNS W |  FEM L |  PINNS L_in |  PINNS L ')
    for i in range(MP_FEM.shape[0]):
        print(MP_FEM[i,0],MP_PINNS_in[i,0],MP_PINNS[i,0], MP_FEM[i,1],MP_PINNS_in[i,1],MP_PINNS[i,1], MP_FEM[i,2],MP_PINNS_in[i,2],MP_PINNS[i,2])

