import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata
from ImportFile import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def goldak_search(Ec,model, t, laser_power, scan_speed, mp_d_exp,mp_w_exp,Tm,absorptivity):

        # create grid in the horizontal cross section at x=0 and vertical cross-section at y=0
        domain = Ec.extrema_values[1:4]
        length = 0.003 # the gap between two point is 0.001 mm
        nx = int((domain[0, 1] - domain[0, 0]).item() // length)
        ny = int((domain[1, 1] - domain[1, 0]).item() // length)
        nz = int((domain[2, 1] - domain[2, 0]).item() // length)

        x = torch.from_numpy(np.array(range(int(nx)))*length) + domain[0, 0]
        y = torch.from_numpy(np.array(range(int(ny)))*length) + domain[1, 0]
        z = torch.from_numpy(np.array(range(int(nz)))*length) + domain[2, 0]

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

        absorptivity = torch.Tensor(np.random.uniform(size=laser_power.shape[0]))*0.0+1.0
        #absorptivity.requires_grad = True

        for j in range(laser_power.shape[0]):
            if j == 0:
                lp = torch.full(size=(grid_size,1), fill_value=1.0)*laser_power[j] * absorptivity[j]
                ss = torch.full(size=(grid_size,1), fill_value=scan_speed[j])
            else:
                lp = torch.cat([lp,torch.full(size=(grid_size,1), fill_value=1.0)*laser_power[j] * absorptivity[j]], axis=0)
                ss = torch.cat([ss, torch.full(size=(grid_size,1), fill_value=scan_speed[j])], axis=0)
                grid_mesh = torch.cat([grid_mesh, torch.cat([hor_grid, ver_grid], axis=0)], axis=0)
                grid_t = torch.cat([grid_t, torch.full(size=(grid_size, 1), fill_value=t)],axis=0)

        grid_mesh = grid_mesh.to(Ec.device)
        grid_t = grid_t.to(Ec.device)
        lp = lp.to(Ec.device)
        ss = ss.to(Ec.device)
        fc = torch.full(size=(grid_t.shape[0], 1), fill_value=torch.rand(1).item()).to(Ec.device) * \
             (Ec.parameters_values[4, 1] - Ec.parameters_values[4, 0]) + Ec.parameters_values[4, 0] # free channel

        mp_w_exp = torch.from_numpy(mp_w_exp).to(Ec.device)
        mp_d_exp = torch.from_numpy(mp_d_exp).to(Ec.device)
        goldak_rec = []
        mp_d_tot = []
        mp_w_tot = []
        mp_l_tot = []
        for test_i in range(200):
            torch.cuda.empty_cache()
            # initialization of goldak parameters
            goldak_range = Ec.parameters_values[-4:, :].to(Ec.device)# JT: need to double check the index
            
            
            goldak_range = torch.cat(
                [Ec.parameters_values[1:2, :].to(Ec.device),  # range for the thermal conductivity factor
                 goldak_range,                                # range for the four goldak parameters
                 torch.Tensor([[1.0,9.0]]).to(Ec.device),torch.Tensor([[-1.0,0.8]]).to(Ec.device),#torch.Tensor([[0.0,10.0]]).to(Ec.device),torch.Tensor([[-1,1.5]]).to(Ec.device), # absorptivty = sigmoid(a * P/V + b)
                 torch.Tensor([[0.02, 0.18]]).to(Ec.device)],axis=0).to(Ec.device)


            goldak_mean = torch.mean(goldak_range, axis=1).to(Ec.device) # initial guess of goldak_a, goldak_b, goldak_cr, goldak_cf
            goldak_halrang = ((goldak_range[:,1]-goldak_range[:,0])/2.0).to(Ec.device)
            goldak_ini = torch.Tensor(size=([goldak_range.shape[0]])).to(Ec.device)  
            
            for gold_i in range(goldak_range.shape[0]):
                    goldak_ini[gold_i] = np.random.uniform()
            goldak_ini = goldak_ini * (goldak_range[:,1]-goldak_range[:,0]) + goldak_range[:,0]
            #goldak_ini = torch.Tensor(np.random.uniform(size=goldak_range.shape[0])).to(Ec.device)*(goldak_range[:,1]-goldak_range[:,0]) + goldak_range[:,0]

            # initialization of the p and q in goldak_a = p * P/V + q
            goldak_ini[2] = np.random.uniform(0.1,0.9)#(0.9,2.0)
            goldak_ini[-1] = np.random.uniform(0.02, 0.18)
            goldak_ini[1] =  0.0275 # np.random.uniform(0.02,0.05)
            #goldak_ini[3] =  np.random.uniform(0.05,0.10)
            #goldak_ini[4] =  np.random.uniform(0.1,0.25)
            
            #goldak_ini[0] =  np.random.uniform(-0.5,2.0)
            

            #goldak_ini[0] = np.random.uniform(0.5,1.2)#
            #goldak_ini[-1] = np.random.uniform(-0.05, 0.05)
            #goldak_ini[1] = 0.053
            #goldak_ini[2] = 0.1
            #goldak_ini[3] = 0.4
            '''goldak_range[:,0]
            goldak_ini[-1] = 0.8
            goldak_ini[1] = 0.18
            goldak_ini[2] = 0.05
            goldak_ini[3] = 0.05
            goldak_ini[4] = 0.2
            '''
            goldak_ini.requires_grad = True
            opt_goldak = goldak_ini.detach().clone()

            a_ed= torch.Tensor([0.0]).to(Ec.device)
            a_ed.requires_grad = True

            # optimizer
            #optimizer = optim.Adam([goldak_ini], lr=0.1)
            optimizer = optim.LBFGS([goldak_ini], lr=0.1, max_iter=5000, history_size=100)#, line_search_fn="strong_wolfe", tolerance_change=1.0*np.finfo(float).eps)
            #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.3, patience = 5, verbose=False,threshold=0.0001)
            T_norm = Ec.umax

            decayRate = 0.96
            my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.3)#torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer , gamma=decayRate)
            scheduler_down = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8,
                                                              verbose=True)
            scheduler_up = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8 ** -0.25,
                                                            verbose=True)
            # train
            running_loss = list()

            for epoch in range(10):
                def closure():
                    optimizer.zero_grad()
                    mp_loss = 0.0

                    # predict
                    goldak_new = goldak_ini
                    goldak = goldak_new  * torch.full(
                        size=(grid_mesh.shape[0], goldak_range.shape[0]), fill_value=1.0).to(Ec.device)

                    absorptivity = torch.sigmoid(goldak[:, -3:-2] * (lp * 1000) / ((ss * 1000)) + goldak[:,-2:-1])
                    goldak_a = goldak[:, 2:3] * (lp * 1000) / ((ss * 1000)) + goldak[:, -1:]
                    inputs = torch.cat([grid_t, grid_mesh, fc, goldak[:, 0:1], lp*absorptivity,  ss, goldak[:, 1:2], goldak_a, goldak[:, 3:5]],
                                       axis=1).type(torch.FloatTensor)

                    T = model(inputs.to(Ec.device)) * T_norm

                    for j in range(laser_power.shape[0]):
                        # measure melt pool depth
                        T_hor = torch.sigmoid(T[j*grid_size:j*grid_size+hor_size] - Tm) #/ (T[:hor_size] - Tm + 1e-10)  # For T> Tm, T->1; otherwise T->0
                        w_i = torch.sum(T_hor.reshape(nx, ny), axis=1)

                        # mp_l = torch.max(torch.sum(T_hor.reshape(nx, ny), axis=0))  # the melt pool length

                        # measure melt pool depth
                        T_ver = torch.sigmoid(T[j*grid_size+hor_size:(j+1)*grid_size] - Tm) #/ (T[hor_size:] - Tm + 1e-10)  # For T> Tm, T->1; otherwise T->0
                        d_i = torch.sum(T_ver.reshape(nx, nz), axis=1)

                        mp_w = torch.max(w_i) * length * 1e3# the melt pool width
                        mp_d = torch.max(d_i) * length * 1e3# the melt pool depth

                        mp_loss = mp_loss + (((mp_d  - mp_d_exp[j])**2)**0.5 + ((mp_w  - mp_w_exp[j])**2)**0.5)/laser_power.shape[0]
                        #(((((mp_d  - mp_d_exp[j])) ** 2 + (
                        #(mp_w  - mp_w_exp[j])) ** 2) ) ** 0.5) /(laser_power.shape[0] * 2.0)
                        torch.cuda.empty_cache()

                        '''
                        if j == 0:
                            mp_w = torch.Tensor([torch.max(w_i)]).to(Ec.device)
                            mp_d = torch.Tensor([torch.max(d_i)]).to(Ec.device)
                        else:
                            mp_w = torch.cat([mp_w, torch.max(w_i)], dim=0)  # the melt pool width
                            mp_d = torch.cat([mp_d, torch.max(d_i)], axis=0)  # the melt pool depth
                        '''

                    # loss
                    goldak_loss = torch.sum(torch.relu_(((goldak_ini-goldak_mean)**2)**0.5-goldak_halrang)[3:5]) + torch.sum((-goldak_ini+((goldak_ini)**2)**0.5)[1:-2])+\
                                  torch.sum(torch.relu_(((goldak_ini-goldak_mean)**2)**0.5-goldak_halrang)[0:2]) 
                                  
                    #+ ((absorp_linear_T)**2)**0.5#+ (-absorp_linear_T+((absorp_linear_T)**2)**0.5)#+ torch.sum(((torch.relu(goldak_ini)-goldak_ini)**2)**0.5)
                    #100 * torch.sum((((goldak_ini - torch.clamp(goldak_ini,
                    #                    min=goldak_range[:,0], max=goldak_range[:,1]))/goldak_range[:,0])**2)**0.5)#

                    loss = mp_loss + 10000. * goldak_loss
                    loss.backward()#retain_graph=True)
                    running_loss.append(loss.item())
                    del inputs, T, T_ver, T_hor, goldak
                    torch.cuda.empty_cache()
                    return loss
                optimizer.step(closure=closure)
                if epoch==0:
                    patience = 0
                    if running_loss[-1]/running_loss[0] > 10:
                        opt_loss = running_loss[0]
                    else:
                        opt_loss = running_loss[-1]

                else:
                    if (running_loss[-1] - opt_loss)/opt_loss < 0.1:
                        patience += 1
                    if patience > 3:
                        if (((running_loss[0] - running_loss[-1])/running_loss[0])**2)**0.5 < 0.1:
                            break
                        scheduler_up.step()
                        patience = 0

                if (running_loss[-1]/opt_loss > 10) or np.isnan(running_loss[-1]):
                    goldak_ini = opt_goldak.clone()
                    goldak_ini.requires_grad = True
                    scheduler_down.step()
                else:
                    opt_goldak = goldak_ini.detach().clone()
                    opt_loss = running_loss[-1]
                    a_ed_copy = a_ed.detach().clone()


                #my_lr_scheduler.step()
                    #scheduler.step(loss)

            # report
            print(r"The optimization loss is %f",running_loss[-1],"Goldak parameters: ",goldak_ini,"free channel: ",fc[0,0].item(),
                  "Absorptivity: ",torch.unique(torch.sigmoid(goldak_ini[-3:-2] * (lp*1000)/((ss*1000)) + goldak_ini[-2:-1])).cpu().detach().numpy())

            if running_loss[-1] < 20:
                goldak_rec.append(np.insert(goldak_ini.cpu().detach().numpy(),-1,fc[0,0].item()))#np.append(,a_ed.item()))
                goldak_ini.requires_grad = False
                def melt_pool_measure(goldak_ini,grid_t, grid_mesh, lp, ss):
                    goldak_new = goldak_ini
                    goldak = goldak_new  * torch.full(
                        size=(grid_mesh.shape[0], goldak_range.shape[0]), fill_value=1.0).to(Ec.device)

                    absorptivity = torch.sigmoid(goldak[:, -3:-2] * (lp * 1000) / ((ss * 1000)) + goldak[:,-2:-1])
                    goldak_a = goldak[:, 2:3] * (lp * 1000) / ((ss * 1000)) + goldak[:, -1:]
                    inputs = torch.cat([grid_t, grid_mesh, fc, goldak[:, 0:1], lp*absorptivity,  ss, goldak[:, 1:2], goldak_a, goldak[:, 3:5]],
                                       axis=1).type(torch.FloatTensor)

                    T = model(inputs.to(Ec.device)) * T_norm
                    mp_d_rec=[]
                    mp_w_rec=[]
                    mp_l_rec=[]
                    for j in range(laser_power.shape[0]):
                        # measure melt pool depth
                        T_hor = torch.relu(T[j * grid_size:j * grid_size + hor_size] - Tm) / (T[j * grid_size:j * grid_size + hor_size] - Tm + 1e-10) # / (T[:hor_size] - Tm + 1e-10)  # For T> Tm, T->1; otherwise T->0
                        w_i = torch.sum(T_hor.reshape(nx, ny), axis=1)
                        l_i = torch.sum(T_hor.reshape(nx, ny), axis=0)

                        # mp_l = torch.max(torch.sum(T_hor.reshape(nx, ny), axis=0))  # the melt pool length

                        # measure melt pool depth
                        T_ver = torch.relu(T[j * grid_size + hor_size:(j + 1) * grid_size] - Tm) / (T[j * grid_size + hor_size:(j + 1) * grid_size] - Tm + 1e-10) # / (T[hor_size:] - Tm + 1e-10)  # For T> Tm, T->1; otherwise T->0
                        d_i = torch.sum(T_ver.reshape(nx, nz), axis=1)

                        mp_w_rec.append((torch.max(w_i) * length * 1e3).item())  # the melt pool width
                        mp_d_rec.append((torch.max(d_i) * length * 1e3).item())  # the melt pool depth
                        mp_l_rec.append((torch.max(l_i) * length * 1e3).item())  # the melt pool depth
                        
                    del T,T_ver,T_hor,inputs,goldak
                    torch.cuda.empty_cache()
                    return mp_w_rec, mp_d_rec, mp_l_rec

                mp_w_rec, mp_d_rec, mp_l_rec = melt_pool_measure(goldak_ini,grid_t, grid_mesh, lp, ss)
                print("Melt pool depth & width: ", mp_d_rec, mp_w_rec)
                mp_d_tot.append(mp_d_rec)
                mp_w_tot.append(mp_w_rec)
                mp_l_tot.append(mp_l_rec)
        goldak_rec = np.array(goldak_rec)
        mp_d_tot = np.array(mp_d_tot)
        mp_w_tot = np.array(mp_w_tot)
        mp_l_tot = np.array(mp_l_tot)
        
        print(goldak_rec, mp_d_tot, mp_w_tot)
        print("goldak record: ")
        for p_i in range(goldak_rec.shape[0]):
                print(goldak_rec[p_i,:])
        print( "melt pool depth: ")
        for p_i in range(mp_d_tot.shape[0]):      
                print(mp_d_tot[p_i,:])
        print( "melt pool width: ")
        for p_i in range(mp_w_tot.shape[0]):  
                print(mp_w_tot[p_i,:])
        
        output_rec = np.concatenate([goldak_rec, mp_d_tot, mp_w_tot, mp_l_tot], axis=1)
        np.savetxt('./inv_sea_rec.txt',output_rec,delimiter=',')
        print("The optimization goldak parameter a is %f; \n b is %f; \n cr is %f; \n cf is %f.", goldak_ini[0], goldak_ini[1], goldak_ini[2], goldak_ini[3])
        plt.plot(np.array(running_loss))
        plt.show()
        plt.savefig("./loss.png")





def goldak_mc(Ec, model, t, laser_power, scan_speed, mp_d_exp, mp_w_exp, Tm, absorptivity):
    # create grid in the horizontal cross section at x=0 and vertical cross-section at y=0
    domain = Ec.extrema_values[1:4]
    length = 0.003  # the gap between two point is 0.001 mm
    T_norm = Ec.umax
    nx = int((domain[0, 1] - domain[0, 0]).item() // length)
    ny = int((domain[1, 1] - domain[1, 0]).item() // length)
    nz = int((domain[2, 1] - domain[2, 0]).item() // length)

    x = torch.from_numpy(np.array(range(int(nx))) * length) + domain[0, 0]
    y = torch.from_numpy(np.array(range(int(ny))) * length) + domain[1, 0]
    z = torch.from_numpy(np.array(range(int(nz))) * length) + domain[2, 0]

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
    grid_t = torch.full(size=(grid_size, 1), fill_value=t)

    absorptivity = torch.Tensor(np.random.uniform(size=laser_power.shape[0])) * 0.0 + 1.0
    # absorptivity.requires_grad = True

    for j in range(laser_power.shape[0]):
        if j == 0:
            lp = torch.full(size=(grid_size, 1), fill_value=1.0) * laser_power[j] * absorptivity[j]
            ss = torch.full(size=(grid_size, 1), fill_value=scan_speed[j])
        else:
            lp = torch.cat([lp, torch.full(size=(grid_size, 1), fill_value=1.0) * laser_power[j] * absorptivity[j]],
                           axis=0)
            ss = torch.cat([ss, torch.full(size=(grid_size, 1), fill_value=scan_speed[j])], axis=0)
            grid_mesh = torch.cat([grid_mesh, torch.cat([hor_grid, ver_grid], axis=0)], axis=0)
            grid_t = torch.cat([grid_t, torch.full(size=(grid_size, 1), fill_value=t)], axis=0)

    grid_mesh = grid_mesh.to(Ec.device)
    grid_t = grid_t.to(Ec.device)
    lp = lp.to(Ec.device)
    ss = ss.to(Ec.device)
    '''
    grid_mesh = torch.cat([hor_grid, ver_grid], axis=0).to(Ec.device)
    grid_t = torch.full(size=(grid_mesh.shape[0], 1), fill_value=t).to(Ec.device)
    lp = torch.full(size=(grid_mesh.shape[0], 1), fill_value=laser_power * absorptivity).to(Ec.device)
    ss = torch.full(size=(grid_mesh.shape[0], 1), fill_value=scan_speed).to(Ec.device)

    # random goldak parameters
    goldak_range = Ec.parameters_values.to(Ec.device)
    '''
    goldak_range = Ec.parameters_values[-4:, :].to(Ec.device)  # JT: need to double check the index
    goldak_range = torch.cat(
        [goldak_range, torch.Tensor([[-1, 1]]).to(Ec.device), torch.Tensor([[-1, 1]]).to(Ec.device)], axis=0).to(
        Ec.device)
    rec = dict()
    plt_data = []

    for run_i in range(5000):
        goldak_ran = torch.Tensor(np.random.uniform(size=goldak_range.shape[0])).to(Ec.device) *(goldak_range[:, 1] - goldak_range[:, 0]) + goldak_range[:, 0]


        '''
        goldak = goldak_ran * (goldak_range[:, 1] - goldak_range[:, 0]) + goldak_range[:, 0]
        goldak = goldak * torch.full(size=(grid_mesh.shape[0], 1), fill_value=1.0).to(Ec.device)
        #goldak = torch.Tensor(np.random.uniform(size=(grid_mesh.shape[0],8))).to(Ec.device)
        #for j in range(goldak_range.shape[0]):
        #    goldak[:,j] = goldak[:,j]* (goldak_range[j, 1] - goldak_range[j, 0]) + goldak_range[j, 0]
        inputs = torch.cat([grid_t, grid_mesh, goldak[:, :1], ss, goldak[:, 2:]],axis=1).type(torch.FloatTensor)
        '''

        goldak = goldak_ran * torch.full(
            size=(grid_mesh.shape[0], goldak_range.shape[0]), fill_value=1.0).to(Ec.device)
        absorptivity = torch.sigmoid(goldak[:, -2:-1] * (lp * 1000) / ((ss * 1000) ** 0.5) + goldak[:,
                                                                                             -1:])  # 0.126*(lp*1000)/((ss*1000)**0.5) - 0.297 #goldak[:, -1:]#
        inputs = torch.cat([grid_t, grid_mesh, lp * absorptivity, ss, goldak[:, :-2]],
                           axis=1).type(torch.FloatTensor)

        T = model(inputs.to(Ec.device)) * T_norm

        # measure melt pool depth
        mp_d = []
        mp_w = []
        mp_l = []
        for j in range(laser_power.shape[0]):
            # measure melt pool depth
            T_hor = torch.relu(T[j * grid_size:j * grid_size + hor_size] - Tm) / (T[j * grid_size:j * grid_size + hor_size] - Tm + 1e-10)  # / (T[:hor_size] - Tm + 1e-10)  # For T> Tm, T->1; otherwise T->0
            w_i = torch.sum(T_hor.reshape(nx, ny), axis=1)
            l_i = torch.sum(T_hor.reshape(nx, ny), axis=0) # the melt pool length
            # mp_l = torch.max(torch.sum(T_hor.reshape(nx, ny), axis=0))  # the melt pool length

            # measure melt pool depth
            T_ver = torch.relu(T[j * grid_size + hor_size:(j + 1) * grid_size] - Tm) / (T[j * grid_size + hor_size:(j + 1) * grid_size] - Tm + 1e-10)  # / (T[hor_size:] - Tm + 1e-10)  # For T> Tm, T->1; otherwise T->0
            d_i = torch.sum(T_ver.reshape(nx, nz), axis=1)

            mp_w.append((torch.max(w_i) * length * 1e3).item())  # the melt pool width
            mp_d.append((torch.max(d_i) * length * 1e3).item())  # the melt pool depth
            mp_l.append((torch.max(l_i) * length * 1e3).item())
        # loss
        mp_d=np.array(mp_d)
        mp_w = np.array(mp_w)
        mp_l = np.array(mp_l)
        loss = ((mp_d - mp_d_exp)**2)**0.5 + ((mp_w - mp_w_exp)**2)**0.5 #((((np.array(mp_d)  - np.array(mp_d_exp)) ** 2 + ((np.array(mp_w)  - (np.array(mp_w_exp))) ** 2) / 2.0) ** 0.5

        # save
        if (mp_w > 0).all() and (mp_d > 0).all():
            info = {'input': inputs.cpu().detach().numpy()[0, :], 'mp width': mp_w,
                        'mp_depth': mp_d, 'mp_length': mp_l,
                        'loss': loss}
            rec[str(run_i)] = info
            plt_data.append(np.concatenate([inputs.cpu().detach().numpy()[0, :],np.concatenate([mp_w,mp_d,mp_l],axis=0)], axis=0).tolist())
        del inputs, T, T_ver, T_hor
        del goldak, goldak_ran
        torch.cuda.empty_cache()

    return rec, plt_data



def goldak_fit(Ec,model, t, laser_power, scan_speed, mp_d_exp,mp_w_exp,Tm):

        # create grid in the horizontal cross section at x=0 and vertical cross-section at y=0
        domain = Ec.extrema_values[1:4]
        length = 0.001 # the gap between two point is 0.001 mm
        nx = int((domain[0, 1] - domain[0, 0]).item() // length)
        ny = int((domain[1, 1] - domain[1, 0]).item() // length)
        nz = int((domain[2, 1] - domain[2, 0]).item() // length)

        x = torch.from_numpy(np.array(range(int(nx)))*length) + domain[0, 0]
        y = torch.from_numpy(np.array(range(int(ny)))*length) + domain[1, 0]
        z = torch.from_numpy(np.array(range(int(nz)))*length) + domain[2, 0]

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

        grid_mesh = torch.cat([hor_grid, ver_grid], axis=0).to(Ec.device)
        grid_t = torch.full(size=(grid_mesh.shape[0],1), fill_value=t).to(Ec.device)
        lp = torch.full(size=(grid_mesh.shape[0],1), fill_value=laser_power).to(Ec.device)
        ss = torch.full(size=(grid_mesh.shape[0],1), fill_value=scan_speed).to(Ec.device)

        rec=dict()
        for i in range(100):
            rec = one_random_evaluate(Ec, grid_t, grid_mesh, lp, ss, model, length, hor_size, nx, ny, nz,Tm, mp_w_exp, mp_d_exp, rec,
                            i)
        return rec
def one_random_evaluate(Ec,grid_t, grid_mesh, lp, ss, model,length,hor_size,nx,ny,nz,Tm, mp_w_exp, mp_d_exp,rec,i):
        # initialization of goldak parameters
        goldak_range = Ec.parameters_values[2:6]  # JT: need to double check the index
        goldak_range = torch.cat([goldak_range,torch.Tensor([[0.5,1.0]])],axis=0)

        goldak_ini = torch.Tensor(np.random.uniform(size=5))*(goldak_range[:,1]-goldak_range[:,0]) + goldak_range[:,0]
        goldak_ini = goldak_ini.to(Ec.device)
        goldak_ini.requires_grad = True
        goldak_cp = torch.clone(goldak_ini)

        goldak = torch.clamp(goldak_ini[:4], min=Ec.parameters_values[2:6, 0].to(Ec.device),max=Ec.parameters_values[2:6, 1].to(Ec.device)) * torch.full(size=(grid_mesh.shape[0],4), fill_value=1.0).to(Ec.device)
        inputs = torch.cat([grid_t, grid_mesh, lp*torch.clamp(goldak_ini[-1],min=0.5,max=1.0), ss, goldak], axis=1).type(torch.FloatTensor)

        T_norm = Ec.umax

        T = model(inputs.to(Ec.device)) * T_norm

        # measure melt pool depth
        T_hor = torch.relu(T[:hor_size] - Tm) / (T[:hor_size] - Tm + 1e-10)  # For T> Tm, T->1; otherwise T->0
        mp_w = torch.max(torch.sum(T_hor.reshape(nx, ny), axis=1))  # the melt pool width
        mp_l = torch.max(torch.sum(T_hor.reshape(nx, ny), axis=0))  # the melt pool length

        # measure melt pool depth
        T_ver = torch.relu(T[hor_size:] - Tm) / (T[hor_size:] - Tm + 1e-10)  # For T> Tm, T->1; otherwise T->0
        mp_d = torch.max(torch.sum(T_ver.reshape(nx, nz), axis=1))  # the melt pool depth

        # loss
        #goldak_loss = 100 * torch.sum(torch.relu_(((goldak_ini-goldak_mean)**2)**0.5-goldak_halrang))
        loss = ((((mp_d*length*1e3 - mp_d_exp)) ** 2 + ((mp_w*length*1e3 - mp_w_exp)) ** 2)/2.0 )**0.5

        if (loss < 5.0) and (torch.sum(goldak_ini!=torch.clamp(goldak_ini,
                                                     min=goldak_range[:, 0].to(Ec.device),
                                                     max=goldak_range[:, 1].to(Ec.device)))==0):
            info = {'t':inputs.cpu().detach().numpy()[0,0],'mp width':mp_w,
                    'mp_depth':mp_d,'mp_length':mp_l,
                    'goldak_a_b_cf_cr_absorptivity':goldak_ini.cpu().detach().numpy(),
                    'loss':loss.item()}
            rec[str(i)] = info
        return rec


if __name__ == "__main__":

    Ec = EquationClass()

    model_path = 'network_20230715_153014/'
    model = torch.load(os.path.join(model_path, 'model_full.pkl')) #torch.load(os.path.join(model_path,'model_full.pkl'))
    absorptivity = 0.8
    laser_power = np.array([0.2, 0.175, 0.2, 0.175, 0.125])#, 0.125])
    scan_speed = np.array([0.9, 0.8, 1.1, 0.9, 0.7])#, 0.8])
    mp_d_exp = np.array([123, 122, 99, 107, 98])#, 84])
    mp_w_exp = np.array([110, 107, 105, 101, 102])#, 101])
    Tm = 1307.0
    t = 0.9
    #goldak_fit(Ec, model, t, laser_power, scan_speed, mp_d_exp, mp_w_exp, Tm)

    goldak_search(Ec,model, t, laser_power, scan_speed, mp_d_exp,mp_w_exp,Tm, absorptivity)
    '''
    rec, plt_data = goldak_mc(Ec, model, t, laser_power, scan_speed, mp_d_exp, mp_w_exp, Tm, absorptivity)
    plt_data = np.array(plt_data)
    plt.subplot(3,2,1)
    plt.scatter(plt_data[:,6],plt_data[:,-8])
    plt.subplot(3,2,2)
    plt.scatter(plt_data[:,7],plt_data[:,-8])
    plt.subplot(3,2,3)
    plt.scatter(plt_data[:,8],plt_data[:,-8])
    plt.subplot(3,2,4)
    plt.scatter(plt_data[:,9],plt_data[:,-8])
    plt.subplot(3,2,5)
    plt.scatter(plt_data[:,4]/laser_power[3],plt_data[:,-8])
    plt.show(block=True)
    plt.subplot(3, 2, 1)
    plt.scatter(plt_data[:, -7], plt_data[:, -2])
    plt.subplot(3, 2, 2)
    plt.scatter(plt_data[:, -6], plt_data[:, -2])
    plt.subplot(3, 2, 3)
    plt.scatter(plt_data[:, -5], plt_data[:, -2])
    plt.subplot(3, 2, 4)
    plt.scatter(plt_data[:, -4], plt_data[:, -2])
    plt.subplot(3, 2, 5)
    plt.scatter(plt_data[:, -9] / laser_power, plt_data[:, -2])
    plt.show(block=True)
    print("done")
    '''


