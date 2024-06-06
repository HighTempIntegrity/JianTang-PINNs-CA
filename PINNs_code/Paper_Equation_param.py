from ImportFile import *
from EquationBaseClass import EquationBaseClass
from GeneratorPoints import generator_points
from SquareDomain import SquareDomain
from BoundaryConditions import PeriodicBC, DirichletBC, AbsorbingBC, NeumannBC


class EquationClass(EquationBaseClass):

    def __init__(self, ):
        EquationBaseClass.__init__(self)

        self.type_of_points = "will be defined from outside"
        self.output_dimension = 1
        self.space_dimensions = 3
        self.time_dimensions = 1


        self.parameter_dimensions = 8
        self.parameters_values = torch.tensor([[0.005, 0.02],  # conductivity/1000
                                               [0.002, 0.005],  # density in kg/mm3 *Cp
                                               [0.04, 0.16],  # Laser Power/1000
                                               [0.4, 1.4],  # laser speed mm/s/1000
                                               [0.01, 0.1],  # goldak a width
                                               [0.01, 0.23],  # goldak b depth
                                               [0.01, 0.1],  # goldak cf
                                               [0.01, 0.3]]) # goldak cr

        self.extrema_values = torch.tensor([[-0.1, 1],
                                            [-1, 1.8],
                                            [-1, 1],
                                            [-1, 0.03]])
        self.vm=1.5 #distance the laser travels for t_input=1
        self.tmax=0.0015 #not needed with speed as input parameter
        self.umax = 3000
        self.list_of_BC = list([[self.ub1, self.ub1], [self.ub1, self.ub1], [self.ub1, self.ub1]])
        self.extrema_values = self.extrema_values if self.parameters_values is None else torch.cat(
            [self.extrema_values, self.parameters_values], 0)
        self.square_domain = SquareDomain(self.output_dimension,
                                          self.time_dimensions,
                                          self.space_dimensions,
                                          self.list_of_BC,
                                          self.extrema_values,
                                          self.type_of_points,
                                          vel_wave=self.a)
        self.setuparr=np.array([]) #which setups should be used for support?
        if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
            self.val_path= "fem/"
        else:
            self.val_path = "fem/"



        self.a_type = "constant"

    def add_collocation_points(self, n_coll, random_seed):
        self.square_domain.type_of_points = self.type_of_points
        # Plotting x-t collocation points at middle surface
        '''xcoll,ycoll=self.square_domain.add_collocation_points(n_coll, random_seed)
        xcoll=xcoll.detach().numpy()
        mask = xcoll[:,2]>-0.05
        mask1= xcoll[:,2]<0.05
        mask2 = xcoll[:,3]>0
        mask3=mask& mask1 & mask2
        plt.scatter(xcoll[mask3,0], xcoll[mask3,1], s=1)
        plt.show()'''
        # plotting x-y collocation points at surface over time steps for gif
        '''xcoll,ycoll=self.square_domain.add_collocation_points(n_coll, random_seed)
        xcoll=xcoll.detach().numpy()

        path="D:\polybox\Masterthesis\PINN\Taniya\heat3d_empty_test\\Images/"
        for i in range(0,15):
            time=(i+1)/15
            mask1 = xcoll[:, 0] < time
            mask2= xcoll[:, 0] > (i)/15
            mask=mask1&mask2
            fig, (ax) = plt.subplots(ncols=1)
            fig.subplots_adjust(wspace=0.01)
            plt.scatter(xcoll[mask,1], xcoll[mask,2], s=1)

            plt.scatter(0+time*1.5, 0, s=10)
            plt.xlim(-1,1.8)
            plt.ylim(-1,1)
            ax.set(xticklabels=[])
            ax.set(yticklabels=[])
            ax.set_title('Sampled Collocation Points')
            plt.savefig(path + "/collocation" + str(i) + ".png", dpi=500)
            print("plotting collocation",i)
        '''
        return self.square_domain.add_collocation_points(n_coll, random_seed)

    def add_boundary_points(self, n_boundary, random_seed):
        return self.square_domain.add_boundary_points(n_boundary, random_seed)

    def add_initial_points(self, n_initial, random_seed):
        extrema_0 = self.extrema_values[:, 0]  # minimum values
        extrema_f = self.extrema_values[:, 1]  # maximum values
        x_time_0 = generator_points(n_initial, self.time_dimensions + self.space_dimensions + self.parameter_dimensions,
                                    random_seed, "initial_center", True)
        x_time_0[:, 0] = torch.full(size=(n_initial,), fill_value=0.0)
        x_time_0 = x_time_0 * (extrema_f - extrema_0) + extrema_0

        y_time_0 = self.u0(x_time_0)

        return x_time_0, y_time_0

    def a(self, x):
        return 1

    def add_internal_points(self, n_internal, random_seed):
        set = 2  # 1=Rosenthal, 2=FEM
        if set == 1:
            print("Rosenthal Support")
            extrema_0 = self.extrema_values[:, 0]  # minimum values
            extrema_f = self.extrema_values[:, 1]  # maximum values
            x_intern = generator_points(n_internal,
                                        self.time_dimensions + self.space_dimensions + self.parameter_dimensions,
                                        random_seed, self.type_of_points, True)
            x_intern[:, 0] = torch.full(size=(n_internal,), fill_value=0.0)
            x_intern = x_intern * (extrema_f - extrema_0) + extrema_0
            y_intern = self.rosenthal(x_intern)
        else:
            print("FEM Support")
            q=0
            for setup in self.setuparr:
                # load data
                print('importing setup',setup)
                node_coord_df = pd.read_csv(self.val_path + 'nodes.csv', index_col='label')
                test_nodes = node_coord_df.to_numpy()

                temp_df = pd.read_csv(self.val_path + 'sobol_fem/fem_temp' + str(setup) + '.csv', index_col='label')
                Exact = temp_df.to_numpy()
                times = temp_df.columns.to_numpy()
                times = times[10:11]

                param_df = pd.read_csv(self.val_path + 'sobol_fem/parameter.csv', index_col='label')
                param = param_df.to_numpy()
                p = param[setup - 1, :]
                print('Parameters: ',p)
                p_c = torch.cat(
                    [torch.tensor(()).new_full(size=(test_nodes.shape[0], 1), fill_value=p[0]),
                     torch.tensor(()).new_full(size=(test_nodes.shape[0], 1), fill_value=p[1]),
                     torch.tensor(()).new_full(size=(test_nodes.shape[0], 1), fill_value=p[2]),
                     torch.tensor(()).new_full(size=(test_nodes.shape[0], 1), fill_value=p[3]),
                     torch.tensor(()).new_full(size=(test_nodes.shape[0], 1), fill_value=p[4])]
                    , 1)


                # create dataset with different times, mostly shaping into correct form
                i = 0
                for t in times:
                    t_i = np.ones(test_nodes.shape[0]) * float(t)
                    wanted = i
                    test_time = t_i / 1.5 * 1000 * p[3]
                    temp_inp_i = np.column_stack([test_time, test_nodes])
                    temp_out_i = Exact[np.arange(len(Exact)), wanted]
                    temp_out_i = temp_out_i[:, None]
                    # append all time steps into one variable
                    if q==0:
                        test_input = temp_inp_i.astype(np.float32)
                        test_inp_tens = torch.from_numpy(test_input)
                        test_inp_tens = torch.cat([test_inp_tens, p_c], 1)
                        total_input = test_inp_tens
                        output = temp_out_i
                    else:
                        test_input = temp_inp_i.astype(np.float32)
                        test_inp_tens = torch.from_numpy(test_input)
                        test_inp_tens = torch.cat([test_inp_tens, p_c], 1)
                        total_input = torch.cat([total_input, test_inp_tens], 0)
                        output = np.row_stack((output, temp_out_i))
                    i=i+1
                    q=q+1
            total_input=total_input.numpy()
            complete=np.column_stack((total_input,output))
            '''            
            np.random.seed(random_seed)
            np.random.shuffle(complete)
            '''
            print('Num of available support points:', np.shape(complete)[0])
            print('Num of chosen Points:', n_internal)
            assert (n_internal < np.shape(complete)[0]), "Too many support points chosen, dataset too small!"
            dim=self.parameter_dimensions+self.space_dimensions+self.time_dimensions
            selection = complete[:n_internal, :]
            test_inp = selection[:, 0:dim]
            output = selection[:, dim]
            test_inp = test_inp.astype(np.float32)
            test_inp_tens = torch.from_numpy(test_inp)
            x_intern = test_inp_tens

            y_intern = torch.from_numpy(output.astype(np.float32))
            y_intern = torch.reshape(y_intern, (-1,))
        return x_intern, y_intern

    def apply_bc(self, model, x_b_train, u_b_train, u_pred_var_list, u_train_var_list):

        self.square_domain.apply_boundary_conditions(model, x_b_train, u_b_train, u_pred_var_list, u_train_var_list)

    def apply_ic(self, model, x_u_train, u_train, u_pred_var_list, u_train_var_list):
        for j in range(self.output_dimension):
            if x_u_train.shape[0] != 0:
                u = model(x_u_train)[:, j]
                u_pred_var_list.append(u)
                u_train_var_list.append(u_train[:])
                L2=torch.mean((u - u_train[:]) ** 2) / torch.mean((u_train[:]) ** 2)
                max=torch.max(u-u_train[:])
                #print('init+support relative L2',L2.detach().cpu().numpy().round(4))
                #print('init+support max difference',max.detach().cpu().numpy().round(4))

    def compute_res(self, network, x_f_train, solid_object):
        self.network = network
        x_f_train.requires_grad = True
        u = network(x_f_train)[:, 0].reshape(-1, )
        grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones_like(u).to(self.device), create_graph=True)[
            0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_y = grad_u[:, 2]
        grad_u_z = grad_u[:, 3]


        rhocp = self.heat_capacity_density(u)
        c = self.conductivity(u)
        grad_u_xx = torch.autograd.grad(c * grad_u_x, x_f_train, grad_outputs=torch.ones_like(u).to(self.device),
                                        create_graph=True)[0][:, 1]
        grad_u_yy = torch.autograd.grad(c * grad_u_y, x_f_train, grad_outputs=torch.ones_like(u).to(self.device),
                                        create_graph=True)[0][:, 2]
        grad_u_zz = torch.autograd.grad(c * grad_u_z, x_f_train, grad_outputs=torch.ones_like(u).to(self.device),
                                        create_graph=True)[0][:, 3]
        time = self.vm/(x_f_train[:, 7]*1000)
        q = self.source(x_f_train) * time / (rhocp * self.umax)
        res2 = (grad_u_xx.reshape(-1, ) + grad_u_yy.reshape(-1, ) + grad_u_zz.reshape(-1, )) * time / (rhocp)
        residual = (grad_u_t.reshape(-1, ) - res2 - q)

        # enforce zero-gradient in the region before laser gradient
        mask_init = torch.le(x_f_train[:, 0], 0)
        residual[mask_init] = grad_u_t.reshape(-1, )[mask_init]
        # enforce temperatures above 0
        value=25
        mask_temp = torch.le(u*self.umax, value)
        residual[mask_temp] = abs(u[mask_temp]*self.umax-value)*residual[mask_temp]/abs(residual[mask_temp]) + residual[mask_temp]

        #debugging printout
        #print('--Residual--')
        #print("Max du/dt", torch.max(abs(grad_u_t.reshape(-1, ))).detach().cpu().numpy().round(4), "; mean: ",
        #      torch.mean(grad_u_t.reshape(-1, )).detach().cpu().numpy().round(4))
        #print("Max d/dx(c*du/dx)", torch.max(abs(res2)).detach().cpu().numpy().round(4), "; mean: ",
        #      torch.mean(res2).detach().cpu().numpy().round(4))
        #print("Max source", (torch.max(abs(q))).detach().cpu().numpy().round(4), "; mean: ",
        #      (torch.mean(q)).detach().cpu().numpy().round(4))
        #print("max predicted temp: ", (torch.max((u * self.umax))).detach().cpu().numpy().round(4), "min temp: ",
        #      (torch.min((u * self.umax))).detach().cpu().numpy().round(4))
        #print("Max residual: ", torch.max(abs(residual)).detach().cpu().numpy().round(4), "; Mean: ",
        #      torch.mean(residual).detach().cpu().numpy().round(4))

        return residual

    def v0(self, x):
        return torch.zeros((x.shape[0], 1))

    def ub0(self, t):
        type_BC = [DirichletBC()]
        u0 = 25 / self.umax
        u = torch.full(size=(t.shape[0], 1), fill_value=u0)
        return u, type_BC

    def ub1(self, t):
        type_BC = [NeumannBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def ub2(self, t):
        type_BC = [NeumannBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def u0(self, x):
        self.ini = 25 / self.umax
        u0 = torch.ones(x.shape[0]) * self.ini
        return u0

    def conductivity(self, u):
        u = u * self.umax
        #c = 0.0000193533 * u + 0.0092405
        c = (0234.887443886408 + 0.018367198 * u +  225.10088 * torch.tanh(0.0118783 * (u -1543.815)))/1000
        return c
  
    def heat_capacity_density(self, u):
        u = u * self.umax
        #rhocp = 0.00000141051 * u + 0.00374571
        rhocp = 8351.910158e-09 * (446.337248 + 0.14180844 * u - 61.431671211432 *torch.exp(-0.00031858431233904*(u-525)**2) +1054.9650568*torch.exp(-0.00006287810196136*(u-1545)**2))
        return rhocp

    def rosenthal(self, x):
        # c=conductivity, a=diffusivity
        sig = 0.05
        q = 150
        n = 0.5
        T0 = 25
        # make all x physical
        c = x[:, 4] * 1000
        rho = 8220
        cp = 605
        a = c / (rho * cp)
        x_phys = x[:, 1] / 1000  # *self.xmax-1
        y_phys = x[:, 2] / 1000  # *self.ymax-1
        z_phys = x[:, 3] / 1000
        norm = self.norm_time / self.tmax
        v = 1.5 / self.tmax / 1000
        timestep = 0 / 1000 + x[:, 0] / norm * 1.5 / 1000
        r = ((x_phys - timestep) ** 2 + y_phys ** 2 + (z_phys - 0.03) ** 2) ** 0.5
        T = T0 + (q * n) / (2 * pi * c * r) * torch.exp(-(v * (x_phys - timestep + r)) / (2 * a))
        # heat=heat/torch.max(heat)
        return T / self.umax
        

    def source(self, x):
        
        q = x[:,6]*1000
        
        x_phys = x[:, 1]
        y_phys = x[:, 2]
        z_phys = x[:, 3]
        v = self.vm
        #v=x[:,7]*1000*self.tmax

        a=x[:,8] #goldak width
        b=x[:,9] # goldak depth
        cf=x[:,10] # goldak front length
        cr=x[:,11] # goldak rail length

        ff=2./(1.+cr/cf) # ratio of front part
        fr=2./(1.+cf/cr) # ratio of rail part
        timestep = -0. + x[:, 0] * v

        argfact=1000

        c=cf*(torch.full(q.size(),0.5).to(self.device) + 0.5*torch.tanh(argfact*(x_phys-timestep)).to(self.device)) +\
          cr*(torch.full(q.size(),0.5).to(self.device) + 0.5*torch.tanh(argfact*(timestep-x_phys)).to(self.device))
        f=ff*(torch.full(q.size(),0.5).to(self.device) + 0.5*torch.tanh(argfact*(x_phys-timestep)).to(self.device)) +\
          fr*(torch.full(q.size(),0.5).to(self.device) + 0.5*torch.tanh(argfact*(timestep-x_phys)).to(self.device))
        c.to(self.device)
        f.to(self.device)

        mask = 1 - 1. / (1 + torch.exp((x[:, 0] - 0.03) * 400)) #have a logistic function as ramp in the beginning at t=0
        heat = (2*f*q/ (a*b*c) *(np.sqrt(3/np.pi))**3 * torch.exp(-3 * ((x_phys-timestep)**2 /c**2 + y_phys**2/a**2 + (z_phys-0.03)**2 /b**2)) * mask)

        # heat=heat/torch.max(heat)
        return heat


    def compute_generalization_error(self, model, extrema, val_path=None, images_path=None):
        set = np.array([2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 24])
        dif = [i for i in set if i not in self.setuparr]
        diff_array=np.array(dif)
        print('trained with ',self.setuparr)
        print('testing on ',diff_array)
        Error_out=0
        relError_out=0
        q=0

        for setup in diff_array:
            node_coord_df = pd.read_csv(self.val_path + 'nodes.csv', index_col='label')
            temp_df = pd.read_csv(self.val_path + 'mat_fem/Temp' + str(setup) + '.csv', index_col='label')
            param_df = pd.read_csv(self.val_path + 'mat_fem/material_params.csv', index_col='index')

            test_nodes = node_coord_df.to_numpy()
            Exact = temp_df.to_numpy()
            max_T = np.max(Exact)
            times = temp_df.columns.to_numpy()
            times = times[:16] #only steady state times are evaluated for performance
            tmax = max(times)
            param = param_df.to_numpy()
            # create dataset with different times, mostly shaping into correct form
            max_array = np.zeros((3, temp_df.columns.size))
            i = 0
            T_max_pred = 0
            p = param[setup - 1, :]
            p_c = torch.cat(
                [torch.tensor(()).new_full(size=(test_nodes.shape[0], 1), fill_value=p[0]),
                 torch.tensor(()).new_full(size=(test_nodes.shape[0], 1), fill_value=p[1]),
                 torch.tensor(()).new_full(size=(test_nodes.shape[0], 1), fill_value=p[2]),
                 torch.tensor(()).new_full(size=(test_nodes.shape[0], 1), fill_value=p[3])]
                , 1)
            for t in times:

                t_i = np.ones(test_nodes.shape[0]) * float(t)
                wanted = i
                test_time = t_i / self.vm * 1000 * p[3]
                temp_inp_i = np.column_stack([test_time, test_nodes])
                temp_out_i = Exact[np.arange(len(Exact)), wanted]
                temp_out_i = temp_out_i[:, None]

                test_input = temp_inp_i
                output_exact = temp_out_i
                test_inp = test_input.astype(np.float32)
                test_inp_tens = torch.from_numpy(test_inp)
                test_inp_tens = torch.cat([test_inp_tens, p_c], 1)
                test_out = model(test_inp_tens.to('cpu'))[:, 0].cpu().detach().numpy().reshape(-1, 1) * self.umax
                assert (output_exact.shape[1] == test_out.shape[1])
                max_err = np.max(abs(output_exact - test_out))
                max_rel = max_err / np.max(output_exact)
                avg_err_i = np.average(abs(output_exact - test_out) / np.max(output_exact))
                if max(test_out) > T_max_pred:
                    T_max_pred = max(test_out)
                max_array[0, i] = max_err
                max_array[1, i] = max_rel
                max_array[2, i] = avg_err_i
                i = i + 1
            Error_out=Error_out+np.average(max_array[2, 4:16])
            relError_out=relError_out+np.average(max_array[1, 4:16])
            q=q+1
        L2_test = Error_out/q
        print("average of Relative average error Test:", L2_test)
        rel_L2_test = relError_out/q
        print("average of Relative maximum error Test:", rel_L2_test)
        return L2_test, rel_L2_test

    def plotting(self, model, images_path, extrema, solid):
        model.cpu()
        model = model.eval()
        eval_dim = 2000
        x = torch.reshape(torch.linspace(extrema[1, 0], extrema[1, 1], eval_dim), [eval_dim, 1])
        y = torch.reshape(torch.linspace(extrema[2, 0], extrema[2, 1], eval_dim), [eval_dim, 1])
        t1 = 0
        t2 = 0.0012 / self.norm_time
        z1 = 0.03
        # Plot the surface.
        X = x.numpy()
        Y = y.numpy()
        X, Y = np.meshgrid(X, Y)
        x_t = torch.reshape(torch.from_numpy(X), [eval_dim ** 2, 1])
        y_t = torch.reshape(torch.from_numpy(Y), [eval_dim ** 2, 1])
        plot_var1 = torch.cat([torch.tensor(()).new_full(size=(eval_dim ** 2, 1), fill_value=t1), x_t, y_t,
                               torch.tensor(()).new_full(size=(eval_dim ** 2, 1), fill_value=z1)], 1)
        plot_var2 = torch.cat([torch.tensor(()).new_full(size=(eval_dim ** 2, 1), fill_value=t2), x_t, y_t,
                               torch.tensor(()).new_full(size=(eval_dim ** 2, 1), fill_value=z1)], 1)


        Z3 = model(plot_var1)[:, 0]
        Z3 = torch.reshape(Z3, [eval_dim, eval_dim])
        Z3 = Z3.detach().numpy() * self.umax
        Z4 = model(plot_var2)[:, 0]
        Z4 = torch.reshape(Z4, [eval_dim, eval_dim])
        Z4 = Z4.detach().numpy() * self.umax

        integral1 = (np.sum(Z4) - np.sum(Z3)) * 0.05 * 2 * 2.8 / eval_dim ** 2 * 8220e-9 * 605
        integral2 = (np.sum(Z4 - 25)) * 0.05 * 2 * 2.8 / eval_dim ** 2 * 8220e-9 * 605

        print('PINN initial, power input: ', integral1)
        print('optimal initial, power input: ', integral2)
        print('real power input: ', 0.5 * 150 * (t2 - t1) * self.tmax)

        val_path = self.val_path
        node_coord_df = pd.read_csv(val_path + 'nodes.csv', index_col='label')
        temp_df = pd.read_csv(val_path + 'time12.csv', index_col='label')
        test_inp = node_coord_df.to_numpy()

        Exact = temp_df.to_numpy()
        edata = np.column_stack([test_inp, Exact])
        mask = (edata[:, 2] == z1)
        edata = edata[mask, :]
        x = edata[:, 0]
        y = edata[:, 1]
        z = edata[:, 2]
        T = edata[:, 3]
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        z = np.array(z, dtype=np.float32)
        x_t = torch.reshape(torch.from_numpy(x), [np.size(x), 1])
        y_t = torch.reshape(torch.from_numpy(y), [np.size(x), 1])
        z_t = torch.reshape(torch.from_numpy(z), [np.size(x), 1])
        t_t = torch.tensor(()).new_full(size=(np.size(x), 1), fill_value=t2)
        p_c= torch.tensor(()).new_full(size=(np.size(x), 1), fill_value=0.1)
        p_v = torch.tensor(()).new_full(size=(np.size(x), 1), fill_value=1.5)
        p_p = torch.tensor(()).new_full(size=(np.size(x), 1), fill_value=150)
        p_r = torch.tensor(()).new_full(size=(np.size(x), 1), fill_value=0.05)

        plot_var2 = torch.cat([t_t, x_t, y_t, z_t, p_c, p_v, p_p, p_r], 1)
        Z4 = model(plot_var2)[:, 0]
        Z4 = Z4.detach().numpy() * self.umax

        Dif = abs(T - Z4) / max(Z4)
        difference = max(abs(Dif))
        return difference
