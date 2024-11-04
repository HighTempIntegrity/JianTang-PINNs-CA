from ThermoPINNs import *#EquationBaseClass, SquareDomain, generator_points, DirichletBC, NeumannBC
import numpy as np
import torch


class EquationClass(EquationBaseClass):
    def __init__(self, ):
        EquationBaseClass.__init__(self)

        self.type_of_points = "will be defined from outside"
        self.output_dimension = 1
        self.space_dimensions = 3
        self.time_dimensions = 1

        # define input parameters except for space and time dimensions
        self.parameter_dimensions = 8
        self.parameters_values = torch.tensor([[0.005, 0.02],  # conductivity/1000
                                               [0.002, 0.005],  # density in kg/mm3 *Cp
                                               [0.05, 0.16],  # Laser Power/1000
                                               [0.5, 1.5],  # laser speed mm/s/1000
                                               [0.02, 0.08],  # goldak a width
                                               [0.02, 0.2],  # goldak b depth
                                               [0.02, 0.08],  # goldak cf
                                               [0.02, 0.3]])  # goldak cr
        # define time and space domain
        self.extrema_values = torch.tensor([[-0.1, 1],
                                            [-1, 1.8],
                                            [-1, 1],
                                            [-1, 0.03]])

        # the time is normalized based on the total time for scanning a laser track with "vm" length (unit: mm)
        self.vm = 1.5  # distance the laser travels for t_input=1

        # the output temperature is normaliyed by umax
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
        self.setuparr = np.array([])
        self.a_type = "constant"

    def a(self):
        # for absorbing boundary conditions; did not use in our study
        return 1.0

    def add_collocation_points(self, n_coll, random_seed):
        self.square_domain.type_of_points = self.type_of_points # type_of_points will be defined in PINNS.py
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

    def apply_bc(self, model, x_b_train, u_b_train, u_pred_var_list, u_train_var_list):
        self.square_domain.apply_boundary_conditions(model, x_b_train, u_b_train, u_pred_var_list, u_train_var_list)

    def apply_ic(self, model, x_u_train, u_train, u_pred_var_list, u_train_var_list):
        for j in range(self.output_dimension):
            if x_u_train.shape[0] != 0:
                u = model(x_u_train)[:, j]
                u_pred_var_list.append(u)
                u_train_var_list.append(u_train[:])

    def compute_res(self, network, x_f_train):
        '''
        Compute the residual of PDEs
        :param network: NN model
        :param x_f_train: input of collocation points
        :return: residual of PDEs
        '''
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
        time = self.vm / (x_f_train[:, 7] * 1000)
        q = self.source(x_f_train) * time / (rhocp * self.umax)
        res2 = (grad_u_xx.reshape(-1, ) + grad_u_yy.reshape(-1, ) + grad_u_zz.reshape(-1, )) * time / (rhocp)
        residual = (grad_u_t.reshape(-1, ) - res2 - q)

        # enforce zero-gradient in the region before laser gradient
        mask_init = torch.le(x_f_train[:, 0], 0)
        residual[mask_init] = grad_u_t.reshape(-1, )[mask_init]
        # enforce temperatures above 0
        value = 25
        mask_temp = torch.le(u * self.umax, value)
        residual[mask_temp] = abs(u[mask_temp] * self.umax - value) * residual[mask_temp] / abs(residual[mask_temp]) + \
                              residual[mask_temp]
        return residual

    def v0(self, x):
        return torch.zeros((x.shape[0], 1))

    def ub0(self, t):
        type_BC = [DirichletBC()]
        u0 = 25 / self.umax
        u = torch.full(size=(t.shape[0], 1), fill_value=u0)
        return u, type_BC

    def ub1(self, t):
        '''
        Define boundary conditions
        :param t:
        :return:
        '''
        type_BC = [NeumannBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def ub2(self, t):
        type_BC = [NeumannBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def u0(self, x):
        '''
        Assign 25 as the initial temperature for the whole domain (one can define their own initial condition)
        :param t: inputs of the collocation point for initial condition
        :return: output at these collocation point for initial condition
        '''
        self.ini = 25 / self.umax
        u0 = torch.ones(x.shape[0]) * self.ini
        return u0

    def conductivity(self, u):
        u = u * self.umax
        # c = 0.0000193533 * u + 0.0092405
        c = (0234.887443886408 + 0.018367198 * u + 225.10088 * torch.tanh(0.0118783 * (u - 1543.815))) / 1000
        return c

    def heat_capacity_density(self, u):
        u = u * self.umax
        # rhocp = 0.00000141051 * u + 0.00374571
        rhocp = 8351.910158e-09 * (446.337248 + 0.14180844 * u - 61.431671211432 * torch.exp(
            -0.00031858431233904 * (u - 525) ** 2) + 1054.9650568 * torch.exp(-0.00006287810196136 * (u - 1545) ** 2))
        return rhocp


    def source(self, x):
        '''
        define the Goldak heat source for the PDE
        :param x: inputs of collocation points
        :return: heat input
        '''
        q = x[:, 6] * 1000

        x_phys = x[:, 1]
        y_phys = x[:, 2]
        z_phys = x[:, 3]
        v = self.vm

        a = x[:, 8]  # goldak width
        b = x[:, 9]  # goldak depth
        cf = x[:, 10]  # goldak front length
        cr = x[:, 11]  # goldak rail length

        ff = 2. / (1. + cr / cf)  # ratio of front part
        fr = 2. / (1. + cf / cr)  # ratio of rail part
        timestep = -0. + x[:, 0] * v

        argfact = 1000

        c = cf * (torch.full(q.size(), 0.5).to(self.device) + 0.5 * torch.tanh(argfact * (x_phys - timestep)).to(
            self.device)) + \
            cr * (torch.full(q.size(), 0.5).to(self.device) + 0.5 * torch.tanh(argfact * (timestep - x_phys)).to(
            self.device))
        f = ff * (torch.full(q.size(), 0.5).to(self.device) + 0.5 * torch.tanh(argfact * (x_phys - timestep)).to(
            self.device)) + \
            fr * (torch.full(q.size(), 0.5).to(self.device) + 0.5 * torch.tanh(argfact * (timestep - x_phys)).to(
            self.device))
        c.to(self.device)
        f.to(self.device)

        mask = 1 - 1. / (
                    1 + torch.exp((x[:, 0] - 0.03) * 400))  # have a logistic function as ramp in the beginning at t=0
        heat = (2 * f * q / (a * b * c) * (np.sqrt(3 / np.pi)) ** 3 * torch.exp(
            -3 * ((x_phys - timestep) ** 2 / c ** 2 + y_phys ** 2 / a ** 2 + (z_phys - 0.03) ** 2 / b ** 2)) * mask)

        return heat