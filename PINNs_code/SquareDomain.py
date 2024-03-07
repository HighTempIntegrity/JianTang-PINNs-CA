import torch
import numpy as np
from GeneratorPoints import generator_points
from BoundaryConditions import NeumannBC


class SquareDomain:
    def __init__(self, output_dimension, time_dimensions, space_dimensions, list_of_BC, extrema_values, type_of_points, parameters_values=None, vel_wave=None):
        print(parameters_values)
        self.output_dimension = output_dimension
        self.time_dimensions = time_dimensions
        self.space_dimensions = space_dimensions
        self.list_of_BC = list_of_BC
        self.extrema_values = extrema_values if parameters_values is None else torch.cat([extrema_values, parameters_values],0)
        self.type_of_points = type_of_points
        self.input_dimensions = self.extrema_values.shape[0]
        self.extrema_0 = self.extrema_values[:, 0]
        self.extrema_f = self.extrema_values[:, 1]
        self.vel_wave = vel_wave
        self.domain_extrema = self.extrema_values[:,0:4]

        self.BC = list()

    def add_collocation_points(self, n_coll, random_seed):

        x_coll = generator_points(n_coll, self.input_dimensions, random_seed, self.type_of_points, False)
        x_coll = x_coll * (self.extrema_f - self.extrema_0) + self.extrema_0
        y_coll = torch.full((n_coll, self.output_dimension), np.nan)
        return x_coll, y_coll

    def add_boundary_points(self, n_boundary, random_seed):
        x_list_b = list()
        y_list_b = list()

        for i in range(self.time_dimensions, self.time_dimensions + self.space_dimensions):
            BC_01 = list()
            val_0 = np.delete(self.extrema_values, i, 0)[:, 0]
            val_f = np.delete(self.extrema_values, i, 0)[:, 1]
            x_boundary_0 = generator_points(n_boundary, self.input_dimensions, random_seed, "sobol", True)          

            x_boundary_0[:, i] = torch.full(size=(n_boundary,), fill_value=0.0)

            x_boundary_0_wo_i = np.delete(x_boundary_0, i, 1)        

            [y_boundary_0, type_BC] = self.list_of_BC[i - self.time_dimensions][0](x_boundary_0_wo_i * (val_f - val_0) + val_0)

            BC_01.append(type_BC)
            x_list_b.append(x_boundary_0)
            y_list_b.append(y_boundary_0)
            x_boundary_1 = generator_points(n_boundary, self.input_dimensions, random_seed, "sobol", True)
            x_boundary_1[:, i] = torch.tensor(()).new_full(size=(n_boundary,), fill_value=1.0)
            x_boundary_1_wo_i = np.delete(x_boundary_1, i, 1)

            [y_boundary_1, type_BC] = self.list_of_BC[i - self.time_dimensions][1](
                x_boundary_1_wo_i * (val_f - val_0) + val_0)
            BC_01.append(type_BC)

            self.BC.append(BC_01)

            x_list_b.append(x_boundary_1)
            y_list_b.append(y_boundary_1)

        x_b = torch.cat(x_list_b, 0)
        y_b = torch.cat(y_list_b, 0)

        x_b = x_b * (self.extrema_f - self.extrema_0) + self.extrema_0
        return x_b, y_b

    '''def add_boundary_points(self, n_boundary, random_seed):

        self.BC = self.list_of_BC
        
        domain_dimensions = self.time_dimensions + self.space_dimensions

        input_boundary = generator_points(6 * n_boundary, self.input_dimensions, random_seed, "sobol", True)
        input_boundary = input_boundary * (self.extrema_f - self.extrema_0) + self.extrema_0

        for i in range(self.time_dimensions, domain_dimensions):
            x0 = self.domain_extrema[i, 0]
            xL = self.domain_extrema[i, 1]

            #self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
            # soboleng = generator_points(n_boundary, domain_dimensions, random_seed, "sobol", True)
            # torch.random.manual_seed(random_seed)
            # input_boundary = torch.rand([n_boundary, 2]).type(torch.FloatTensor)
            #input_boundary = self.convert(input_boundary)


            input_boundary[n_boundary*(2*i-2):n_boundary*(2*i-1), i] = torch.full((1,n_boundary), x0)
            input_boundary[n_boundary*(2*i-1):n_boundary*(2*i), i] = torch.full((1,n_boundary), xL)

            # input_boundary_0 = torch.clone(soboleng)
            #input_boundary_0[:, i] = torch.full(input_boundary_0[:, i].shape, x0)

            #input_boundary_L = torch.clone(soboleng)
            #input_boundary_L[:, i] = torch.full(input_boundary_L[:, i].shape, xL)
     
        #output_boundary_0 = torch.zeros((input_boundary.shape[0], 1))
        #output_boundary_L = torch.zeros((input_boundary.shape[0], 1))

        output_boundary = torch.zeros((input_boundary.shape[0], 1))

        self.BC = [[[NeumannBC()], [NeumannBC()]],
                    [[NeumannBC()], [NeumannBC()]],
                    [[NeumannBC()], [NeumannBC()]]]
        
        return input_boundary, output_boundary'''

    def apply_boundary_conditions(self, model, x_b_train, u_b_train, u_pred_var_list, u_train_var_list):

        for j in range(self.output_dimension):
            for i in range(self.space_dimensions):
                half_len_x_b_train_i = int(x_b_train.shape[0] / (2 * self.space_dimensions))

                x_b_train_i = x_b_train[i * int(x_b_train.shape[0] / self.space_dimensions):(i + 1) * int(x_b_train.shape[0] / self.space_dimensions), :]
                u_b_train_i = u_b_train[i * int(x_b_train.shape[0] / self.space_dimensions):(i + 1) * int(x_b_train.shape[0] / self.space_dimensions), :]

                boundary = 0
                while boundary < 2:
                    x_b_train_i_half = x_b_train_i[half_len_x_b_train_i * boundary:half_len_x_b_train_i * (boundary + 1), :]
                    u_b_train_i_half = u_b_train_i[half_len_x_b_train_i * boundary:half_len_x_b_train_i * (boundary + 1), :]

                    x_half_1 = x_b_train_i_half
                    x_half_2 = x_b_train_i[half_len_x_b_train_i * (boundary + 1):half_len_x_b_train_i * (boundary + 2), :]

                    boundary_conditions = self.BC[i][boundary][j]
                    boundary = boundary_conditions.apply(model, x_half_1, u_b_train_i_half, j, u_pred_var_list, u_train_var_list, space_dim = i,  x_boundary_sym=x_half_2, boundary=boundary, vel_wave=self.vel_wave)

                    boundary = boundary + 1
