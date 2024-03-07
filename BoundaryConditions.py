import torch


class DirichletBC:
    def __init__(self):
        pass

    def apply(self, model, x_boundary, u_boundary, n_out, u_pred_var_list, u_train_var_list, space_dim = None, x_boundary_sym=None, boundary=None, vel_wave=None):
        u_boundary_pred = model(x_boundary)
        u_pred_var_list.append(u_boundary_pred[:, n_out])
        u_train_var_list.append(u_boundary[:, n_out])
        return boundary
class NeumannBC:
    def __init__(self):
        pass

    def apply(self, model, x_boundary, u_boundary, n_out, u_pred_var_list, u_train_var_list, space_dim = None, x_boundary_sym=None, boundary=None, vel_wave=None):
        x_boundary.requires_grad=True
        u_boundary_pred = model(x_boundary)
        grad_u = torch.autograd.grad(u_boundary_pred, x_boundary, grad_outputs=torch.ones_like(u_boundary_pred), create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_n = grad_u[:, 1 + space_dim]

        if boundary == 1:
            n = torch.ones_like(grad_u_n)
        else:
            n = -torch.ones_like(grad_u_n)

        absorb = grad_u_n
        u_pred_var_list.append(absorb)
        u_train_var_list.append(torch.zeros_like(absorb))
        return boundary


class InitBC:
    def __init__(self):
        pass

    def apply(self, model, x_boundary, u_boundary, n_out, u_pred_var_list, u_train_var_list, space_dim = None, x_boundary_sym=None, boundary=None, vel_wave=None):
        x_boundary.requires_grad=True
        u_boundary_pred = model(x_boundary)
        grad_u = torch.autograd.grad(u_boundary_pred, x_boundary, grad_outputs=torch.ones_like(u_boundary_pred), create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_n = grad_u[:, 1 + space_dim]

        if boundary == 1:
            n = torch.ones_like(grad_u_n)
        else:
            n = -torch.ones_like(grad_u_n)

        absorb = grad_u_n
        u_pred_var_list.append(absorb)
        u_train_var_list.append(torch.zeros_like(absorb))
        return boundary



class PeriodicBC:
    def __init__(self):
        pass

    def apply(self, model, x_boundary, u_boundary, n_out, u_pred_var_list, u_train_var_list, space_dim = None,x_boundary_sym=None, boundary=None, vel_wave=None):
        u_boundary_pred = model(x_boundary)
        u_boundary_pred_sym = model(x_boundary_sym)
        u_pred_var_list.append(u_boundary_pred[:, n_out])
        u_train_var_list.append(u_boundary_pred_sym[:, n_out])
        boundary = boundary + 1
        return boundary

class AbsorbingBC:
    def __init__(self):
        pass

    def apply(self, model, x_boundary, u_boundary, n_out, u_pred_var_list, u_train_var_list, space_dim = None, x_boundary_sym=None, boundary=None, vel_wave=None):
        x_boundary.requires_grad=True
        u_boundary_pred = model(x_boundary)
        grad_u = torch.autograd.grad(u_boundary_pred, x_boundary, grad_outputs=torch.ones_like(u_boundary_pred), create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_n = grad_u[:, 1 + space_dim]

        if boundary == 1:
            n = torch.ones_like(grad_u_n)
        else:
            n = -torch.ones_like(grad_u_n)

        absorb = grad_u_t + n*vel_wave(x_boundary)*grad_u_n
        u_pred_var_list.append(absorb)
        u_train_var_list.append(torch.zeros_like(absorb))
        return boundary

