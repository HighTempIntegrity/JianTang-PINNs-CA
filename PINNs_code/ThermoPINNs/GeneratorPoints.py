from pyDOE import lhs
import numpy as np
import sobol_seq
import torch


def generator_points(samples, dim, random_seed, type_of_points, boundary):
    if type_of_points == "random":
        torch.random.manual_seed(random_seed)
        return torch.rand([samples, dim]).type(torch.FloatTensor)
    elif type_of_points == "lhs":
        return torch.from_numpy(lhs(dim, samples=samples, criterion='center')).type(torch.FloatTensor)
    elif type_of_points == "gauss":
        if samples != 0:

            x, _ = np.polynomial.legendre.leggauss(samples)
            x = 0.5 * (x.reshape(-1, 1) + 1)

            if dim == 1:
                return torch.from_numpy(x).type(torch.FloatTensor)
            if dim == 2:
                x = x.reshape(-1, )
                x = np.transpose([np.repeat(x, len(x)), np.tile(x, len(x))])
                return torch.from_numpy(x).type(torch.FloatTensor)
        else:
            return torch.zeros([0, dim])
    elif type_of_points == "grid":
        if samples != 0:

            x = np.linspace(0, 1, samples + 2)
            x = x[1:-1].reshape(-1, 1)
            if dim == 1:
                return torch.from_numpy(x).type(torch.FloatTensor)
            if dim == 2:
                x = x.reshape(-1, )
                if not boundary:
                    x = np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])
                else:
                    x = np.concatenate([x.reshape(-1, 1), x.reshape(-1, 1)], 1)
                print(x)
                return torch.from_numpy(x).type(torch.FloatTensor)
        else:
            return torch.zeros([0, dim])
    #the following work also for dim>3
    elif type_of_points == "uniform":
        print(type_of_points)
        data = np.random.uniform(0,1,(samples, dim))
        return torch.from_numpy(data).type(torch.FloatTensor)
    elif type_of_points == "sobol":
        print(type_of_points)
        skip = random_seed
        data = np.full((samples, dim), np.nan)
        for j in range(samples):
            seed = j + skip
            data[j, :], next_seed = sobol_seq.i4_sobol(dim, seed)
        return torch.from_numpy(data).type(torch.FloatTensor)
    #new formatting
    elif type_of_points == "y_norm":
        print(type_of_points)
        skip = random_seed
        data1 = np.full((samples, 2), np.nan)
        data2 = np.full((samples, 1), np.nan)
        data3 = np.full((samples, dim-3), np.nan)

        data2[:, 0] = np.random.normal(0.5, 0.1, samples)
        for j in range(samples):
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim-1, seed)
            data1[j, :] = rnd[0:2]
            data3[j, :] = rnd[2:]
        data=np.concatenate((data1, data2, data3), axis=1)
        return torch.from_numpy(data).type(torch.FloatTensor)
    elif type_of_points == "y_lap":
        print(type_of_points)
        skip = random_seed
        data1 = np.full((samples, 2), np.nan)
        data2 = np.full((samples, 1), np.nan)
        data3 = np.full((samples, dim-3), np.nan)
        for j in range(samples):
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim-1, seed)
            data1[j, :] = rnd[0:2]
            data2[j,:] = np.random.laplace(0.5, 1/12., 1)
            data3[j, :] = rnd[2:]
        data=np.concatenate((data1, data2, data3), axis=1)
        return torch.from_numpy(data).type(torch.FloatTensor)
    elif type_of_points == "z_norm":
        print(type_of_points)
        skip = random_seed
        data1 = np.full((samples, 3), np.nan)
        data2 = np.full((samples, 1), np.nan)
        data3 = np.full((samples, dim-4), np.nan)
        xa, xb = 0, 1
        loc = 1
        scale = 0.25
        a = (xa - loc) / scale
        b = (xb - loc) / scale
        from scipy.stats import truncnorm
        data2[:,0] = truncnorm.rvs(a, b, loc=loc, scale=scale, size=samples)
        for j in range(samples):
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim-1, seed)
            data1[j, :] = rnd[0:3]
            #data2[j,:] = 1 - abs(np.random.normal(0, 0.25, 1))
            if (dim > 4):
                data3[j, :] = rnd[3:]
        if dim > 4:
            data = np.concatenate((data1, data2, data3), axis=1)
        else:
            data = np.concatenate((data1, data2), axis=1)
        return torch.from_numpy(data).type(torch.FloatTensor)
    elif type_of_points == "z_beta":
        print(type_of_points)
        skip = random_seed
        data1 = np.full((samples, 3), np.nan)
        data2 = np.full((samples, 1), np.nan)
        data3 = np.full((samples, dim-4), np.nan)
        data2[:, 0] = np.random.beta(5, 1, samples)
        for j in range(samples):
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim-1, seed)
            data1[j, :] = rnd[0:3]
            if (dim > 4):
                data3[j, :] = rnd[3:]
        if dim > 4:
            data = np.concatenate((data1, data2, data3), axis=1)
        else:
            data = np.concatenate((data1, data2), axis=1)
        return torch.from_numpy(data).type(torch.FloatTensor)
    elif type_of_points == "yz_norm_norm":
        print(type_of_points)
        skip = random_seed
        data1 = np.full((samples, 2), np.nan)
        dataY = np.full((samples, 1), np.nan)
        dataZ = np.full((samples, 1), np.nan)
        dataP = np.full((samples, dim-4), np.nan)
        xa, xb = 0, 1
        loc = 1
        scale = 0.25
        a = (xa - loc) / scale
        b = (xb - loc) / scale
        from scipy.stats import truncnorm
        dataZ[:,0] = truncnorm.rvs(a, b, loc=loc, scale=scale, size=samples)
        dataY[:, 0] = np.random.normal(0.5, 0.1, samples)
        for j in range(samples):
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim-2, seed)
            data1[j, :] = rnd[0:2]
            if (dim > 4):
                dataP[j, :] = rnd[2:]
        if dim > 4:
            data = np.concatenate((data1, dataY ,dataZ, dataP), axis=1)
        else:
            data = np.concatenate((data1, dataY ,dataZ), axis=1)
        return torch.from_numpy(data).type(torch.FloatTensor)
    elif type_of_points == "yz_norm_beta":
        print(type_of_points)
        skip = random_seed
        data1 = np.full((samples, 2), np.nan)
        dataY = np.full((samples, 1), np.nan)
        dataZ = np.full((samples, 1), np.nan)
        dataP = np.full((samples, dim-4), np.nan)
        dataY[:, 0] = np.random.normal(0.5, 0.1, samples)
        dataZ[:, 0] = np.random.beta(5, 1, samples)
        for j in range(samples):
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim-2, seed)
            data1[j, :] = rnd[0:2]
            if (dim > 4):
                dataP[j, :] = rnd[2:]
        if dim > 4:
            data = np.concatenate((data1, dataY ,dataZ, dataP), axis=1)
        else:
            data = np.concatenate((data1, dataY ,dataZ), axis=1)
        return torch.from_numpy(data).type(torch.FloatTensor)
    elif type_of_points == "yz_lap_norm":
        print(type_of_points)
        skip = random_seed
        data1 = np.full((samples, 2), np.nan)
        dataY = np.full((samples, 1), np.nan)
        dataZ = np.full((samples, 1), np.nan)
        dataP = np.full((samples, dim-4), np.nan)
        xa, xb = 0, 1
        loc = 1
        scale = 0.25
        a = (xa - loc) / scale
        b = (xb - loc) / scale
        from scipy.stats import truncnorm
        dataZ[:,0] = truncnorm.rvs(a, b, loc=loc, scale=scale, size=samples)
        dataY[:, 0] = np.random.laplace(0.5, 1 / 12., samples)
        for j in range(samples):
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim-2, seed)
            data1[j, :] = rnd[0:2]
            if (dim > 4):
                dataP[j, :] = rnd[2:]
        if dim > 4:
            data = np.concatenate((data1, dataY ,dataZ, dataP), axis=1)
        else:
            data = np.concatenate((data1, dataY ,dataZ), axis=1)
        return torch.from_numpy(data).type(torch.FloatTensor)
    elif type_of_points == "yz_lap_beta":
        print(type_of_points)
        skip = random_seed
        data1 = np.full((samples, 2), np.nan)
        dataY = np.full((samples, 1), np.nan)
        dataZ = np.full((samples, 1), np.nan)
        dataP = np.full((samples, dim-4), np.nan)
        dataY[:, 0] = np.random.laplace(0.5, 1 / 12., samples)
        dataZ[:, 0] = np.random.beta(5, 1, samples)
        for j in range(samples):
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim-2, seed)
            data1[j, :] = rnd[0:2]
            if (dim > 4):
                dataP[j, :] = rnd[2:]
        if dim > 4:
            data = np.concatenate((data1, dataY ,dataZ, dataP), axis=1)
        else:
            data = np.concatenate((data1, dataY ,dataZ), axis=1)
        return torch.from_numpy(data).type(torch.FloatTensor)
    elif type_of_points == "x_tri_move":
        print(type_of_points)
        skip = random_seed
        x0=1 #start of the laser
        xspeed=1.5 #speed of the laser in mm/t_max
        xsize=2.8 #total length of domain
        x0=x0/xsize #normed between one and 0
        xspeed=xspeed/xsize #normed to the size of the domain
        datat = np.full((samples, 1), np.nan)
        dataX = np.full((samples, 1), np.nan)
        dataP = np.full((samples, dim-2), np.nan)
        for j in range(samples): #majority of coll points sampled around laser
            if j % 50000 ==0:
                print(j/samples *100, "% done")
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim-1, seed)
            datat[j, :]=rnd[0]
            dataX[j,:] = np.random.triangular(0,x0+datat[j, :]*xspeed, 1)
            dataP[j,:] = rnd[1:]
        data=np.concatenate((datat, dataX, dataP), axis=1)
        return torch.from_numpy(data).type(torch.FloatTensor)
    elif type_of_points == "xy_tri_norm":
        print(type_of_points)
        skip = random_seed
        x0=1 #start of the laser
        xspeed=1.5 #speed of the laser in mm/t_max
        xsize=2.8 #total length of domain
        x0=x0/xsize #normed between one and 0
        xspeed=xspeed/xsize #normed to the size of the domain
        datat = np.full((samples, 1), np.nan)
        dataX = np.full((samples, 1), np.nan)
        dataY = np.full((samples, 1), np.nan)
        dataP = np.full((samples, dim-3), np.nan)

        dataY[:, 0] = np.random.normal(0.5, 0.1, samples)
        for j in range(samples): #majority of coll points sampled around laser
            if j % 50000 ==0:
                print(j/samples *100, "% done")
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim-2, seed)
            datat[j, :]=rnd[0]
            dataX[j,:] = np.random.triangular(0,x0+datat[j, :]*xspeed, 1)
            dataP[j,:] = rnd[1:]
        data=np.concatenate((datat, dataX,dataY, dataP), axis=1)
        return torch.from_numpy(data).type(torch.FloatTensor)
    elif type_of_points == "xy_tri_lap":
        print(type_of_points)
        skip = random_seed
        x0=1 #start of the laser
        xspeed=1.5 #speed of the laser in mm/t_max
        xsize=2.8 #total length of domain
        x0=x0/xsize #normed between one and 0
        xspeed=xspeed/xsize #normed to the size of the domain
        datat = np.full((samples, 1), np.nan)
        dataX = np.full((samples, 1), np.nan)
        dataY = np.full((samples, 1), np.nan)
        dataP = np.full((samples, dim-3), np.nan)
        dataY[:, 0] = np.random.laplace(0.5, 1 / 12., samples)
        for j in range(samples): #majority of coll points sampled around laser
            if j % 50000 ==0:
                print(j/samples *100, "% done")
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim-2, seed)
            datat[j, :]=rnd[0]
            dataX[j,:] = np.random.triangular(0,x0+datat[j, :]*xspeed, 1)
            dataP[j,:] = rnd[1:]
        data=np.concatenate((datat, dataX,dataY, dataP), axis=1)
        return torch.from_numpy(data).type(torch.FloatTensor)
    elif type_of_points == "xz_tri_norm":
        print(type_of_points)
        skip = random_seed
        x0=1 #start of the laser
        xspeed=1.5 #speed of the laser in mm/t_max
        xsize=2.8 #total length of domain
        x0=x0/xsize #normed between one and 0
        xspeed=xspeed/xsize #normed to the size of the domain
        datat = np.full((samples, 1), np.nan)
        dataX = np.full((samples, 1), np.nan)
        dataY = np.full((samples, 1), np.nan)
        dataZ = np.full((samples, 1), np.nan)
        xa, xb = 0, 1
        loc = 1
        scale = 0.25
        a = (xa - loc) / scale
        b = (xb - loc) / scale
        from scipy.stats import truncnorm
        dataZ[:,0] = truncnorm.rvs(a, b, loc=loc, scale=scale, size=samples)
        dataP = np.full((samples, dim-4), np.nan)
        for j in range(samples): #majority of coll points sampled around laser
            if j % 50000 ==0:
                print(j/samples *100, "% done")
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim-2, seed)
            datat[j, :]=rnd[0]
            dataX[j,:] = np.random.triangular(0,x0+datat[j, :]*xspeed, 1)
            dataY[j,:]=rnd[1]
            if (dim > 4):
                dataP[j, :] = rnd[2:]
        if dim > 4:
            data=np.concatenate((datat, dataX, dataY, dataZ, dataP), axis=1)
        else:
            data=np.concatenate((datat, dataX, dataY, dataZ), axis=1)
        return torch.from_numpy(data).type(torch.FloatTensor)
    elif type_of_points == "xz_tri_beta":
        print(type_of_points)
        skip = random_seed
        x0=1 #start of the laser
        xspeed=1.5 #speed of the laser in mm/t_max
        xsize=2.8 #total length of domain
        x0=x0/xsize #normed between one and 0
        xspeed=xspeed/xsize #normed to the size of the domain
        datat = np.full((samples, 1), np.nan)
        dataX = np.full((samples, 1), np.nan)
        dataY = np.full((samples, 1), np.nan)
        dataZ = np.full((samples, 1), np.nan)
        dataP = np.full((samples, dim-4), np.nan)

        dataZ[:, 0] = np.random.beta(5, 1, samples)
        for j in range(samples): #majority of coll points sampled around laser
            if j % 50000 ==0:
                print(j/samples *100, "% done")
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim-2, seed)
            datat[j, :]=rnd[0]
            dataX[j,:] = np.random.triangular(0,x0+datat[j, :]*xspeed, 1)
            dataY[j,:]=rnd[1]
            if (dim > 4):
                dataP[j, :] = rnd[2:]
        if dim > 4:
            data=np.concatenate((datat, dataX, dataY, dataZ, dataP), axis=1)
        else:
            data=np.concatenate((datat, dataX, dataY, dataZ), axis=1)
        return torch.from_numpy(data).type(torch.FloatTensor)

    #Still old formatting
    elif type_of_points == "moving_center":
        print(type_of_points)
        skip = random_seed
        '''pts=1 #how many centered points per step
        step=1 #how many time steps in center'''
        # prob=0.25 #factor on how large the randomness of the center points is: 0= 100% centered, 1=completely random
        ring_thick = 0.3
        s_ring_rad = 0.0
        exponent = 1
        n=int(samples/10) #number of total center-bias points

        x0=1 #start of the laser
        xspeed=1.5 #speed of the laser in mm/t_max
        xsize=2.8 #total length of domain
        x0=x0/xsize #normed between one and 0
        xspeed=xspeed/xsize #normed to the size of the domain

        #CHANGE IF YOU CHANGE THE BUFFER Time (minimum extrema of t)!
        t0=0.1 #buffer time

        tlaser=1 #time the laser is actually on
        sum=t0+tlaser
        t0=t0/sum #normed for scaling
        tlaser=tlaser/sum #normed for scaling

        datat = np.full((samples, 1), np.nan)
        dataX = np.full((samples, 1), np.nan)
        dataY = np.full((samples, 1), np.nan)
        dataZ = np.full((samples, 1), np.nan)
        dataP = np.full((samples, dim-4), np.nan)
        i=0
        p=1

        dataY[:, 0] = np.random.laplace(0.5, 1 / 12., samples)
        dataZ[:, 0] = np.random.beta(5, 1, samples)
        for j in range(samples-n-1): #majority of coll points sampled around laser
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim-3, seed)
            datat[j, :]=rnd[0]**exponent
            dataX[j,:] = np.random.triangular(0,x0+datat[j, :]*xspeed, 1)
            if (dim > 4):
                dataP[j, :] = rnd[1:]
        for j in range(samples - n-1, samples): #Center-bias: some points directly in laser center
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim, seed)
            datat[j, :] = rnd[0]**exponent
            #dataX[j, :] = x0 + (datat[j, :] - t0) / tlaser * xspeed + (rnd[1]-0.5) * prob
            #dataY[j, :] = 0.5 + (rnd[2]-0.5) * prob
            #dataZ[j, :] = 1 - rnd[3] * prob/2
            radi =  s_ring_rad + rnd[1] * ring_thick
            theta = rnd[2] * np.pi / 2
            phi  = rnd[3] * 2 * np.pi
            if np.cos(phi) < 0:
                dataX[j, :] = x0  + (datat[j, :] - t0) / tlaser * xspeed + radi * np.cos (phi) * np.sin(theta)*3
            else:
                dataX[j, :] = x0  + (datat[j, :] - t0) / tlaser * xspeed + radi * np.cos (phi) * np.sin(theta)
            dataY[j, :] = 0.5 + radi * np.sin (phi) * np.sin(theta)
            dataZ[j, :] = 1- radi * np.cos(theta)
            if(dim>4):
                dataP[j,:]= rnd[4:]
            '''if p==pts:
                i = i + 1
                p=1
            else:
                p=p+1'''
        if dim > 4:
            data=np.concatenate((datat, dataX, dataY, dataZ, dataP), axis=1)
        else:
            data=np.concatenate((datat, dataX, dataY, dataZ), axis=1)

        return torch.from_numpy(data).type(torch.FloatTensor)
    elif type_of_points == "initial_center":
        print(type_of_points)
        skip = random_seed
        # prob=0.25 #factor on how large the randomness of the center points is: 0= 100% centered, 1=completely random
        ring_thick = 0.3
        s_ring_rad = 0.0
        n=int(samples/10) #number of total center-bias points

        x0=1 #start of the laser
        xspeed=1.5 #speed of the laser in mm/t_max
        xsize=2.8 #total length of domain
        x0=x0/xsize #normed between one and 0
        xspeed=xspeed/xsize #normed to the size of the domain

        #CHANGE IF YOU CHANGE THE BUFFER Time (minimum extrema of t)!
        t0=0.1 #buffer time

        tlaser=1 #time the laser is actually on
        sum=t0+tlaser
        t0=t0/sum #normed for scaling
        tlaser=tlaser/sum #normed for scaling

        datat = np.full((samples, 1), 0.1)
        dataX = np.full((samples, 1), np.nan)
        dataY = np.full((samples, 1), np.nan)
        dataZ = np.full((samples, 1), np.nan)
        dataP = np.full((samples, dim-4), np.nan)
        i=0
        p=1

        dataY[:, 0] = np.random.laplace(0.5, 1 / 12., samples)
        dataZ[:, 0] = np.random.beta(5, 1, samples)
        for j in range(samples-n-1): #majority of coll points sampled around laser
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim-3, seed)
            dataX[j,:] = np.random.triangular(0,x0+datat[j, :]*xspeed, 1)
            if (dim > 4):
                dataP[j, :] = rnd[1:]
        for j in range(samples - n-1, samples): #Center-bias: some points directly in laser center
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim, seed)
            #dataX[j, :] = x0 + (datat[j, :] - t0) / tlaser * xspeed + (rnd[1]-0.5) * prob
            #dataY[j, :] = 0.5 + (rnd[2]-0.5) * prob
            #dataZ[j, :] = 1 - rnd[3] * prob/2
            radi =  s_ring_rad + rnd[1] * ring_thick
            theta = rnd[2] * np.pi / 2
            phi  = rnd[3] * 2 * np.pi
            dataX[j, :] = x0  + radi * np.cos (phi) * np.sin(theta)
            dataY[j, :] = 0.5 + radi * np.sin (phi) * np.sin(theta)
            dataZ[j, :] = 1- radi * np.cos(theta)
            if(dim>4):
                dataP[j,:]= rnd[4:]
        if dim > 4:
            data=np.concatenate((datat, dataX, dataY, dataZ, dataP), axis=1)
        else:
            data=np.concatenate((datat, dataX, dataY, dataZ), axis=1)
        return torch.from_numpy(data).type(torch.FloatTensor)