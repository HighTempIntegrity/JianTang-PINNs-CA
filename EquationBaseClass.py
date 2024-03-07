import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.colors as mc
import colorsys
from scipy.special import legendre
import sobol_seq
import itertools


class EquationBaseClass:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu") #

    def convert(self, vector, extrema_values):
        vector = np.array(vector)
        max_val = np.max(np.array(extrema_values), axis=1)
        min_val = np.min(np.array(extrema_values), axis=1)
        vector = vector * (max_val - min_val) + min_val
        return torch.from_numpy(vector).type(torch.FloatTensor)

    def lighten_color(self, color, amount=0.5):
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
