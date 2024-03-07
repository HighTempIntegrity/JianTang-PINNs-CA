import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms

import scipy.io
from scipy import special as spec
import sys
import json
import time
import pprint
import os

import math
import random

from pyDOE import lhs
import numpy as np
import sobol_seq

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mc
import colorsys
from matplotlib import rc
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd

from os import listdir
from os.path import isfile, join

# from vtk import vtkXMLStructuredGridReader
# import vtk
# from vtk.util.numpy_support import vtk_to_numpy
pi = math.pi


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Change to switch between parameterized or simple version
from Paper_Equation_param import EquationClass
# Paper_Equation_tempDep
from FitClass import *
from ModClass import *
from DatasetClass import *

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{euscript}')
SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the tick labels
