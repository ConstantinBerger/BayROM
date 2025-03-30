import sys
import numpy as np
import seaborn as sns
import scipy
import nifty8 as ift
import pickle
import matplotlib
import torch
import h5py
import argparse
import os
import pickle
import pylops
import time

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import RectBivariateSpline, interp1d, RegularGridInterpolator
from scipy import signal
from scipy.stats import multivariate_normal
from scipy.io import loadmat
from skimage.metrics import structural_similarity
from microfilm import microplot
from tqdm import tqdm, trange
from datetime import datetime

from Operator import Operator, PSFResponseOperator
from Field import Field
from Reconstruction import Optimizer
from helpers import reconstruct_low_res, plot_image, overlay_images

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
#
# preamble = "\n".join([r"\usepackage{amsmath,amssymb,amsfonts}"
#                       r"\usepackage[utf8]{inputenc}", r"\usepackage{fourier}", r"\usepackage[T1]{fontenc}",
#     r"\usepackage[detect-all]{siunitx}", ])
#
# pgf_with_latex = {"pgf.texsystem": "pdflatex", "text.usetex": True, "font.family": "serif", "font.serif": [],
#     "font.sans-serif": [], "font.monospace": [], "pgf.preamble": preamble, "text.latex.preamble": preamble, }
# matplotlib.rcParams.update(pgf_with_latex)

ift.random.push_sseq_from_seed(27)
