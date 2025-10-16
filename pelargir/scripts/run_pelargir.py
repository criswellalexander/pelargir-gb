import sys, os, shutil, warnings
import argparse
import matplotlib.pyplot as plt

import numpy as np
import cupy as xp
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator
from matplotlib.collections import LineCollection
import matplotlib.cm
from matplotlib import patches
# import jax.numpy as jnp
# import jax; jax.config.update("jax_enable_x64", True)
from corner import corner, overplot_lines
import legwork as lw
import astropy.units as u
from tqdm import tqdm
from math import factorial
plt.style.use('default')


from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import TransformContainer
from eryn.moves import GaussianMove, StretchMove, CombineMove, DistributionGenerate, MTDistGenMove, Move
from eryn.utils.utility import groups_from_inds
from multiprocessing import Pool

def execute_local_imports():
    from ..models import PopModel
    from ..inference import GalacticBinaryPrior, PopulationHyperPrior
    from ..utils import get_amp_freq, lisa_noise_psd
    from .. import distributions as st

    return

def execute_gpu_imports(mandatory=False):
    import numpy as np
    import cupy as xp
    
    try:
        if xp.cuda.is_available():
            continue
        else:
            if not mandatory:
                warnings.warn("GPU requested but unavailable, reverting to CPU.")
            else:
    
    execute_local_imports()
    



...


if __name__ == '__main__':

    # Create parser
    parser = argparse.ArgumentParser(prog='pelargir', usage='%(prog)s [options] rundir',
                                     description='Run Pelargir global population inference')

    # Add arguments
    parser.add_argument('rundir', metavar='rundir', type=str, help='The path to the run directory')

    ## LATER -- UPDATE TO INFO NEEDED BY PELARGIR
    parser.add_argument('--cpu', action='store_true', help="Disable GPU functionality and run on CPU.")
    
    
    

    parser.add_argument('--nofit', action='store_true', help="Disable spectral fit reconstruction plots.")

    parser.add_argument('--cornersplit', type=str, default=None, help="How to split the corner plots. Default None (one corner plot). Can be 'type' or 'submodel'.")

    parser.add_argument('--cornersmooth', type=float, default=0.75, help="Level of 2D contour smoothing for the corner plots. Default 0.75.")

    parser.add_argument('--plotdatadir', type=str, default=None, help="/path/to/plot_data.pickle; where to save the plot data as a pickle file. Defaults to [out_dir]/plot_data.pickle.")

    parser.add_argument('--cornerfmt', type=str, default=None, help="MPL rcParams dict for formatting the corner plots.")
    parser.add_argument('--cornermaxticks', type=int, default=3, help="Maximum number of ticks for the corner plots.")

    # execute parser
    args = parser.parse_args()
    
    
    
    ## set numpy seed; this is required for reproduceable results with Eryn
    np.random.seed(args.seed)
    
    
    if not args.cpu:
        ## set environment variables
        os.environ['PELARGIR_GPU'] = '1'
        os.environ['SCIPY_ARRAY_API'] = '1'
        os.environ['PELARGIR_ERYN'] = '1'
        
        ## do gpu imports
        execute_gpu_imports()
        
        
