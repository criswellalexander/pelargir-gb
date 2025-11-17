import numpy as np
import cupy as xp
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator
from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
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
import scipy.stats as scst
import scipy.special as sc
import warnings

## set environment variables
import os, sys
import argparse

## Eryn imports
from eryn.ensemble import EnsembleSampler
from eryn.state import State, BranchSupplemental
from eryn.backends import SupplementalBackend
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import TransformContainer
from eryn.moves import GaussianMove, StretchMove, CombineMove, DistributionGenerate, MTDistGenMove, Move
from eryn.utils.utility import groups_from_inds
from multiprocessing import Pool


def execute_local_imports():

    from models import PopModel
    from inference import GalacticBinaryPrior, PopulationHyperPrior
    from utils import get_amp_freq, lisa_noise_psd, set_style, to_numpy
    from plotting import plot_spectra, plot_spectra_flexible, plot_corners, plot_model_chains, plot_model_loglikes, plot_Nres_hist
    import plotting
    from moves import make_PriorMove, PoissonMove
    import distributions as st
    
    return

def execute_gpu_imports(mandatory=False):
    import numpy as np
    import cupy as xp
    
    try:
        if xp.cuda.is_available():
            gpu = True
            os.environ['PELARGIR_GPU'] = '1'
            os.environ['SCIPY_ARRAY_API'] = '1'
            os.environ['PELARGIR_ERYN'] = '1'
            print('GPU enabled.')
        else:
            gpu = False
            if not mandatory:
                warnings.warn("GPU requested but unavailable, reverting to CPU.")
                xp = np
                
    except:
        warnings.warn("An error occurred while initializing GPU functionality, reverting to CPU.")
        xp = np
        gpu = False
    
    if mandatory and not gpu:
        raise RuntimeError("GPU was marked as mandatory but was not successfully loaded.")
    
    execute_local_imports()
    
    return

def simulate_dataset(rng,pop_theta=None,N=int(1e7),figdir='.'):
    
    if pop_theta is None:
        print("Simulating galaxy with default parameters...")
        pop_theta = {'m_mu':xp.array([0.6]),'m_sigma':xp.array([0.15]),
                     'd_gamma_a':xp.array([4]),'d_gamma_b':xp.array([4]),
                     'a_alpha':xp.array([1/2])}
    
    ## initialize and condition the prior
    pop_prior = GalacticBinaryPrior(rng)
    pop_prior.condition(pop_theta)
    
    ## sample N binaries
    samps = pop_prior.sample_conditional((N,))
    
    ## plot the distributions and save
    plt.close()
    fig = corner(samps[:,::200].get().T,labels=list(pop_prior.conditional_dict.keys()))
    plotting.savefig_to_path('initial_population_distributions',saveto=figdir)
    plt.close()
    
    return samps


if __name__ == '__main__':

    # Create parser
    parser = argparse.ArgumentParser(prog='pelargir', usage='%(prog)s [options] rundir',
                                     description='Run Pelargir global population inference')

    # Add arguments
    parser.add_argument('rundir', metavar='rundir', type=str, help='The path to the run directory')

    ## LATER -- UPDATE TO INFO NEEDED BY PELARGIR
    parser.add_argument('--cpu', action='store_true', help="Disable GPU functionality and run on CPU.")
    
    ## will be depreciated later
    parser.add_argument('--pelargirpath', type=str, help='Directory containing pelargir',
                        default='/home/awc/Documents/LISA/projects/lisa_population_inference/pelargir-gb/pelargir/')
    
    parser.add_argument('--Nsim', type=int, help='Number of binaries in simulated Galaxy.', default=int(1e7))
    
    parser.add_argument('--seed', type=int, help='RNG seed used for analysis', default=170817)
    parser.add_argument('--simseed', type=int, help='RNG seed used for creating the simulated dataset', default=150914)
    
    parser.add_argument('--fmin', type=float, help='Minimum frequency', default=1e-4)
    parser.add_argument('--fmax', type=float, help='Maximum frequency', default=5e-3)
    parser.add_argument('--fbin', type=float, help='Frequency bin width', default=2e-5)
    
    parser.add_argument('--logsigma', type=float, help='Standard deviation of the foreground log amplitude, in dex.', default=0.1)
    
    ## Eryn/sampling arguments
    parser.add_argument('--Ntemps', type=int, help='Number of temperatures to use in parallel tempering', default=1)
    parser.add_argument('--Nreal', type=int, help='Number of Poisson realizations per likelihood evaluation', default=2)
    parser.add_argument('--Nwalkers', type=int, help='Number of walkers to use within Eryn', default=1)
    parser.add_argument('--moveset', type=str, help='Which of the pre-built movesets to use. \
                                                     Options include: stretch, stretch+prior, gauss, gaussmix, gaussmix+prior.\
                                                     Default is gaussmix+prior.', default='gaussmix+prior')
    
    parser.add_argument('--Nsteps', type=int, help='Number of steps to run the sampler.', default=1)
    parser.add_argument('--plot_every', type=int, help='Step intervals at which progress plots will be made. \
                        If None, plots are only made at the end.', default=100)
    
    # execute parser
    args = parser.parse_args()
    
    
    
    ## set numpy seed; this is required for reproduceable results with Eryn
    np.random.seed(args.seed)
    
    sys.path.insert(1, args.pelargirpath)
    if not args.cpu:
        ## do gpu imports
        execute_gpu_imports()
    else:
        xp = np
        execute_local_imports()
        
    set_style()
    
    ## set frequency bins
    fbins = xp.arange(args.fmin,args.fmax,args.fbin)
    
    ## initialize sim rng
    sim_rng = xp.random.default_rng(args.simseed)
    
    ## simulate the dataset
    ## TODO -- pass pop_theta via argparse
    sim_gbs = simulate_dataset(sim_rng,N=args.Nsim,figdir=args.rundir+'/plots/',pop_theta=None)
    sim_amps, sim_fgws = get_amp_freq(sim_gbs)
    
    ## initialize the simulation hyperprior object
    sim_hyperprior = PopulationHyperPrior(sim_rng)
    
    ## initialize the model to threshold the simulation
    sim_popmodel = PopModel(args.Nsim,sim_rng,hyperprior=sim_hyperprior,Nsamp=1,Nreal=1,fbins=fbins)
    
    ## get the data 
    data_N_res, data_coarse_fg = sim_popmodel.thresher.serial_array_sort(xp.array([sim_fgws,sim_amps]),
                                                                         sim_popmodel.fbins,
                                                                         snr_thresh=sim_popmodel.thresh_val)
    data_fg = sim_popmodel.reweight_foreground(data_coarse_fg)[1:]
    
    ## setup w.r.t. the data
    datadict = {'fs':fbins[1:],
                'fg':data_fg,
                'fg_sigma':xp.array(args.logsigma),
                'Nres':data_N_res,
                'noise':lisa_noise_psd(fbins[1:])}
    
    
    ## initialize a new rng for the analysis
    rng = xp.random.default_rng(args.seed)
    
    ## build the hyperprior for Eryn
    translation_dict = {0:'m_mu',
                        1:'m_sigma',
                        2:'d_gamma_a',
                        3:'d_gamma_b',
                        4:'a_alpha'}
    eryn_hyperprior_dict = {0:st.uniform(rng,loc=0.2,scale=0.9,cast=True),
                            1:st.invgamma(rng,5,cast=True),
                            2:st.uniform(rng,loc=1,scale=10,cast=True), ## these are pretty arbitrary
                            3:st.uniform(rng,loc=1,scale=10,cast=True), ## these are pretty arbitrary
                            4:st.uniform(rng,loc=-0.5,scale=2,cast=True)}
    eryn_trans_dict = {translation_dict[key]:eryn_hyperprior_dict[key] for key in eryn_hyperprior_dict.keys()}
    
    eryn_prior = ProbDistContainer(eryn_hyperprior_dict)
    
    ## set up inference model
    eryn_popmodel = PopModel(args.Nsim,rng,hyperprior=eryn_trans_dict,fbins=fbins,Nreal=args.Nreal)
    eryn_popmodel.construct_likelihood(datadict,hp_beta=0.05,hp_alpha=5)
    log_like_fn = eryn_popmodel.fg_N_ln_prob
    
    ## setup Eryn
    ndim = len(eryn_popmodel.hyperprior.hyperprior_dict)
    nwalkers = args.Nwalkers
    ntemps = args.Ntemps
    Nf = len(fbins[1:])
    
    # parallel tempering kwargs dictionary
    tempering_kwargs=dict(ntemps=ntemps)
    
    
    ## initialize some moves
    ## MH with prior draws as the proposal function
    PriorMove = make_PriorMove(eryn_prior)
    GibbsGaussianMove = GaussianMove(cov_all={'model_0':np.array([0.1,0.025,1,1,0.1])},
                                     mode='random'
                                     )
    JointGaussianMove = GaussianMove(cov_all={'model_0':np.array([0.1,0.025,1,1,0.1])},
                                     mode='vector'
                                     )
    
    ## set moves
    movesets = {'stretch':StretchMove(),
                'stretch+prior':[(StretchMove(),0.7),(PriorMove,0.3)],
                'gauss':JointGaussianMove,
                'gaussmix':[(JointGaussianMove,0.3),(GibbsGaussianMove,0.7)],
                'gaussmix+prior':[(JointGaussianMove,0.25),(GibbsGaussianMove,0.5),(PriorMove,0.25)]}
    
    if args.moveset not in movesets.keys():
        raise RuntimeError("Requested moveset is not implemented (or misspelled): {}\n \
                            Implemened movesets are {}".format(args.moveset,list(movesets.keys())))
    moves = movesets[args.moveset]
    
    ## initialize the Branch Supplemental to track spectra, Nres
    branch_supp = BranchSupplemental({"spectra": np.zeros((ntemps,nwalkers,1,Nf,args.Nreal,1)),
                                      "Nres": np.zeros((ntemps,nwalkers,1,1,args.Nreal,1))},
                                     base_shape=(ntemps, nwalkers,1),
                                     copy=True)
    supp_dims = {'spectra':(Nf,args.Nreal,1),
                 'Nres':(1,args.Nreal,1)}
    supp_backend = SupplementalBackend(supp_dims)
    
    # starting positions
    # randomize throughout prior
    coords = eryn_prior.rvs(size=(ntemps,nwalkers,))
    
    ## initialize starting state object with supplemenal
    state = State(coords,
                        branch_supplemental={'model_0':branch_supp})
    
    ## initialize ensemble
    ensemble = EnsembleSampler(nwalkers,
                               ndim,
                               log_like_fn,
                               eryn_prior,
                               moves=moves,
                               track_moves=True,
                               tempering_kwargs=tempering_kwargs,
                               provide_supplemental=True,
                               dynamic_branch_supplemental=True,
                               backend=supp_backend
                              )
    
    if args.plot_every is not None:
        figpath = args.rundir+'/run/plots/'
        chainpath = args.rundir+'/run/chains/'
        os.mkdir(figpath)
        os.mkdir(chainpath)
        steps_taken = 0
        for ri in range(args.Nsteps%args.plot_every + 1):
            print("Running steps {}-{}".format(steps_taken+1,steps_taken+args.plot_every))
            steps_left = args.Nsteps - steps_taken
            if steps_left < args.plot_every:
                steps_i = steps_left
            else:
                steps_i = args.plot_every
                
            
            ## run the sampler
            state = ensemble.run_mcmc(state, steps_i, burn=0, progress=True, thin_by=1)
            
            steps_taken += steps_i
            
            ## make and save plots
            plot_model_chains(ensemble,names=eryn_popmodel.hpar_names,temp_index=0,
                              show=False,save=True,saveto=figpath,savename='chains_{}'.format(steps_taken))
            plot_model_loglikes(ensemble,names=eryn_popmodel.hpar_names,temp_index=0,
                                show=False,save=True,saveto=figpath,savename='loglikes_{}'.format(steps_taken))
            plot_Nres_hist(ensemble,datadict,bins=np.linspace(0,3000,30),temp_index=0,
                           show=False,save=True,saveto=figpath,savename='Nres_hist_{}'.format(steps_taken))
            plot_spectra(ensemble,datadict,chain_kwargs=dict(temp_index=0),iteration=-1,ylim=(1e-40,1e-35),
                         show=False,save=True,saveto=figpath,savename='spectra_{}'.format(steps_taken))
            
            ## save chains
            np.save(chainpath+'/chain_{}'.format(steps_taken), 
                    ensemble.get_chain()['model_0'])
            np.save(chainpath+'/spec_chain_{}'.format(steps_taken), 
                    ensemble.get_chain_supplemental()['model_0']['spectra'])
            np.save(chainpath+'/Nres_chain_{}'.format(steps_taken), 
                    ensemble.get_chain_supplemental()['model_0']['Nres'])
            print("Plots and chains saved.")
    else:
        ## run the sampler
        print("Running full analysis for {} steps...".format(args.Nsteps))
        state = ensemble.run_mcmc(state, args.Nsteps, burn=0, progress=True, thin_by=1)
    
    ## make and save plots
    print("Run complete. Making final plots...")
    plot_model_chains(ensemble,names=eryn_popmodel.hpar_names,temp_index=0,
                      show=False,save=True,saveto=args.rundir)
    plot_model_loglikes(ensemble,names=eryn_popmodel.hpar_names,temp_index=0,
                        show=False,save=True,saveto=args.rundir)
    plot_Nres_hist(ensemble,datadict,bins=np.linspace(0,3000,30),temp_index=0,
                   show=False,save=True,saveto=args.rundir)
    plot_spectra(ensemble,datadict,chain_kwargs=dict(temp_index=0),iteration=-1,ylim=(1e-40,1e-35),
                 show=False,save=True,saveto=args.rundir)
    
    ## save chains
    np.save(args.rundir+'/chain_final', 
            ensemble.get_chain()['model_0'])
    np.save(args.rundir+'/spec_chain_final',
            ensemble.get_chain_supplemental()['model_0']['spectra'])
    np.save(args.rundir+'/Nres_chain_final', 
            ensemble.get_chain_supplemental()['model_0']['Nres'])
    print("Final plots and chains saved.")
    
    print("Done!")