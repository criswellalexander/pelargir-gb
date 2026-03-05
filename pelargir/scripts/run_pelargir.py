import os
import shutil
import ctypes
import ctypes.util
import functools
from pathlib import Path

## fixes an issue on ACCRE
def cuda_lib_hook(lib_path,lib_names=["libnvrtc.so.12","libcusolver.so.11"]):
    
    _real_CDLL_new = ctypes.CDLL.__new__
    _real_find_library = ctypes.util.find_library
    
    LIB_MAP = {}
    for lib_name in lib_names:
        lib_path = Path(str(lib_path)+'/'+lib_name)
        LIB_MAP[lib_name] = str(lib_path)
    
    def _remap(name):
        return LIB_MAP.get(name, name)
    @functools.wraps(_real_CDLL_new)
    def _CDLL_new(cls, name, *args, **kwargs):
        obj = _real_CDLL_new(cls)
        obj.__init__(_remap(name), *args, **kwargs)
        return obj
    @functools.wraps(_real_find_library)
    def _find_library(name):
        print(f"Finding library {name}")
        return _real_find_library(_remap(name))
    
    ctypes.CDLL.__new__ = staticmethod(_CDLL_new)  # ty: ignore[invalid-assignment]
    ctypes.util.find_library = _find_library
    
    return

## handles ACCRE's terrible CUDA setup; most folks shouldn't need this
if 'PELARGIR_CUDA_PATH' in os.environ.keys():
    print("Performing CUDA libnvrtc hook...")
    cuda_lib_hook(os.environ['PELARGIR_CUDA_PATH'])



import numpy as np
import cupy as xp
import matplotlib.pyplot as plt
# from matplotlib.ticker import AutoLocator
# from matplotlib.pyplot import cycler
# from matplotlib.colors import LinearSegmentedColormap, ListedColormap
# from matplotlib.collections import LineCollection
# import matplotlib.cm
# from matplotlib import patches
# import jax.numpy as jnp
# import jax; jax.config.update("jax_enable_x64", True)
from corner import corner
# import legwork as lw
# import astropy.units as u
# from tqdm import tqdm
# from math import factorial
# import scipy.stats as scst
# import scipy.special as sc
import warnings
import pickle

## set environment variables
import sys
import argparse

## Eryn imports
from eryn.ensemble import EnsembleSampler
from eryn.state import State, BranchSupplemental
from eryn.backends import SupplementalBackend
from eryn.prior import ProbDistContainer
# from eryn.utils import TransformContainer
from eryn.moves import GaussianMove, StretchMove, CombineMove, DistributionGenerate, MTDistGenMove, Move


def execute_gpu_imports(mandatory=False):
    
    
    return

def simulate_dataset(rng,pop_theta=None,N=int(1e7),figdir='.'):
    
    if pop_theta is None:
        print("Simulating galaxy with default parameters...")
        pop_theta = {'m_mu': xp.array([0.6]), ## mean of mass dist in solar masses
                     'm_sigma': xp.array([0.15]), ## std. dev. of mass dist in solar masses
                     'rh_disk': xp.array([2.9]), ## radial scale height of the MW in kpc
                     'r_bulge': xp.array([0.25]), ## Gaussian bulge characteristic radius in kpc
                     'q_bd': xp.array([0.33]), ## ratio of bulge mass / disk mass
                     'a_alpha': xp.array([0.5])} ## slope of orbital separation distribution
    if xp is np:
        truths = np.array([pop_theta[key] for key in pop_theta.keys()]).flatten()
    else:
        truths = xp.asnumpy([pop_theta[key].get() for key in pop_theta.keys()]).flatten()
    
    ## initialize and condition the prior
    pop_prior = GalacticBinaryPrior(rng)
    pop_prior.condition(pop_theta)
    
    ## sample N binaries
    samps = pop_prior.sample_conditional(N)
    
    ## plot the distributions and save
    plt.close()
    Nplot = 100000
    stride = int(N/Nplot)
    fig = corner(to_numpy(samps.squeeze()[:,::stride]).T,labels=list(pop_prior.conditional_dict.keys()))
    plotting.savefig_to_path('initial_population_distributions',saveto=figdir)
    plt.close()
    
    return samps, truths


if __name__ == '__main__':

    # Create parser
    parser = argparse.ArgumentParser(prog='pelargir', usage='%(prog)s [options] rundir',
                                     description='Run Pelargir global population inference')

    # Add arguments
    parser.add_argument('rundir', metavar='rundir', type=str, help='The path to the run directory')

    parser.add_argument('--cpu', action='store_true', help="Disable GPU functionality and run on CPU.")
    parser.add_argument('--gpu_mandatory', action='store_true', help="Enforce GPU functionality.")
    
    ## ACCRE CUDA fix
    parser.add_argument('--fixlib', action='store_true', help="Fix errors due to cupy not finding libnvrtc.")
    parser.add_argument('--cudalib', type=str, help="Path to CUDA libraries. Only used if --fixlib is specified. Default is ACCRE CUDA 12.9 path.",
                        default="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.9.1/lib64/libnvrtc.so.12")
    
    parser.add_argument('--pelargirpath', type=str, help='Directory containing pelargir',
                        default='/home/awc/Documents/LISA/projects/lisa_population_inference/pelargir-gb/pelargir/')
    
    parser.add_argument('--Nsim', type=int, help='Number of binaries in simulated Galaxy.', default=int(1e7))
    
    parser.add_argument('--seed', type=int, help='RNG seed used for analysis', default=170817)
    parser.add_argument('--simseed', type=int, help='RNG seed used for creating the simulated dataset', default=150914)
    
    parser.add_argument('--fmin', type=float, help='Minimum frequency', default=1e-4)
    parser.add_argument('--fmax', type=float, help='Maximum frequency', default=5e-3)
    parser.add_argument('--fbin', type=float, help='Frequency bin width', default=2e-5)
    
    parser.add_argument('--block_after', type=int, help='Number of frequency bins to treat individually before switching to blocked array sorting.',default=4)
    
    ## abstracted likelihood assumptions
    parser.add_argument('--logsigma', type=float, help='Standard deviation of the foreground log amplitude, in dex.', default=0.1)
    parser.add_argument('--noscatter', action='store_true',help='Turn off likelihood scatter entirely for the simulated resolved GB parameters.')
    parser.add_argument('--nodynscatter', action='store_true',help='Turn off dynamic likelihood scatter for the resolved GB likelihood.')
    
    ## Eryn/sampling arguments
    parser.add_argument('--vectorize', action='store_true',help='If active, parallelizes likelihood evaluations across walkers and temperatures.')
    parser.add_argument('--Ntemps', type=int, help='Number of temperatures to use in parallel tempering', default=1)
    parser.add_argument('--Tmax', type=float, help='Maximum temperatures to use in parallel tempering', default=xp.inf)
    parser.add_argument('--Nreal', type=int, help='Number of Poisson realizations per likelihood evaluation', default=2)
    parser.add_argument('--Nwalkers', type=int, help='Number of walkers to use within Eryn', default=1)
    parser.add_argument('--moveset', type=str, help='Which of the pre-built movesets to use. \
                                                     Options include: stretch, stretch+prior, gauss, gaussmix, gaussmix+prior.\
                                                     Default is gaussmix+prior.', default='gaussmix+prior')
    parser.add_argument('--move_length', type=float, help='Standard deviation to use as the autocovariance of the Gaussian moves', default=0.001)
    parser.add_argument('--Nsteps', type=int, help='Number of steps to run the sampler.', default=1)
    parser.add_argument('--plot_every', type=int, help='Step intervals at which progress plots will be made. \
                        If None, plots are only made at the end.', default=100)
    parser.add_argument('--thin_by', type=int, help='How much to thin the chain for the final set of plots.', default=1)
    parser.add_argument('--discard', type=int, help='How many steps of burn-in to discard from the chain for the final set of plots..', default=0)
    parser.add_argument('--overwrite_dir', action='store_true',help='If active, allows for overwriting of existing directories when specifying rundir.')
    
    ## plotting arguments
    parser.add_argument('--specymin', type=float, help='Spectra plots ymin. If None, plots will autoscale.', default=None)
    parser.add_argument('--specymax', type=float, help='Spectra plots ymax. If None, plots will autoscale.', default=None)

    ## sim-only and resume
    parser.add_argument('--sim_only', action='store_true', help='Only run the simulation portion of the script.')
    parser.add_argument('--resume', action='store_true', help='Indicates to resume a run from a previous state. Warning: you must specify identical settings.')
    
    # execute parser
    args = parser.parse_args()
    
    if not os.path.exists(args.rundir):
        os.mkdir(args.rundir)
    elif args.overwrite_dir:
        shutil.rmtree(args.rundir)
        os.mkdir(args.rundir)
    else:
        raise RuntimeError("Run directory already exists. If desired, set --overwrite_dir to force deletion and creation.")
    
    os.mkdir(args.rundir+'/run/')
    
    ## save settings
    with open(args.rundir+'commandline_args.txt', 'w') as f:
        f.write(' '.join(sys.argv[1:]))
    
    ## set numpy seed; this is required for reproduceable results with Eryn
    np.random.seed(args.seed)
    
    sys.path.insert(1, args.pelargirpath)
    if not args.cpu:
        ## do gpu imports
        try:
            if xp.cuda.is_available():
                gpu = True
                os.environ['PELARGIR_GPU'] = '1'
                os.environ['SCIPY_ARRAY_API'] = '1'
                os.environ['PELARGIR_ERYN'] = '1'
                print('GPU enabled.')
            else:
                gpu = False
                if not args.gpu_mandatory:
                    warnings.warn("GPU requested but unavailable, reverting to CPU.")
                    xp = np
                    
        except:
            warnings.warn("An error occurred while initializing GPU functionality, reverting to CPU.")
            xp = np
            gpu = False
        
        if args.gpu_mandatory and not gpu:
            raise RuntimeError("GPU was marked as mandatory but was not successfully loaded.")
    else:
        gpu = False
        xp = np
        
    ## now do imports
    from models import PopModel
    from inference import GalacticBinaryPrior, PopulationHyperPrior
    from utils import get_amp_freq, lisa_noise_psd, set_style, to_numpy
    from plotting import plot_corners, plot_Nres_hist, plot_spectra, plot_spectra_chains, plot_model_chains, plot_model_loglikes, plot_astro_dists, plot_data_spectrum, plot_sanity_check
    import plotting
    from moves import make_PriorMove, PoissonMove
    import distributions as st
        
    set_style()
    
    ## set frequency bins. Lowest bin has edge effects and is dropped
    fbins = xp.arange(args.fmin-args.fbin/2,args.fmax+args.fbin/2,args.fbin)
    
    print("Frequency resolution is {:0.2e} Hz; using {} frequency bins.".format(args.fbin,fbins.size))

    if not args.resume:
        ## initialize sim rng
        sim_rng = xp.random.default_rng(args.simseed)
        
        ## simulate the dataset
        ## TODO -- pass pop_theta via argparse
        sim_gbs, truths = simulate_dataset(sim_rng,N=args.Nsim,figdir=args.rundir,pop_theta=None)
        sim_amps, sim_fgws = get_amp_freq(sim_gbs)
        
        ## initialize the simulation hyperprior object
        sim_hyperprior = PopulationHyperPrior(sim_rng)
        
        ## initialize a GBPrior object to use for plotting later
        plot_prior_obj = GalacticBinaryPrior(sim_rng)
        
        ## initialize the model to threshold the simulation
        sim_popmodel = PopModel(args.Nsim,sim_rng,hyperprior=sim_hyperprior,Nsamp=1,Nreal=1,fbins=fbins)
        
        print("Preprocessing simulated data...")
        ## get the data 
        data_N_res, data_coarse_fg, data_res_idx = sim_popmodel.thresher.serial_array_sort(xp.array([sim_fgws,sim_amps]),
                                                                             sim_popmodel.fbins,
                                                                             snr_thresh=sim_popmodel.thresh_val,
                                                                             get_indices=True)
        data_fg = sim_popmodel.reweight_foreground(data_coarse_fg)[1:]
        
        ## remove extra shape-1 dims
        sim_gbs = sim_gbs.squeeze()
        resgb_thetas = sim_gbs[:,data_res_idx].T
        
        ## setup w.r.t. the data
        datadict = {'fs':fbins[1:],
                    'fg':data_fg,
                    'fg_sigma':xp.array(args.logsigma),
                    'Nres':data_N_res,
                    'noise':lisa_noise_psd(fbins[1:]),
                    'gb_thetas':resgb_thetas,
                    'gb_thetas_all':to_numpy(sim_gbs.T),
                    'truevals':truths}

        ## plot initial spectrum
        plot_data_spectrum(datadict,show=False,save=True,saveto=args.rundir,savename='simulated_spectrum')
    
        ## saving data; cast to numpy first so it can be unpickled sans GPU/CUDA
        print("Saving simulated spectrum to {}".format(args.rundir+'/data/'))
        os.mkdir(args.rundir+'/data/')
        datadict_save = {key:to_numpy(datadict[key]) for key in datadict}
        with open(args.rundir+'/data/dataset.pickle','wb') as f:
            pickle.dump(datadict_save,f)
        if args.sim_only:
            print("Simulation complete. Option --sim_only is set; exiting.")
            exit()
    else:
        ## load previous simulation
        with open(args.rundir+'/data/dataset.pickle','rb') as f:
            datadict = pickle.load(f)
        ## place relevant pieces of the dataset on the GPU, if applicable
        for key in datadict.keys():
            if key != 'gb_thetas_all' and key != 'truevals':
                datadict[key] = xp.asarray(datadict[key])

    
    print("Initializing population inference model...")
    ## initialize a new rng for the analysis
    rng = xp.random.default_rng(args.seed)
    
    ## build the hyperprior for Eryn
    translation_dict = {0:'m_mu',
                        1:'m_sigma',
                        2:'rh_disk',
                        3:'r_bulge',
                        4:'q_bd',
                        5:'a_alpha'}
    eryn_hyperprior_dict = {0:st.uniform(rng,loc=0.59,scale=0.02,cast=True),
                            1:st.uniform(rng,loc=0.14,scale=0.02,cast=True),
                            2:st.uniform(rng,loc=2.8,scale=0.2,cast=True),
                            3:st.uniform(rng,loc=0.24,scale=0.02,cast=True),
                            4:st.uniform(rng,loc=0.32,scale=0.02,cast=True),
                            5:st.uniform(rng,loc=0.48,scale=0.04,cast=True)}
    # eryn_hyperprior_dict = {0:st.uniform(rng,loc=0.2,scale=0.9,cast=True),
    #                         1:st.invgamma(rng,7,cast=True),
    #                         2:st.uniform(rng,loc=1,scale=9,cast=True),
    #                         3:st.uniform(rng,loc=0.05,scale=1.95,cast=True),
    #                         4:st.uniform(rng,loc=0.01,scale=0.98,cast=True),
    #                         5:st.uniform(rng,loc=-0.5,scale=2.0,cast=True)}
    eryn_trans_dict = {translation_dict[key]:eryn_hyperprior_dict[key] for key in eryn_hyperprior_dict.keys()}
    
    eryn_prior = ProbDistContainer(eryn_hyperprior_dict)
    
    ## fancy names
    fancynames = [r'$\mu_m$',
                  r'$\sigma_m$',
                  r'$r_{\rm disk}$',
                  r'$r_{\rm bulge}$',
                  r'$q_{\rm BD}$',
                  r'$\alpha_a$']
    
    ## set up resolved binary scatter
    if args.noscatter:
        scatter = False
        dynamic_scatter = False
    elif args.nodynscatter:
        scatter=True
        dynamic_scatter = False
    else:
        scatter=True
        dynamic_scatter = True
    
    ## set up inference model
    eryn_popmodel = PopModel(args.Nsim,rng,hyperprior=eryn_trans_dict,fbins=fbins,Nreal=args.Nreal,block_after=args.block_after,
                             res_scatter=scatter,res_dynamic_scatter=dynamic_scatter)
    eryn_popmodel.construct_likelihood(datadict,hp_beta=0.05,hp_alpha=5)
    log_like_fn = eryn_popmodel.ln_prob

    ## check consistency
    import pdb; pdb.set_trace()
    test_f1, test_spec1, test_N1 = eryn_popmodel.run_model(pop_theta=xp.asarray(datadict['truevals']))
    test_N, test_spec_coarse = eryn_popmodel.thresher.serial_array_sort(xp.array([sim_fgws,sim_amps]),
                                                                             eryn_popmodel.fbins,
                                                                             snr_thresh=eryn_popmodel.thresh_val,
                                                                             get_indices=False)
    test_spec = eryn_popmodel.reweight_foreground(test_spec_coarse)[1:]
    plot_sanity_check(datadict,test_spec,test_spec1,show=False,save=True,saveto=args.rundir,savename='consistency_check')
    
    ## setup Eryn
    print("Setting up Eryn sampling...")
    ndim = len(eryn_popmodel.hyperprior.hyperprior_dict)
    nwalkers = args.Nwalkers
    ntemps = args.Ntemps
    Nf = len(fbins[1:])
    
    # parallel tempering kwargs dictionary
    tempering_kwargs=dict(ntemps=ntemps,
                          Tmax=args.Tmax)
    
    
    ## initialize some moves
    ## MH with prior draws as the proposal function
    PriorMove = make_PriorMove(eryn_prior)
    ## Gaussian proposals
    move_cov = np.diag(args.move_length*np.ones(eryn_popmodel.Npar))
    GibbsGaussianMove = GaussianMove(cov_all={'model_0':move_cov},
                                     mode='random'
                                     )
    JointGaussianMove = GaussianMove(cov_all={'model_0':move_cov},
                                     mode='vector'
                                     )
    
    ## set moves
    movesets = {'stretch':StretchMove(),
                'stretch+prior':[(StretchMove(),0.7),(PriorMove,0.3)],
                'gauss':JointGaussianMove,
                'gaussmix':[(JointGaussianMove,0.3),(GibbsGaussianMove,0.7)],
                'gaussmix+prior':[(JointGaussianMove,0.25),(GibbsGaussianMove,0.5),(PriorMove,0.25)],
                'gmpp':[(JointGaussianMove,0.3),(GibbsGaussianMove,0.3),(PriorMove,0.1),(PoissonMove(),0.3)]}
    
    if args.moveset not in movesets.keys():
        raise RuntimeError("Requested moveset is not implemented (or misspelled): {}\n \
                            Implemened movesets are {}".format(args.moveset,list(movesets.keys())))
    moves = movesets[args.moveset]
    
    ## initialize the Branch Supplemental to track spectra, Nres
    branch_supp = BranchSupplemental({"spectra": np.zeros((ntemps,nwalkers,1,Nf,args.Nreal,1)),
                                      "Nres": np.zeros((ntemps,nwalkers,1,1,args.Nreal,1))},
                                     base_shape=(ntemps, nwalkers,1),
                                     copy=True)
    # if args.vectorize:
    #     supp_dims = {'spectra':(ntemps,nwalkers,Nf,args.Nreal,1),
    #              'Nres':(ntemps,nwalkers,1,args.Nreal,1)}
    # else:
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
                               vectorize=args.vectorize,
                               provide_supplemental=True,
                               dynamic_branch_supplemental=True,
                               backend=supp_backend
                              )
    
    print("Beginning sampling...")
    if args.plot_every is not None:
        figpath = args.rundir+'/run/plots/'
        chainpath = args.rundir+'/run/chains/'
        os.mkdir(figpath)
        os.mkdir(chainpath)
        steps_taken = 0
        for ri in range(args.Nsteps//args.plot_every + 1):
            steps_left = args.Nsteps - steps_taken
            if steps_left <= 0:
                break
            elif steps_left < args.plot_every:
                steps_i = steps_left
            else:
                steps_i = args.plot_every
            print("Running steps {}-{}".format(steps_taken+1,steps_taken+steps_i))
            
            ## run the sampler
            state = ensemble.run_mcmc(state, steps_i, burn=0, progress=True, thin_by=1)
            
            steps_taken += steps_i
            
            ## make and save plots
            plot_model_chains(ensemble,names=fancynames,temp_index=0,
                              show=False,save=True,saveto=figpath,savename='chains_{}'.format(steps_taken))
            plot_model_loglikes(ensemble,names=eryn_popmodel.hpar_names,temp_index=0,
                                show=False,save=True,saveto=figpath,savename='loglikes_{}'.format(steps_taken))
            plot_Nres_hist(ensemble,datadict,bins=30,temp_index=0,
                           show=False,save=True,saveto=figpath,savename='Nres_hist_{}'.format(steps_taken))
            plot_spectra(ensemble,datadict,chain_kwargs=dict(temp_index=0),iteration=-1,ylim=(args.specymin,args.specymax),xlim=(args.fmin,args.fmax),
                         show=False,save=True,saveto=figpath,savename='spectra_{}'.format(steps_taken))
            plot_spectra_chains(ensemble,datadict,show=False,save=True,
                                 saveto=figpath,savename='spectral_chains_{}'.format(steps_taken),
                                 ylim=(args.specymin,args.specymax),xlim=(args.fmin,args.fmax),temp_index=0)
            plot_astro_dists(ensemble,datadict,plot_prior_obj,model_name='model_0',
                                    show=False,save=True,saveto=figpath,
                                    savename='astro_distributions_{}'.format(steps_taken),temp_index=0)
            samples = ensemble.get_chain(discard=0,temp_index=0,thin=1)['model_0'].reshape(-1,ndim)
            plot_corners(samples,parameters=fancynames,
                         Nbins=20,figsize=(10,10),truths=truths,density=False,plot_datapoints=True,
                                      show=False,save=True,saveto=figpath,savename='corners_{}'.format(steps_taken))
            set_style()
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
    plot_model_chains(ensemble,names=fancynames,temp_index=0,thin=args.thin_by,discard=args.discard,
                      show=False,save=True,saveto=args.rundir)
    plot_model_loglikes(ensemble,names=eryn_popmodel.hpar_names,temp_index=0,thin=args.thin_by,discard=args.discard,
                        show=False,save=True,saveto=args.rundir)
    plot_Nres_hist(ensemble,datadict,bins=30,temp_index=0,thin=args.thin_by,discard=args.discard,
                   show=False,save=True,saveto=args.rundir)
    plot_spectra(ensemble,datadict,chain_kwargs=dict(temp_index=0),iteration=-1,ylim=(args.specymin,args.specymax),xlim=(args.fmin,args.fmax),
                 show=False,save=True,saveto=args.rundir)
    plot_spectra_chains(ensemble,datadict,show=False,save=True,temp_index=0,thin=args.thin_by,discard=args.discard,
                         saveto=args.rundir,savename='spectral_chains',
                         ylim=(args.specymin,args.specymax),xlim=(args.fmin,args.fmax))
    plot_astro_dists(ensemble,datadict,plot_prior_obj,temp_index=0,thin=args.thin_by,discard=args.discard,
                            show=False,save=True,saveto=args.rundir)
    samples = ensemble.get_chain(discard=args.discard,temp_index=0,thin=args.thin_by)['model_0'].reshape(-1,ndim)
    plot_corners(samples,parameters=fancynames,
                 Nbins=20,figsize=(10,10),truths=truths,density=False,plot_datapoints=True,
                 show=False,save=True,saveto=args.rundir)
    ## save chains
    np.save(args.rundir+'/data/chain_final', 
            ensemble.get_chain()['model_0'])
    np.save(args.rundir+'/data/spec_chain_final',
            ensemble.get_chain_supplemental()['model_0']['spectra'])
    np.save(args.rundir+'/data/Nres_chain_final', 
            ensemble.get_chain_supplemental()['model_0']['Nres'])
    print("Final plots and chains saved.")
    
    print("Done!")