"""
Benchmarks the jitted JAX likelihood (jax_likelihood.ln_like) against the cupy likelihood terms
(FG_Likelihood, Nres_Likelihood, Res_Astro_Likelihood.static_ln_conditional_prob) on identical
inputs, and checks parity. Needs the jax backend (cupy reference).

Inputs are synthetic but realistically sized: a noise-scaled foreground on the run_pelargir grid,
Nreal draws around it with 0.1 dex scatter, N_res resolved binaries drawn from the fiducial
population, and N_res-like counts.

Reported per (Nreal, Nparallel): JAX compile time, steady-state JAX and cupy time per call, XLA
memory of the compiled likelihood (temporaries + arguments + outputs), and the largest relative
difference of the total log likelihood.

Usage
-----
    python benchmark_jax_likelihood.py [--Nreal 2 5] [--Nparallel 1 75 375] [--Nres 10000] [--outfile out.csv]
"""
import time
import argparse

FIDUCIAL = dict(m_mu=0.6, m_sigma=0.15, rh_disk=3.31, r_bulge=0.75, q_bd=0.33, a_alpha=0.5)  ## Table 1, arXiv:2604.03390
COLUMNS = ["Nreal", "Nparallel", "Nf", "Ngrid", "Nres", "compile_s", "jax_s", "cupy_s", "speedup", "jax_mem_GB",
           "max_rel_diff", "error"]


def main(args):
    from pelargir import backend
    backend.set_backend('jax')
    xp = backend.xp
    import numpy as np
    import legwork as lw
    import astropy.units as u
    import jax
    import jax.numpy as jnp
    from pelargir.inference import GalacticBinaryPrior, FG_Likelihood, Nres_Likelihood, Res_Astro_Likelihood
    from pelargir.utils import lisa_noise_psd, to_numpy
    from pelargir import jax_population as jp
    from pelargir import jax_likelihood as jl

    def sync():
        xp.cuda.Device().synchronize()

    fbins = xp.arange(args.fmin - args.fbin/2, args.fmax + args.fbin/2, args.fbin)
    delf = float(fbins[1] - fbins[0])
    Sn = xp.asarray(lisa_noise_psd(fbins))
    rx = xp.asarray(lw.psd.approximate_response_function(to_numpy(fbins)*u.Hz, 19.09*u.mHz).value)
    Tobs = (4*u.yr).to(u.s).value
    rng = np.random.default_rng(args.seed)
    noise = to_numpy(Sn[1:])
    fg_true = noise*10**rng.uniform(-1, 1, noise.size)
    Nf = noise.size

    gbprior = GalacticBinaryPrior(xp.random.default_rng(args.seed))
    bounds = jp.prior_bounds(gbprior)
    theta0 = np.array([FIDUCIAL[k] for k in gbprior.pop_params])
    state = np.asarray(jp.sample_theta(jax.random.key(args.seed), jnp.asarray(theta0), args.Nres, bounds)).T

    rows = []
    for Nr in args.Nreal:
        fg_like = FG_Likelihood(xp.asarray(fg_true), xp.asarray(0.1), xp.asarray(noise), Nreal=Nr,
                                hp_alpha=5, hp_beta=0.05)
        nres_like = Nres_Likelihood(args.Nres)
        res_like = Res_Astro_Likelihood(xp.random.default_rng(1), xp.asarray(state), fbins, rx, duration=Tobs,
                                        scatter=False, dynamic_scatter=False)
        consts = jl.make_consts(fg_like, nres_like, fbins + 0.5*delf, Sn, rx, Tobs, 7.0, bounds)
        for Np in args.Nparallel:
            row = dict(Nreal=Nr, Nparallel=Np, Nf=Nf, Ngrid=int(consts.log10_cgrid.size), Nres=args.Nres)
            try:
                thetas = np.tile(theta0, (Np, 1))*(1 + 0.02*rng.standard_normal((Np, 6)))
                draws = fg_true[:, None, None]*10**(0.1*rng.standard_normal((Nf, Nr, Np)))
                Nhat = rng.poisson(args.Nres, (Nr, Np))
                d_x, N_x = xp.asarray(draws), xp.asarray(Nhat)
                gbprior_c = GalacticBinaryPrior(xp.random.default_rng(0), Nreal=Nr)
                gbprior_c.condition({k: xp.asarray(thetas[:, i]) for i, k in enumerate(gbprior.pop_params)})

                def cupy_call():
                    return (fg_like.ln_prob(d_x) + nres_like.ln_prob(N_x)
                            + res_like.static_ln_conditional_prob(gbprior_c, d_x, Sn, 7.0))

                j_args = (jl.to_jax(d_x), jl.to_jax(N_x), jnp.asarray(state), jnp.asarray(thetas), consts)

                def jax_call():
                    return sum(jax.block_until_ready(jl.ln_like(*j_args)))

                t0 = time.time(); jax_call(); row['compile_s'] = time.time() - t0
                times = []
                for _ in range(args.repeats):
                    t0 = time.time(); j = jax_call(); times.append(time.time() - t0)
                row['jax_s'] = min(times)
                mem = jl.ln_like.lower(*j_args).compile().memory_analysis()
                row['jax_mem_GB'] = (mem.temp_size_in_bytes + mem.argument_size_in_bytes + mem.output_size_in_bytes)/1e9

                cupy_call(); sync()
                times = []
                for _ in range(args.repeats):
                    sync(); t0 = time.time(); c = cupy_call(); sync(); times.append(time.time() - t0)
                row['cupy_s'] = min(times)
                row['speedup'] = row['cupy_s']/row['jax_s']
                j, c = np.asarray(j), to_numpy(c)
                row['max_rel_diff'] = float(np.max(np.abs(j - c)/np.abs(c)))
                del d_x, N_x, j_args
            except Exception as err:
                row['error'] = str(err).splitlines()[0][:120]
            xp.get_default_memory_pool().free_all_blocks()
            rows.append(row)
            print(" ".join("{}={}".format(k, "{:.4g}".format(v) if isinstance(v, float) else v)
                           for k, v in row.items()), flush=True)

    if args.outfile is not None:
        import csv
        with open(args.outfile, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, '') for k in COLUMNS})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--Nreal', type=int, nargs='+', default=[2, 5])
    parser.add_argument('--Nparallel', type=int, nargs='+', default=[1, 75, 375])
    parser.add_argument('--Nres', type=int, default=10000)
    parser.add_argument('--fmin', type=float, default=1e-4)
    parser.add_argument('--fmax', type=float, default=5e-3)
    parser.add_argument('--fbin', type=float, default=2e-5)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=170817)
    parser.add_argument('--outfile', type=str, default=None)
    main(parser.parse_args())
