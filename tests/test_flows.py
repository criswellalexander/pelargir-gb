"""
Tests for the flow emulator (flows.py) and the per-galaxy rho_thresh of the JAX forward model:
per-galaxy thresholds against the thresholder on the materialized draws, hyperpriors, bands,
training sets, transforms, the Gauss-Legendre count marginalization, and save/load.

Runs under any backend (the forward-model reference is jax_thresholding, itself tested against
serial_array_sort); needs torch and zuko.
"""
import logging
import os

os.environ.setdefault("PELARGIR_BACKEND", "numpy")

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

pytest.importorskip("jax")
pytest.importorskip("zuko")
import torch
import jax
import jax.numpy as jnp

from pelargir import backend
from pelargir import flows
from pelargir import jax_population as jp
from pelargir import jax_thresholding as jt
from pelargir.inference import GalacticBinaryPrior, PopulationHyperPrior
from pelargir.utils import get_amp_freq, lisa_noise_psd

xp = backend.xp
FIDUCIAL = np.array([0.6, 0.15, 3.31, 0.75, 0.33, 0.5])
FBINS = flows.model_fbins(1e-4, 6e-4, 2e-5)
DELF = FBINS[1] - FBINS[0]
BOUNDS = jp.prior_bounds(GalacticBinaryPrior(np.random.default_rng(0)))


def forward_consts():
    import legwork as lw
    import astropy.units as u
    rx = lw.psd.approximate_response_function(FBINS*u.Hz, 19.09*u.mHz).value
    return FBINS + 0.5*DELF, lisa_noise_psd(FBINS, cpu=True), rx


# =============================================================================
# Per-galaxy rho_thresh in the forward model
# =============================================================================

@pytest.mark.parametrize("cut", [None, 1.0])
def test_per_galaxy_rho_matches_the_thresholder_on_the_same_draws(cut):
    edges, Sn, rx = forward_consts()
    thetas = np.stack([FIDUCIAL, [0.8, 0.1, 5.0, 0.5, 0.6, 1.0], [0.4, 0.2, 2.0, 1.0, 0.2, 0.0]])
    rhos = np.array([5.0, 7.0, 9.5])
    Ns = np.array([[30000, 20000, 25000], [22000, 31000, 18000]])
    key = jax.random.key(3)
    Nres, fg, mask, Nres_f = jp.jax_forward_model(key, thetas, Ns, BOUNDS, edges, Sn, rx, flows.DURATION, 1/DELF,
                                                  snr_thresh=rhos, batch_size=4, prefilter_snr=cut,
                                                  return_mask=True, return_nres_f=True)
    assert_array_equal(Nres_f[1:].sum(axis=0), Nres)
    Nr, Np = Ns.shape
    for r in range(Nr):
        for p in range(Np):
            g = r*Np + p
            draw = jp.sample_galaxy_draw(key, g, thetas[p], Ns[r, p], BOUNDS, out_module=xp)
            A, f = get_amp_freq(draw)
            r_N, r_fg, r_mask = jt.jax_threshold(xp.array([f, A]), edges, Sn, rx, flows.DURATION, 1/DELF,
                                                 snr_thresh=rhos[p], return_mask=True, prefilter_snr=None)
            assert int(Nres[r, p]) == int(jp._host(r_N))
            assert_allclose(fg[:, r, p], jp._host(r_fg), rtol=1e-12, atol=0)
            assert_array_equal(mask[:Ns[r, p], r, p], jp._host(r_mask))


def test_prefilter_above_the_smallest_rho_raises():
    edges, Sn, rx = forward_consts()
    with pytest.raises(ValueError, match="must not exceed"):
        jp.jax_forward_model(jax.random.key(0), np.stack([FIDUCIAL]*2), np.full((1, 2), 1000), BOUNDS, edges, Sn, rx,
                             flows.DURATION, 1/DELF, snr_thresh=np.array([7.0, 0.5]), prefilter_snr=1.0)


# =============================================================================
# Hyperpriors, bands
# =============================================================================

def test_hyperprior_matches_PopulationHyperPrior():
    ref = PopulationHyperPrior(xp.random.default_rng(0)).hyperprior_dict
    assert flows.CONTEXT_NAMES[:flows.N_POP] == GalacticBinaryPrior(None).pop_params == list(ref)
    for k, d in ref.items():
        mine = flows.HYPERPRIOR[k]
        if hasattr(d, 'loc'):
            assert (float(d.loc), float(d.scale)) == (mine.kwds['loc'], mine.kwds['scale'])
        else:
            assert float(d.a) == mine.args[0] and mine.kwds.get('scale', 1.0) == 1.0


def test_new_priors():
    c = flows.sample_context(np.random.default_rng(1), 20000)
    rho, loglam = c[:, 6], c[:, 7]
    assert rho.min() >= flows.RHO_MIN
    assert abs(np.median(rho) - 8) < 0.1
    assert np.log10(5e5) <= loglam.min() and loglam.max() <= np.log10(5e7)


def test_bands_partition_the_grid():
    fs = flows.model_fbins()[1:]
    bands = flows.make_bands(fs)
    assert [b.nf for b in bands] == [5]*9
    assert np.concatenate([np.arange(fs.size)[b.slice] for b in bands]).tolist() == list(range(fs.size))
    bands = flows.make_bands(fs, edges=[1e-4, 3e-4, 1e-3])
    assert [b.nf for b in bands] == [10, 35]
    assert_allclose(np.concatenate([b.fs for b in bands]), fs)
    with pytest.raises(ValueError, match="no bins"):
        flows.make_bands(fs, edges=[1e-4, 1.05e-4, 1e-3])


# =============================================================================
# Training sets
# =============================================================================

@pytest.fixture(scope="module")
def small_set(tmp_path_factory):
    ## lambda_tot ~ 1e5-2e5 keeps the set cheap while every bin still has unresolved binaries
    saved = flows.HYPERPRIOR['log10_lambda_tot']
    import scipy.stats as ss
    flows.HYPERPRIOR['log10_lambda_tot'] = ss.uniform(loc=5.0, scale=0.3)
    try:
        ts = flows.draw_training_set(24, 3, FBINS, seed=5, progress=False)
        ts2 = flows.draw_training_set(24, 3, FBINS, seed=5, max_binaries_per_batch=3e5, progress=False)
    finally:
        flows.HYPERPRIOR['log10_lambda_tot'] = saved
    path = tmp_path_factory.mktemp("ts")/"ts.npz"
    ts.save(path)
    return ts, ts2, flows.TrainingSet.load(path)


def test_training_set_shapes_reproducibility_and_bands(small_set):
    ts, ts2, loaded = small_set
    Nf = FBINS.size - 1
    assert ts.context.shape == (72, 8) and ts.nres_f.shape == (72, Nf) and ts.psd.shape == (72, Nf)
    assert_array_equal(ts.context[::3], ts.context[1::3])      ## realizations share their draw
    assert_array_equal(ts.nres_f, ts2.nres_f)
    ## foreground sums use an atomic segment_sum (~1e-16 run to run)
    assert_allclose(ts.psd, ts2.psd, rtol=1e-12, atol=0)
    assert np.all(ts.psd > 0)
    assert abs(ts.ntot.mean()/np.mean(10**ts.context[:, 7]) - 1) < 0.01
    for k in ('context', 'nres_f', 'psd', 'ntot', 'fbins'):
        assert_array_equal(getattr(loaded, k), getattr(ts, k))
    for b in flows.make_bands(ts.fs, bins_per_band=4):
        c, N, S = flows.band_view(ts, b)
        assert_array_equal(N, ts.nres_f[:, b.start:b.stop].sum(axis=1))
        assert S.shape == (72, b.nf)


class _Killed(Exception):
    pass


def kill_after(monkeypatch, n=None):
    '''
    Make jp._launch (each batch's kernel call) raise _Killed after n calls, as a job killed mid-run
    (n None: never). Returns the list of capacities it was called with.
    '''
    real = getattr(jp._launch, 'real', jp._launch)
    calls = []

    def wrapped(*args, **kwargs):
        if n is not None and len(calls) >= n:
            raise _Killed()
        calls.append(kwargs['capacity'])
        return real(*args, **kwargs)
    wrapped.real = real
    monkeypatch.setattr(jp, '_launch', wrapped)
    return calls


@pytest.fixture
def cheap_lambda(monkeypatch):
    import scipy.stats as ss
    monkeypatch.setitem(flows.HYPERPRIOR, 'log10_lambda_tot', ss.uniform(loc=5.0, scale=0.3))


def test_interrupted_runs_resume_to_the_same_set(tmp_path, monkeypatch, caplog, cheap_lambda):
    full = flows.draw_training_set(12, 2, FBINS, seed=7, progress=False)
    ## one galaxy per batch (each galaxy is one chunk of DRAW_CHUNK)
    kw = dict(progress=False, chunk_dir=tmp_path/"run", max_binaries_per_batch=4e5)
    caplog.set_level(logging.INFO, logger='pelargir')
    kill_after(monkeypatch, 5)
    with pytest.raises(_Killed):
        flows.draw_training_set(12, 2, FBINS, seed=7, **kw)
    assert "starting a new run: 24 galaxies" in caplog.text

    ## the resumed run batches differently and only runs what is left
    caplog.clear()
    calls = kill_after(monkeypatch)
    resumed = flows.draw_training_set(12, 2, FBINS, seed=7, **dict(kw, max_binaries_per_batch=1e9))
    k = resumed.meta['resumed_galaxies']
    assert "resuming from {}: {} of 24 galaxies done".format(tmp_path/"run", k) in caplog.text
    assert 0 < k < 24 and resumed.meta['simulated_galaxies'] == 24 - k
    assert len(calls) < 24
    assert_array_equal(resumed.nres_f, full.nres_f)
    assert_allclose(resumed.psd, full.psd, rtol=1e-12, atol=0)

    caplog.clear()
    again = flows.draw_training_set(12, 2, FBINS, seed=7, **kw)
    assert "is complete" in caplog.text and again.meta['simulated_galaxies'] == 0
    assert_array_equal(again.nres_f, full.nres_f)
    with pytest.raises(ValueError, match="different settings"):
        flows.draw_training_set(12, 2, FBINS, seed=8, **kw)


def test_resume_during_overflow_reruns(tmp_path, monkeypatch, cheap_lambda):
    ## tiny survivor buffers, so most galaxies overflow tier 0 and rerun at tier 1 (and some at tier 2)
    caps = (0.002, 0.01, None)
    full = flows.draw_training_set(12, 2, FBINS, seed=9, capacity_fracs=caps, progress=False)
    assert full.meta['overflow_reruns'] > 2
    kw = dict(progress=False, chunk_dir=tmp_path/"run", max_binaries_per_batch=4e5, capacity_fracs=caps)
    ## 24 tier-0 batches of one galaxy, then killed during tier 1
    kill_after(monkeypatch, 26)
    with pytest.raises(_Killed):
        flows.draw_training_set(12, 2, FBINS, seed=9, **kw)
    calls = kill_after(monkeypatch)
    resumed = flows.draw_training_set(12, 2, FBINS, seed=9, **kw)
    ## no galaxy goes through tier 0 again
    assert jp.tier_capacity(caps[0], jp.DRAW_CHUNK) not in calls
    assert resumed.meta['simulated_galaxies'] == 0
    assert_array_equal(resumed.nres_f, full.nres_f)
    assert_allclose(resumed.psd, full.psd, rtol=1e-12, atol=0)


def test_rows_without_their_flag_are_recomputed(tmp_path, cheap_lambda):
    kw = dict(progress=False, chunk_dir=tmp_path/"run")
    full = flows.draw_training_set(12, 2, FBINS, seed=7, **kw)
    ## a kill between writing a batch's results and its flags: results written, rows not marked done
    flags = np.lib.format.open_memmap(tmp_path/"run"/"next_tier.npy", mode='r+')
    nres = np.lib.format.open_memmap(tmp_path/"run"/"nres_f.npy", mode='r+')
    flags[[0, 5]] = 0
    nres[[0, 5]] = 12345
    flags.flush(); nres.flush()
    del flags, nres
    again = flows.draw_training_set(12, 2, FBINS, seed=7, **kw)
    assert again.meta['simulated_galaxies'] == 2
    assert_array_equal(again.nres_f, full.nres_f)


def test_training_set_does_not_depend_on_batching(small_set):
    ts, ts2, _ = small_set
    assert ts.meta['widths'] != ts2.meta['widths']
    assert_array_equal(ts.nres_f, ts2.nres_f)
    assert_allclose(ts.psd, ts2.psd, rtol=1e-12, atol=0)


# =============================================================================
# forward_galaxies: many galaxies at fixed shapes
# =============================================================================

def test_pad_buckets_matches_pad_bucket():
    Ns = np.concatenate([[0, 1, 1024, 1025], np.random.default_rng(0).integers(1, 10**8, 500)])
    assert_array_equal(jp.pad_buckets(Ns), [jp.pad_bucket(n) for n in Ns])


def test_mask_free_batch_matches_the_masked_one():
    edges, Sn, rx = forward_consts()
    B, n_pad = 3, jp.pad_bucket(30000)
    args = (jp.galaxy_keys(jax.random.key(2), B), jnp.asarray(np.stack([FIDUCIAL]*B)), jnp.asarray([30000, 20000, 0]),
            *map(jnp.asarray, (edges, Sn, rx)), flows.DURATION, 1/DELF, jnp.asarray([5.0, 7.0, 9.0]),
            jnp.full(B, 1.0), BOUNDS, n_pad)
    for capacity in (None, 8192):
        a = jp._forward_batch(*args, capacity=capacity)
        b = jp._forward_batch(*args, capacity=capacity, with_mask=False)
        assert len(b) == 3
        assert_array_equal(np.asarray(a[0]), np.asarray(b[0]))
        assert_array_equal(np.asarray(a[3]), np.asarray(b[2]))
        ## the foreground sums use an atomic segment_sum (~1e-16 run to run)
        assert_allclose(np.asarray(a[1]), np.asarray(b[1]), rtol=1e-12, atol=0)


def test_chunk_buckets():
    K = 4096
    Ns = np.concatenate([[0, 1, K, K + 1, 2*K + 1], np.random.default_rng(1).integers(1, 10**8, 300)])
    b = jp.chunk_buckets(Ns, K)
    n = np.maximum(-(-Ns//K), 1)
    assert np.all(b >= n)
    ## at most CHUNK_GROWTH over the chunks needed, or one more chunk
    assert np.all(b <= np.maximum(np.ceil(n*jp.CHUNK_GROWTH), n + 1))
    assert_array_equal(b[:5], [1, 1, 1, 2, 3])


## chunk-boundary cases (N < K, N = K, N = 2K + 1, N = 0), a galaxy large enough to overflow small
## survivor buffers (fiducial, N = 1e5, ~2000 survivors), and mixed sizes
CHUNK_K = 4096
CHUNK_NS = np.array([3000, CHUNK_K, 2*CHUNK_K + 1, 0, 100000, 61000, 5*CHUNK_K, 17, 45000])


@pytest.mark.parametrize("variant", ["default", "width 1", "overflow", "unfiltered"])
def test_forward_galaxies_matches_the_thresholder(variant):
    edges, Sn, rx = forward_consts()
    consts = (edges, Sn, rx, flows.DURATION, 1/DELF)
    thetas = np.stack([FIDUCIAL, [0.8, 0.1, 5.0, 0.5, 0.6, 1.0], [0.4, 0.2, 2.0, 1.0, 0.2, 0.0]]*3)
    thetas[4] = FIDUCIAL
    Ns, K = CHUNK_NS, CHUNK_K
    rhos = np.array([5.0, 7.0, 9.5]*3)
    rhos[4] = 7.0
    idx = np.arange(100, 109)
    key = jax.random.key(4)
    kw = {"default": {}, "width 1": dict(widths={int(b): 1 for b in jp.chunk_buckets(Ns, K)}),
          "overflow": dict(capacity_fracs=(0.002, 0.01, None)), "unfiltered": dict(prefilter_snr=None)}[variant]
    nres_f, fg, info = jp.forward_galaxies(key, idx, thetas, Ns, rhos, consts, BOUNDS, K=K, **kw)
    n_buckets = len(np.unique(jp.chunk_buckets(Ns, K)))
    if variant == "overflow":
        assert info['overflow'] > 0
    else:
        assert info['overflow'] == 0
    ## a bucket's short last batch may add one more width
    assert info['compiles'] <= 2*n_buckets*len(kw.get('capacity_fracs', jp.CAPACITY_FRACS))
    ## every galaxy runs once in the first pass (a batch must not spill into the next bucket)
    assert info['simulated'] == Ns.size
    assert sum(b['binaries'] for b in info['buckets'] if b['tier'] == 0) == Ns.sum()
    for i in range(Ns.size):
        if Ns[i] == 0:
            assert not nres_f[i].any() and not fg[i].any()
            continue
        draw = jp.sample_galaxy_draw_chunked(key, int(idx[i]), thetas[i], Ns[i], K, BOUNDS, out_module=xp)
        assert draw.shape == (4, Ns[i])
        A, f = get_amp_freq(draw)
        r_N, r_fg, r_mask = jt.jax_threshold(xp.array([f, A]), edges, Sn, rx, flows.DURATION, 1/DELF,
                                             snr_thresh=rhos[i], prefilter_snr=None, return_mask=True)
        assert nres_f[i, 1:].sum() == int(jp._host(r_N))
        assert nres_f[i].sum() == int(jp._host(r_mask).sum())
        assert_allclose(fg[i], jp._host(r_fg), rtol=1e-12, atol=0)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_progress_eta_tracks_real_time():
    '''
    With a constant compute cost per binary and a compile per new shape, the ETA after each bucket
    is the true time left, so consecutive ETAs drop by exactly each bucket's time. With two devices
    the compute is shared, and each device compiles the shape side by side (two compiles recorded,
    one compile's wall time).
    '''
    rng = np.random.default_rng(0)
    binaries = rng.integers(10**8, 10**10, 12).astype(float)
    shapes = rng.integers(1, 3, 12)
    ns, t_compile = 11e-9, 2.7
    for n_dev in (1, 2):
        clock = _Clock()
        prog = jp._Progress(n_dev, clock)
        prog.add(binaries.sum(), shapes.sum())
        times = binaries*ns/n_dev + shapes*t_compile
        etas = []
        for b, n, t in zip(binaries, shapes, times):
            clock.t += t
            for _ in range(n*n_dev):
                prog.compiled(t_compile)
            prog.finished(b)
            prog.shapes_done(n)
            etas.append(prog.remaining())
        assert_allclose(etas, times.sum() - np.cumsum(times), rtol=1e-9, atol=1e-6)
        assert_allclose(-np.diff(etas), times[1:], rtol=1e-9)
    ## a later tier adds its work to the same ETA
    prog.add(5e9, 1)
    assert_allclose(prog.remaining(), 5e9*ns/2 + t_compile, rtol=1e-9)


MULTI_DEVICE_SCRIPT = r'''
import sys, numpy as np, scipy.stats as ss
from pelargir import backend; backend.set_backend("jax"); jax = backend.import_jax()
from pelargir import flow_data as fd, jax_population as jp
assert len(jax.local_devices()) == 2, jax.local_devices()
fd.HYPERPRIOR["log10_lambda_tot"] = ss.uniform(loc=4.0, scale=0.3)
fbins = fd.model_fbins(1e-4, 3e-4, 2e-5)
consts, bounds = fd._forward_consts(fbins)
## forward_galaxies on 1 and 2 devices, one galaxy per batch so both devices get work
ctx = fd.sample_context(np.random.default_rng(2), 12)
Ns = np.random.default_rng(3).integers(0, 30000, 12)
K = 4096
widths = {int(b): 1 for b in jp.chunk_buckets(Ns, K)}
one = jp.forward_galaxies(jax.random.key(1), np.arange(12), ctx[:, :6], Ns, ctx[:, 6], consts, bounds, widths=widths, K=K, devices=1)
two = jp.forward_galaxies(jax.random.key(1), np.arange(12), ctx[:, :6], Ns, ctx[:, 6], consts, bounds, widths=widths, K=K, devices=2)
assert (one[0] == two[0]).all()
assert np.allclose(one[1], two[1], rtol=1e-12, atol=0)
assert min(two[2]["device_batches"].values()) > 0, two[2]["device_batches"]
## a run killed on 2 devices resumes on 1 to the same set
full = fd.draw_training_set(8, 2, fbins, seed=4, progress=False, devices=1)
class Killed(Exception):
    pass
real, calls = jp._launch, []
def killing(*a, **k):
    if len(calls) >= 5:
        raise Killed()
    calls.append(1)
    return real(*a, **k)
jp._launch = killing
try:
    fd.draw_training_set(8, 2, fbins, seed=4, progress=False, devices=2, chunk_dir=sys.argv[1], max_binaries_per_batch=1)
    raise SystemExit("not killed")
except Killed:
    pass
jp._launch = real
res = fd.draw_training_set(8, 2, fbins, seed=4, progress=False, devices=1, chunk_dir=sys.argv[1])
assert 0 < res.meta["resumed_galaxies"] < 16, res.meta["resumed_galaxies"]
assert (res.nres_f == full.nres_f).all() and np.allclose(res.psd, full.psd, rtol=1e-12, atol=0)
print("MULTI-DEVICE OK", two[2]["device_batches"], res.meta["resumed_galaxies"])
'''


def test_multiple_devices_give_the_same_results(tmp_path):
    '''Two (CPU) devices: same results as one, both used, and a run killed on two resumes on one.'''
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if not k.startswith("PELARGIR_")}
    env.update(JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=2",
               PYTHONPATH=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
    res = subprocess.run([sys.executable, "-c", MULTI_DEVICE_SCRIPT, str(tmp_path/"run")], env=env,
                         capture_output=True, text=True, timeout=900)
    assert res.returncode == 0 and "MULTI-DEVICE OK" in res.stdout, res.stderr[-3000:]


def test_forward_galaxies_checks_the_prefilter():
    edges, Sn, rx = forward_consts()
    with pytest.raises(ValueError, match="must not exceed"):
        jp.forward_galaxies(jax.random.key(0), [0], [FIDUCIAL], [1000], [0.5], (edges, Sn, rx, flows.DURATION, 1/DELF),
                            BOUNDS)
    with pytest.raises(ValueError, match="RHO_MIN"):
        flows.draw_training_set(2, 1, FBINS, seed=0, prefilter_snr=1.5, progress=False)


# =============================================================================
# Transforms
# =============================================================================

def test_band_transform_round_trip_and_jacobian():
    rng = np.random.default_rng(2)
    N = rng.integers(0, 500, 200)
    S = 10**rng.uniform(-41, -37, (200, 3))
    t = flows.BandTransform(3).fit(N, S, rng)
    u = rng.uniform(size=200)
    z, logdet = t(torch.as_tensor(N + u), torch.as_tensor(S))
    N_back, S_back = t.inverse(z)
    assert_array_equal(N_back.numpy(), N)
    assert_allclose(S_back.numpy(), S, rtol=1e-12)
    f = lambda x: t(x[:1], x[1:][None])[0][0]
    for i in range(3):
        x = torch.cat([torch.as_tensor([N[i] + u[i]]), torch.as_tensor(S[i])]).to(torch.float64)
        J = torch.autograd.functional.jacobian(f, x)
        assert_allclose(logdet[i].item(), torch.logdet(J).item(), rtol=1e-12)
    with pytest.raises(ValueError, match="positive"):
        flows.BandTransform(3).fit(N, np.where(S > 1e-40, S, 0.0), rng)


# =============================================================================
# Count marginalization and save/load
# =============================================================================

@pytest.fixture(scope="module")
def toy_flow():
    '''A flow fitted to N ~ Poisson(lam(c)), log10 S ~ N(-38 + 0.3 c, 0.05), c ~ U(0,1).'''
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n = 20000
    c = rng.uniform(size=(n, 1))
    lam = 20 + 30*c[:, 0]
    N = rng.poisson(lam)
    S = 10**(-38 + 0.3*c + 0.05*rng.standard_normal((n, 1)))
    band = flows.FrequencyBand(0, 0, 1, np.array([1e-4]))
    flow = flows.BandFlow(band, n_context=1)
    flows.train_band_flow(flow, c, N, S, n_epochs=6, batch_size=256, lr=3e-3, seed=0, progress=False)
    return flow.eval()


def test_count_quadrature_converges_and_recovers_the_joint_density(toy_flow):
    import scipy.stats as ss
    rng = np.random.default_rng(9)
    c = rng.uniform(size=(300, 1))
    lam = 20 + 30*c[:, 0]
    N = rng.poisson(lam)
    logS = -38 + 0.3*c[:, 0] + 0.05*rng.standard_normal(300)
    S = 10**logS[:, None]
    with torch.no_grad():
        lp = {n: toy_flow.log_prob(c, N, S, n_quad=n).numpy() for n in (8, 32, 64, 512)}
    ## the spline flow's integrand has kinks, so Gauss-Legendre converges only algebraically
    ## (~n^-2 in the worst rows); most rows converge to float32 precision by 8 nodes
    err = {n: np.max(np.abs(lp[n] - lp[512])) for n in (8, 32, 64)}
    assert err[8] > err[32] > err[64]
    assert err[32] < 1e-4 and err[64] < 2e-5
    assert np.median(np.abs(lp[8] - lp[512])) < 1e-5
    ## the analytic joint density of (N, S), S in Hz^-1: Poisson x log-normal
    exact = ss.poisson.logpmf(N, lam) + ss.norm.logpdf(logS, -38 + 0.3*c[:, 0], 0.05) - np.log(S[:, 0]*np.log(10))
    assert abs(np.mean(lp[64] - exact)) < 0.05
    assert np.std(lp[64] - exact) < 0.2


def test_emulator_save_load_reproduces_log_prob(tmp_path, small_set):
    ts = small_set[0]
    em, _ = flows.train_emulator(ts, bands=flows.make_bands(ts.fs, bins_per_band=10), n_epochs=1, progress=False)
    N = np.stack([flows.band_view(ts, b)[1] for b in em.bands], axis=1)
    lp = em.log_prob(ts.context[:5], N[:5], ts.psd[:5])
    em.save(tmp_path)
    em2 = flows.BandedFlowEmulator.load(tmp_path)
    assert_array_equal(em2.log_prob(ts.context[:5], N[:5], ts.psd[:5]), lp)
    Ns, Ss = em2.sample(ts.context[:4])
    assert Ns.shape == (4, len(em.bands)) and np.all(np.isfinite(Ss)) and np.all(Ss > 0)
