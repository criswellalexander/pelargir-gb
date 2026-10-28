"""
Logging and progress-bar helpers shared by the command-line scripts. This module does not import the
array backend, so a script can set up its logging before choosing the backend.

Library modules log through logging.getLogger('pelargir') and draw bars only when their caller passes
progress=True; both stay silent unless a script calls setup_logging (or the user configures 'pelargir').
"""
import time
import logging
import contextlib

from tqdm.auto import tqdm

LOGGER_NAME = 'pelargir'
LOG_FORMAT = '[%(asctime)s] %(levelname)s %(name)s: %(message)s'


class _TqdmHandler(logging.StreamHandler):
    '''

    A stderr handler that writes through tqdm.write, so a log line never breaks an active progress bar.

    Arguments
    ------------------------------
    None

    Returns
    ------------------------------
    None

    '''

    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


def add_verbosity_args(parser):
    '''

    Add the mutually exclusive -v/--verbose and -q/--quiet flags to a script's parser.

    Arguments
    ------------------------------
    parser : argparse.ArgumentParser
    	The script's parser.

    Returns
    ------------------------------
    parser : argparse.ArgumentParser
    	The same parser.

    '''
    g = parser.add_mutually_exclusive_group()
    g.add_argument('-v', '--verbose', action='store_true', help="Also log DEBUG messages (e.g. every training epoch).")
    g.add_argument('-q', '--quiet', action='store_true', help="Log warnings only and hide pelargir's progress bars.")
    return parser


def setup_logging(verbose=False, quiet=False):
    '''

    Configure logging for a script: other libraries at WARNING, pelargir at INFO (DEBUG with verbose,
    WARNING with quiet), all through one tqdm-safe stderr handler with timestamps.

    Arguments
    ------------------------------
    verbose : bool
    	Log pelargir's DEBUG messages. (Default: False)
    quiet : bool
    	Log pelargir's warnings only, and disable its progress bars. (Default: False)

    Returns
    ------------------------------
    logger : logging.Logger
    	The 'pelargir' logger.

    '''
    handler = _TqdmHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO)
    return logger


def progress_bar(iterable=None, total=None, desc=None, unit='it', enabled=True, **kwargs):
    '''

    A tqdm progress bar with the project's settings (stderr, resizing, at most one refresh per second),
    disabled when the caller turns progress off or pelargir logs above INFO (a script's -q).

    Arguments
    ------------------------------
    iterable : iterable or None
    	What to iterate over; None for a manually updated bar. (Default: None)
    total : int or float or None
    	Expected number of units; None takes len(iterable) when it has one. (Default: None)
    desc : str or None
    	Label shown before the bar. (Default: None)
    unit : str
    	Name of one unit. (Default: 'it')
    enabled : bool
    	False disables the bar. (Default: True)
    **kwargs :
    	Other tqdm arguments (e.g. leave, unit_scale, position).

    Returns
    ------------------------------
    bar : tqdm.tqdm
    	The bar (a no-op when disabled).

    '''
    disable = (not enabled) or logging.getLogger(LOGGER_NAME).getEffectiveLevel() > logging.INFO
    kwargs = {**dict(dynamic_ncols=True, mininterval=1.0), **kwargs}
    return tqdm(iterable, total=total, desc=desc, unit=unit, disable=disable, **kwargs)


@contextlib.contextmanager
def timed(logger, message, level=logging.INFO):
    '''

    Log a phase's start ("message ...") and its end with the elapsed time ("message: done in X s").

    Arguments
    ------------------------------
    logger : logging.Logger
    	Logger to use.
    message : str
    	Phase description.
    level : int
    	Logging level of both lines. (Default: logging.INFO)

    Returns
    ------------------------------
    timer : dict
    	Filled with 'seconds' when the block ends, for callers that also report the time.

    '''
    timer = {}
    logger.log(level, "%s ...", message)
    t0 = time.time()
    yield timer
    timer['seconds'] = time.time() - t0
    logger.log(level, "%s: done in %s", message, format_seconds(timer['seconds']))


def format_seconds(seconds):
    '''

    A duration as a short human-readable string.

    Arguments
    ------------------------------
    seconds : float
    	Duration [s].

    Returns
    ------------------------------
    text : str
    	e.g. '0.42 s', '37.1 s', '12 min 5 s', '3 h 4 min'.

    '''
    if seconds < 10:
        return f"{seconds:.2f} s"
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{int(seconds//60)} min {int(seconds % 60)} s"
    return f"{int(seconds//3600)} h {int(seconds % 3600//60)} min"
