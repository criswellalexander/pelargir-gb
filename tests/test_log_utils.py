"""
Tests for the scripts' logging and progress-bar helpers (pelargir/log_utils.py): the levels set by
setup_logging and its -v/-q flags, when progress bars are shown, the tqdm-safe handler, and timed.
"""
import io
import logging
import argparse

import pytest

from pelargir import log_utils as lu


@pytest.fixture
def restore_logging():
    '''Undo setup_logging's changes to the root and pelargir loggers after a test.'''
    root, pel = logging.getLogger(), logging.getLogger(lu.LOGGER_NAME)
    saved = (root.handlers[:], root.level, pel.level)
    yield
    root.handlers[:] = saved[0]
    root.setLevel(saved[1])
    pel.setLevel(saved[2])


@pytest.mark.parametrize("flags,level", [((False, False), logging.INFO), ((True, False), logging.DEBUG),
                                         ((False, True), logging.WARNING)])
def test_setup_logging_levels(restore_logging, flags, level):
    logger = lu.setup_logging(*flags)
    assert logger is logging.getLogger('pelargir') and logger.level == level
    ## other libraries stay at WARNING, through one tqdm-safe handler
    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert len(root.handlers) == 1 and isinstance(root.handlers[0], lu._TqdmHandler)
    ## script loggers ('pelargir.scripts.<name>') inherit the level
    assert logging.getLogger('pelargir.scripts.x').getEffectiveLevel() == level


def test_verbosity_flags_are_exclusive():
    parser = lu.add_verbosity_args(argparse.ArgumentParser())
    assert vars(parser.parse_args([])) == dict(verbose=False, quiet=False)
    assert parser.parse_args(['-v']).verbose and parser.parse_args(['--quiet']).quiet
    with pytest.raises(SystemExit):
        parser.parse_args(['-v', '-q'])


def test_progress_bar_follows_the_level_and_enabled(restore_logging):
    lu.setup_logging()
    assert not lu.progress_bar(total=3).disable
    assert lu.progress_bar(total=3, enabled=False).disable
    lu.setup_logging(quiet=True)
    assert lu.progress_bar(total=3).disable
    ## an iterable passes through whether or not the bar shows
    assert list(lu.progress_bar(range(4))) == [0, 1, 2, 3]


def test_handler_writes_through_tqdm_while_a_bar_is_open():
    stream = io.StringIO()
    handler = lu._TqdmHandler(stream)
    handler.setFormatter(logging.Formatter(lu.LOG_FORMAT))
    logger = logging.getLogger('pelargir.test_log_utils')
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        with lu.tqdm(total=2, file=io.StringIO()) as bar:
            logger.info("halfway")
            bar.update()
    finally:
        logger.removeHandler(handler)
        logger.propagate = True
    assert "INFO pelargir.test_log_utils: halfway" in stream.getvalue()


def test_timed_logs_start_and_end(caplog):
    caplog.set_level(logging.INFO, logger='pelargir')
    logger = logging.getLogger('pelargir.test_log_utils')
    with lu.timed(logger, "a phase") as t:
        pass
    assert [r.getMessage() for r in caplog.records][0] == "a phase ..."
    assert caplog.records[1].getMessage().startswith("a phase: done in ") and t['seconds'] >= 0


def test_format_seconds():
    assert lu.format_seconds(0.4213) == "0.42 s"
    assert lu.format_seconds(37.14) == "37.1 s"
    assert lu.format_seconds(725) == "12 min 5 s"
    assert lu.format_seconds(3*3600 + 4*60 + 9) == "3 h 4 min"
