"""Shared constants for run_benchmark.py and plot_benchmarks.py.

A single source of truth, so the two scripts can't silently drift apart
(e.g. the plot annotating a different tolerance than the one actually
used to produce the data it's plotting).
"""

# Every solver's primary convergence tolerance, so none can drift out of
# sync with the others: each bounds a different internal quantity (SSQP's
# per-stage KKT/constraint residual, Ipopt/Fatrop's scaled KKT error,
# ProxDDP's primal-dual residual, CSQP's convergence gap), so this is the
# fairest guarantee available short of reimplementing each library's own
# termination criterion
TOLERANCE = 1e-2
