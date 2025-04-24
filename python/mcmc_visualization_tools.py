################################################################################
#
# The code is copyright 2024 Michael Betancourt and licensed under the
# new BSD (3-clause) license:
#  https://opensource.org/licenses/BSD-3-Clause
#
# For more information see
#  https://github.com/betanalpha/mcmc_visualization_tools.
#
################################################################################

# Load required libraries
import matplotlib
import matplotlib.pyplot as plot
from matplotlib.colors import LinearSegmentedColormap

import re
from functools import partial
import math
import numpy

# Graphic configuration
light = "#DCBCBC"
light_highlight = "#C79999"
mid = "#B97C7C"
mid_highlight = "#A25050"
dark = "#8F2727"
dark_highlight = "#7C0000"

light_teal = "#6B8E8E"
mid_teal = "#487575"
dark_teal = "#1D4F4F"

from importlib import import_module

lib_names = ['mcmc_analysis_tools_pystan3',
             'mcmc_analysis_tools_pystan2',
             'mcmc_analysis_tools_other']

failed_import = True
for lib_name in lib_names:
  try:
    lib = import_module(lib_name)
  except:
    "" # Fail silently
  else:
    globals()['util'] = lib
    failed_import = False
    break

if failed_import:
  raise ImportError('mcmc_visualization_tools.py requires that ',
                    'mcmc_analysis_tools_[pystan2/pystan3/other].py '
                    'from https://github.com/betanalpha/mcmc_diagnostics ',
                    'is available.')

################################################################################
# Utility Functions
################################################################################

# Emit ValueError if the two input arrays are not the same length.
# @param a First list
# @param a_name Name of first list
# @param b Second list
# @param b_name Name of second list
def check_dimensions(a, a_name, b, b_name):
  if len(a) != len(b):
    raise ValueError(f'The arguments `{a_name}` and `{b_name}` '
                      'are not the same length!')

# Check if the given expectand names exist in the samples object, removing any
# that do not.  Print warning describing any missing names and emit stop if no
# expectand names are found.
# @param names A one-dimensional array of strings
# @param samples A named list
# @return List of valid expectand names
def check_expectand_names(names, samples):
  all_names = samples.keys()


  bad_names = [ name for name in names if name not in all_names ]
  B = len(bad_names)
  if B > 0:
    if B == 1:
      print(f'The expectand name {", ".join(bad_names)} is not '
             'in the `samples` object and will be ignored.')
    else:
      print(f'The expectand names {", ".join(bad_names)} are not '
             'in the `samples` object and will be ignored.')

  good_names = [ name for name in names if name in all_names ]
  if len(good_names) == 0:
    raise ValueError('There are no valid expectand names.')

  return good_names

# Check how many values fall below bin_min and above bin_max and print
# appropriate warming message.
# @param bin_min Lower threshold
# @param bin_max Upper threshold
# @param values A one-dimensional array of values to check
# @param name Value description
def check_bin_containment(bin_min, bin_max, values, name="value"):
  values = numpy.array(values)
  N = values.size

  N_low = sum(values < bin_min)
  if N_low > 0:
    if N_low == 0:
      print(f'{N_low} {name} ({N_low / N:.2%})'
            ' fell below the histogram binning.')
    else:
      print(f'{N_low} {name}s ({N_low / N:.2%})'
            ' fell below the histogram binning.')

  N_high = sum(bin_max < values)
  if N_high > 0:
    if N_high == 0:
      print(f'{N_high} {name} ({N_high / N:.2%})'
            ' fell above the histogram binning.')
    else:
      print(f'{N_high} {name}s ({N_high / N:.2%})'
            ' fell above the histogram binning.')

# Configure binning.  Any null arguments are automatically configured to match
# the behavior of the `values1` and `values2` arguments.  Additionally if the
# difference `bin_max - bin_min` does not divide `bin_delta` then `bin_min` and
# `bin_max` are tweaked to respectively smaller and larger values as needed.
# @param bin_min Lower threshold
# @param bin_max Upper threshold
# @param bin_delta Bin width
# @param values1 An array of values
# @param values1 An auxiliary array of values
# @return List of updated bin_min, bin_max, and bin_delta values
def configure_bins(bin_min, bin_max, bin_delta,
                   values1, values2=None):
  if values2 is None:
    # Adapt bin configuration to `values1`
    if bin_min is None:
      bin_min = min(values1)
    if bin_max is None:
      bin_max = max(values1)
  else:
    # Adapt bin configuration to `values1` and `values2`
    if bin_min is None:
      bin_min = min(min(values1), min(values2))
    if bin_max is None:
      bin_max = max(max(values1), max(values2))

  if bin_delta is None:
    bin_delta = (bin_max - bin_min) / 25

  # Tweak bin configuration so that `bin_delta`
  # evenly divides `bin_max - bin_min`
  N = (bin_max - bin_min) / bin_delta
  excess = N - math.floor(N)
  if excess > 1e-15:
    bin_min = bin_min - 0.5 * bin_delta * excess
    bin_max = bin_max + 0.5 * bin_delta * excess

  return [bin_min, bin_max, bin_delta]

# Compute bin plotting.
# @param breaks Bin edges
# @return List of plotting indices and positions
def configure_bin_plotting(breaks):
  B = len(breaks) - 1
  idxs = [ idx for idx in range(B) for r in range(2) ]
  xs = [ breaks[idx + delta] for idx in range(B)
         for delta in [0, 1] ]
  return [idxs, xs]

################################################################################
# Data Visualizations
################################################################################

# Plot line histogram.
# @ax Matplotlib axis object
# @param values Values that comprise the histogram
# @param bin_min Lower threshold
# @param bin_max Upper threshold
# @param bin_delta Bin width
# @param breaks Full binning; supercedes bin_min, bin_max, and bin_delta
#               when not None
# @param prob Boolean determining whether bin contents should be normalized so
#             that the histogram approximates a probability density function;
#             defaults to FALSE
# @param col Color of histogram; defaults to "black"
# @param add Boolean determining whether to add histogram outline to existing
#            plot or to create new axes; defaults to False
# @param xlabel Label for x-axis; defaults to empty string.
# @param title Plot title; defaults to empty string.
def plot_line_hist(ax, values,
                   bin_min=None, bin_max=None, bin_delta=None,
                   breaks=None, prob=False,
                   col="black", add=False,
                   xlabel="", title=""):
  # Remove any nan values
  values = numpy.array(values)
  values = values[~numpy.isnan(values)]

  if breaks is None:
    # Construct binning configuration
    bin_min, bin_max, bin_delta = configure_bins(bin_min, bin_max,
                                                 bin_delta, values)

    # Construct bins
    breaks = numpy.arange(bin_min, bin_max, bin_delta)
  else:
    if bin_min is not None:
      print('Argument `bin_min` is being superceded '
            'by argument `breaks`')
    if bin_max is not None:
       print('Argument `bin_max` is being superceded '
            'by argument `breaks`')
    if bin_delta is not None:
      print('Argument `bin_delta` is being superceded '
            'by argument `breaks`')

    if not numpy.all(numpy.diff(breaks) > 0):
      raise ValueError('The argument `breaks` does '
                       'not define a valid binning.')

    bin_min = min(breaks)
    bin_max = max(breaks)
    bin_delta = numpy.diff(breaks)

  plot_idxs, plot_xs = configure_bin_plotting(breaks)

  # Check bin containment
  check_bin_containment(bin_min, bin_max, values)

  # Compute bin contents
  counts = numpy.histogram(values, bins=breaks)[0]

  ylabel = "Counts"
  if prob:
    ylabel = "Empirical Bin Probability / Bin Width"
    counts = counts / (bin_delta * sum(counts))

  # Plot
  ax.plot(plot_xs, counts[plot_idxs], color=col)
  
  if not add:
    ax.set_title(title)
    ax.set_xlim([bin_min, bin_max])
    ax.set_xlabel(xlabel)
    ax.set_ylim([0, 1.1 * max(counts)])
    ax.set_ylabel(ylabel)

# Plot the overlay of two line histograms.
# @ax Matplotlib axis object
# @param values1 Values that comprise the first histogram
# @param values2 Values that comprise the second histogram
# @param bin_min Lower threshold
# @param bin_max Upper threshold
# @param bin_delta Bin width
# @param breaks Full binning; supercedes bin_min, bin_max, and bin_delta
#               when not None
# @param prob Boolean determining whether bin contents should be normalized so
#             that the histogram approximates a probability density function;
#             defaults to FALSE
# @param xlabel Label for x-axis; defaults to empty string.
# @param title Plot title; defaults to empty string.
# @param col1 Color of first histogram; defaults to "black"
# @param col2 Color of second histogram; defaults to c_mid_teal
def plot_line_hists(ax, values1, values2,
                    bin_min=None, bin_max=None, bin_delta=None,
                    breaks=None, prob=False,
                    xlabel="", title="",
                    col1="black", col2=mid_teal):
  # Remove any nan values
  values1 = numpy.array(values1)
  values1 = values1[~numpy.isnan(values1)]

  values2 = numpy.array(values2)
  values2 = values2[~numpy.isnan(values2)]

  if breaks is None:
    # Construct binning configuration
    bin_min, bin_max, bin_delta = configure_bins(bin_min, bin_max,
                                                 bin_delta,
                                                 values1, values2)

    # Construct bins
    breaks = numpy.arange(bin_min, bin_max, bin_delta)
  else:
    if bin_min is not None:
      print('Argument `bin_min` is being superceded '
            'by argument `breaks`')
    if bin_max is not None:
       print('Argument `bin_max` is being superceded '
            'by argument `breaks`')
    if bin_delta is not None:
      print('Argument `bin_delta` is being superceded '
            'by argument `breaks`')

    if not numpy.all(numpy.diff(breaks) > 0):
      raise ValueError('The argument `breaks` does '
                       'not define a valid binning.')

    bin_min = min(breaks)
    bin_max = max(breaks)
    bin_delta = numpy.diff(breaks)

  plot_idxs, plot_xs = configure_bin_plotting(breaks)

  # Check bin containment
  check_bin_containment(bin_min, bin_max, values1)
  check_bin_containment(bin_min, bin_max, values2)

  # Compute bin contents
  counts1 = numpy.histogram(values1, bins=breaks)[0]
  counts2 = numpy.histogram(values2, bins=breaks)[0]

  ylabel = "Counts"
  if prob:
    ylabel = "Empirical Bin Probability / Bin Width"
    counts1 = counts1 / (bin_delta * sum(counts1))
    counts2 = counts2 / (bin_delta * sum(counts2))

  # Plot
  ymax = 1.1 * max(max(counts1), max(counts2))

  ax.plot(plot_xs, counts1[plot_idxs], color = col1)
  ax.plot(plot_xs, counts2[plot_idxs], color = col2)

  ax.set_title(title)
  ax.set_xlim([bin_min, bin_max])
  ax.set_xlabel(xlabel)
  ax.set_ylim([0, ymax])
  ax.set_ylabel(ylabel)
  ax.get_yaxis().set_visible(False)

################################################################################
# Pushforward Visualizations
################################################################################

# Overlay nested quantile intervals to visualize an ensemble of histograms.
# Individual quantiles are estimated as the average of the empirical quantiles
# across each Markov chain, a consistent quantile estimator for Markov chain
# Monte Carlo.
# @ax Matplotlib axis object
# @param samples A named list of two-dimensional arrays for
#                each expectand.  The first dimension of each element
#                indexes the Markov chains and the second dimension
#                indexes the sequential states within each Markov chain.
# @param val_name_prefix Prefix for the relevant variable names
# @param bin_min Lower threshold
# @param bin_max Upper threshold
# @param bin_delta Bin width
# @param breaks Full binning; supercedes bin_min, bin_max, and bin_delta
#               when not None
# @param baseline_values Baseline values for constructing a baseline histogram;
#                        defaults to None
# @param baseline_color Color for plotting baseline value; defaults to "black"
# @param xlabel Label for x-axis; defaults to empty string
# @param display_ylim Plot limits for y-axis; defaults to None
# @param title Plot title; defaults to empty string
def plot_hist_quantiles(ax, samples, val_name_prefix,
                        bin_min=None, bin_max=None, bin_delta=None,
                        breaks=None,
                        baseline_values=None, baseline_color="black",
                        xlabel="", display_ylim=None, title=""):
  # Construct relevant variable names and format corresponding values.
  # Order of the variables does not affect the shape of the histogram.
  names = [ key for key in samples.keys()
            if re.match('^' + val_name_prefix + '\[', key) ]
  collapsed_values = numpy.hstack([ samples[name].flatten()
                                    for name in names ])

  if breaks is None:
    # Construct binning configuration
    if baseline_values is None:
      [bin_min, bin_max, bin_delta] = configure_bins(bin_min, bin_max,
                                                     bin_delta,
                                                     collapsed_values)
    else:
      [bin_min, bin_max, bin_delta] = configure_bins(bin_min, bin_max,
                                                     bin_delta,
                                                     collapsed_values,
                                                     baseline_values)

    # Construct bins
    breaks = numpy.arange(bin_min, bin_max, bin_delta)
  else:
    if bin_min is not None:
      print('Argument `bin_min` is being superceded '
            'by argument `breaks`')
    if bin_max is not None:
       print('Argument `bin_max` is being superceded '
            'by argument `breaks`')
    if bin_delta is not None:
      print('Argument `bin_delta` is being superceded '
            'by argument `breaks`')

    if not numpy.all(numpy.diff(breaks) > 0):
      raise ValueError('The argument `breaks` does '
                       'not define a valid binning.')

    bin_min = min(breaks)
    bin_max = max(breaks)
    bin_delta = numpy.diff(breaks)

  plot_idxs, plot_xs = configure_bin_plotting(breaks)

  # Check bin containment
  check_bin_containment(bin_min, bin_max, collapsed_values,
                        "predictive value")
  if baseline_values is not None:
    check_bin_containment(bin_min, bin_max, baseline_values,
                          "observed value")

  # Construct quantiles for bin contents
  B = len(breaks) - 1
  counts = [numpy.nan] * B

  def bin_count(xs, b_low, b_high):
    return sum([ 1 for x in xs if b_low <= x and x < b_high ] )

  bin_counters = {}
  for b in range(B):
    bin_counters[b] = partial(bin_count,
                              b_low=breaks[b],
                              b_high=breaks[b + 1])
    if baseline_values is not None:
      counts[b] = bin_counters[b](baseline_values)

  bin_count_samples = \
    util.eval_expectand_pushforwards(samples,
                                     bin_counters,
                                     {'xs': numpy.array(names)})

  probs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
  quantiles = [ util.ensemble_mcmc_quantile_est(bin_count_samples[b],
                                                probs)
                for b in range(B) ]

  plot_quantiles = [ quantiles[idx] for idx in plot_idxs ]

  # Plot
  if display_ylim is None:
    if baseline_values is None:
      display_ylim = [0, max([ q[8] for q in quantiles ])]
    else:
      baseline_counts = numpy.histogram(baseline_values, bins=breaks)[0]
      display_ylim = [0, max(max([ q[8] for q in quantiles ]),
                             max(baseline_counts)) ]

  for b in range(B):
    idx1 = 2 * b
    idx2 = 2 * b + 1
    w = plot_xs[idx2] - plot_xs[idx1]

    h = plot_quantiles[idx1][8] - plot_quantiles[idx1][0]
    rect1 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][0]),
                           w, h, facecolor=light)
    ax.add_patch(rect1)

    h = plot_quantiles[idx1][7] - plot_quantiles[idx1][1]
    rect2 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][1]),
                           w, h, facecolor=light_highlight)
    ax.add_patch(rect2)

    h = plot_quantiles[idx1][6] - plot_quantiles[idx1][2]
    rect3 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][2]),
                           w, h, facecolor=mid)
    ax.add_patch(rect3)

    h = plot_quantiles[idx1][5] - plot_quantiles[idx1][3]
    rect4 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][3]),
                           w, h, facecolor=mid_highlight)
    ax.add_patch(rect4)

    ax.plot([ plot_xs[idx1], plot_xs[idx2] ],
            [ plot_quantiles[idx1][4], plot_quantiles[idx1][4] ],
            linewidth=1, color=dark)

  if baseline_values is not None:
    baseline_counts = numpy.histogram(baseline_values, bins=breaks)[0]
    plot_counts = [ baseline_counts[idx] for idx in plot_idxs ]
    ax.plot(plot_xs, plot_counts, color="white", linewidth=4)
    ax.plot(plot_xs, plot_counts, color=baseline_color, linewidth=2)

  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_xlim([bin_min, bin_max])
  ax.set_ylabel("Counts")
  ax.set_ylim(display_ylim)

# Overlay disconnected nested quantile intervals to visualize an ensemble of
# one-dimensional pushforward distributions.
# Individual quantiles are estimated as the average of the empirical quantiles
# across each Markov chain, a consistent quantile estimator for Markov chain
# Monte Carlo.
# @ax Matplotlib axis object
# @param samples A named list of two-dimensional arrays for
#                each expectand.  The first dimension of each element
#                indexes the Markov chains and the second dimension
#                indexes the sequential states within each Markov chain.
# @param names List of relevant variable names
# @param baseline_values Baseline values; defaults to None
# @param baseline_color Color for plotting baseline value; defaults to "black"
# @params residual Boolean value indicating whether to overlay quantiles and
#                  baseline values or plot their differences
# @param xlabel Label for x-axis; defaults to empty string
# @param xticklabels Labels for x-axis tics; defaults to None
# @param ylabel Label for y-axis; defaults to None
# @param display_ylim Plot limits for y-axis; defaults to None
# @param title Plot title; defaults to empty string
def plot_disc_pushforward_quantiles(ax, samples, names,
                                    baseline_values=None,
                                    baseline_color="black",
                                    residual=False,
                                    xlabel="", xticklabels=None,
                                    ylabel="", display_ylim=None,
                                    title=""):
  # Check that baseline values are well-defined
  if baseline_values is not None:
    if len(baseline_values) != len(names):
      print('The list of baseline values has the wrong '
            'dimension.  Baselines will not be plotted.')
      baseline_values = None

  # Check that names are in samples
  names = check_expectand_names(names, samples)

  # Construct bins
  N = len(names)
  breaks = [ n + 0.5 for n in range(N + 1) ]
  plot_idxs, plot_xs = configure_bin_plotting(breaks)

  # Construct marginal quantiles
  probs = [10, 20, 30, 40, 50, 60, 70, 80, 90]

  if baseline_values is not None and residual:
    def calc(n):
      return util.ensemble_mcmc_quantile_est(samples[names[n]] - \
                                             baseline_values[n],
                                             probs)
    quantiles = [ calc(n) for n in range(N) ]
  else:
    def calc(n):
      return util.ensemble_mcmc_quantile_est(samples[names[n]],
                                             probs)
    quantiles = [ calc(n) for n in range(N) ]

  plot_quantiles = [ quantiles[idx] for idx in plot_idxs ]

  # Plot
  if display_ylim is None:
    if baseline_values is None:
      display_ylim = [ min([ q[0] for q in quantiles ]),
                       max([ q[8] for q in quantiles ]) ]
    else:
      if residual:
        display_ylim = [ min([0] + [ q[0] for q in quantiles ]),
                         max([0] + [ q[8] for q in quantiles ]) ]
      else:
        display_ylim = [ min(min([ q[0] for q in quantiles ]),
                             min(baseline_values)),
                         max(max([ q[8] for q in quantiles ]),
                             max(baseline_values)) ]
    delta = 0.05 * (display_ylim[1] - display_ylim[0])
    display_ylim[0] -= delta
    display_ylim[1] += delta

  if ylabel is None:
    if baseline_values is not None or not residual:
      ylabel = "Marginal Quantiles"
    else:
      ylabel = "Marginal Quantiles - Baselines"

  if xticklabels is not None:
    if len(xticklabels) == N:
      ax.set_xticks([ n + 1 for n in range(N) ])
      ax.set_xticklabels(xticklabels)
    else:
      print('The list of x tick labels has the wrong dimension '
            'and baselines will not be plotted.')
  for n in range(N):
    idx1 = 2 * n
    idx2 = 2 * n + 1
    w = plot_xs[idx2] - plot_xs[idx1]

    h = plot_quantiles[idx1][8] - plot_quantiles[idx1][0]
    rect1 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][0]),
                           w, h, facecolor=light)
    ax.add_patch(rect1)

    h = plot_quantiles[idx1][7] - plot_quantiles[idx1][1]
    rect2 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][1]),
                           w, h, facecolor=light_highlight)
    ax.add_patch(rect2)

    h = plot_quantiles[idx1][6] - plot_quantiles[idx1][2]
    rect3 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][2]),
                           w, h, facecolor=mid)
    ax.add_patch(rect3)

    h = plot_quantiles[idx1][5] - plot_quantiles[idx1][3]
    rect4 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][3]),
                           w, h, facecolor=mid_highlight)
    ax.add_patch(rect4)

    ax.plot([ plot_xs[idx1], plot_xs[idx2] ],
            [ plot_quantiles[idx1][4], plot_quantiles[idx1][4] ],
            linewidth=1, color=dark)

  if baseline_values is not None:
    if residual:
      ax.axhline(y=0, linewidth=2, linestyle="dashed", color='#DDDDDD')
    else:
      for n in range(N):
        idx1 = 2 * n
        idx2 = 2 * n + 1
        ax.plot([plot_xs[idx1], plot_xs[idx2]],
                [baseline_values[n], baseline_values[n]],
                color="white", linewidth=4)
        ax.plot([plot_xs[idx1], plot_xs[idx2]],
                [baseline_values[n], baseline_values[n]],
                color=baseline_color, linewidth=2)

  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_xlim([0.5, N + 0.5])
  ax.set_ylabel(ylabel)
  ax.set_ylim(display_ylim)

# Overlay connected nested quantile intervals to visualize an ensemble of
# one-dimensional pushforward distributions.
# Individual quantiles are estimated as the average of the empirical quantiles
# across each Markov chain, a consistent quantile estimator for Markov chain
# Monte Carlo.
# @ax Matplotlib axis object
# @param samples A named list of two-dimensional arrays for
#                each expectand.  The first dimension of each element
#                indexes the Markov chains and the second dimension
#                indexes the sequential states within each Markov chain.
# @param names List of relevant variable names
# @param plot_xs One-dimensional array of x-axis values
#                associated with each variable.
# @param baseline_values Baseline values; defaults to None
# @param baseline_color Color for plotting baseline value; defaults to "black"
# @params residual Boolean value indicating whether to overlay quantiles and
#                  baseline values or plot their differences
# @param xlabel Label for x-axis; defaults to empty string
# @param display_xlim Plot limits for x-axis; defaults to None
# @param ylabel Label for y-axis; defaults to empty string
# @param display_ylim Plot limits for y-axis; defaults to None
# @param title Plot title; defaults to empty string
def plot_conn_pushforward_quantiles(ax, samples, names, plot_xs,
                                    baseline_values=None,
                                    baseline_color="black",
                                    residual=None,
                                    xlabel="", display_xlim=None,
                                    ylabel="", display_ylim=None,
                                    title=""):
  # Check dimensions
  check_dimensions(plot_xs, 'plot_xs', names, 'names')

  # Check that baseline values are well-defined
  if baseline_values is not None:
    if len(baseline_values) != len(names):
      print('The list of baseline values has the wrong '
            'dimension.  Baselines will not be plotted.')
      baseline_values = None

  # Check that names are in samples
  names = check_expectand_names(names, samples)

  # Construct quantiles for bin contents
  N = len(names)
  probs = [10, 20, 30, 40, 50, 60, 70, 80, 90]

  if baseline_values is not None and residual:
    def calc(n):
      return util.ensemble_mcmc_quantile_est(samples[names[n]] - \
                                             baseline_values[n],
                                             probs)
    plot_quantiles = [ calc(n) for n in range(N) ]
  else:
    def calc(n):
      return util.ensemble_mcmc_quantile_est(samples[names[n]],
                                             probs)
    plot_quantiles = [ calc(n) for n in range(N) ]

  # Plot
  if display_xlim is None:
    display_xlim = [ min(plot_xs), max(plot_xs) ]

  if display_ylim is None:
    if baseline_values is None:
      display_ylim = [ min([ q[0] for q in plot_quantiles ]),
                       max([ q[8] for q in plot_quantiles ]) ]
    else:
      if residual:
        display_ylim = [ min([0] + [ q[0] for q in plot_quantiles ]),
                         max([0] + [ q[8] for q in plot_quantiles ]) ]
      else:
        display_ylim = [ min(min([ q[0] for q in plot_quantiles ]),
                             min(baseline_values)),
                         max(max([ q[8] for q in plot_quantiles ]),
                             max(baseline_values)) ]
    delta = 0.05 * (display_ylim[1] - display_ylim[0])
    display_ylim[0] -= delta
    display_ylim[1] += delta

  if ylabel is None:
    if baseline_values is not None or not residual:
      ylabel = "Marginal Quantiles"
    else:
      ylabel = "Marginal Quantiles - Baselines"

  ax.fill_between(plot_xs,
                  [q[0] for q in plot_quantiles],
                  [q[8] for q in plot_quantiles],
                  facecolor=light, color=light)
  ax.fill_between(plot_xs,
                  [q[1] for q in plot_quantiles],
                  [q[7] for q in plot_quantiles],
                  facecolor=light_highlight, color=light_highlight)
  ax.fill_between(plot_xs,
                  [q[2] for q in plot_quantiles],
                  [q[6] for q in plot_quantiles],
                  facecolor=mid, color=mid)
  ax.fill_between(plot_xs,
                  [q[3] for q in plot_quantiles],
                  [q[5] for q in plot_quantiles],
                  facecolor=mid_highlight, color=mid_highlight)
  ax.plot(plot_xs, [q[4] for q in plot_quantiles], color=dark)

  if baseline_values is not None:
    if residual:
      ax.axhline(y=0, linewidth=2, linestyle="dashed", color='#DDDDDD')
    else:
      ax.plot(plot_xs, baseline_values, color="white", linewidth=4)
      ax.plot(plot_xs, baseline_values, color=baseline_color, linewidth=2)

  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_xlim(display_xlim)
  ax.set_ylabel(ylabel)
  ax.set_ylim(display_ylim)

# Overlay an ensemble of function realizations to visualize a probability
# distribution over a space of one-dimensional functions.
# @ax Matplotlib axis object
# @param samples A named list of two-dimensional arrays for
#                each expectand.  The first dimension of each element
#                indexes the Markov chains and the second dimension
#                indexes the sequential states within each Markov chain.
# @param names List of relevant variable names
# @param plot_xs One-dimensional array of x-axis values
#                associated with each variable.
# @param N_plots Number of realizations to plot
# @param baseline_values Baseline values; defaults to None
# @param baseline_color Color for plotting baseline value; defaults to "black"
# @params residual Boolean value indicating whether to overlay quantiles and
#                  baseline values or plot their differences
# @param xlab Label for x-axis; defaults to empty string
# @param display_xlim Plot limits for x-axis; defaults to None
# @param ylab Label for y-axis; defaults to None
# @param display_ylim Plot limits for y-axis; defaults to None
# @param main Plot title; defaults to empty string
def plot_realizations(ax, samples, names, plot_xs, N_plots=50,
                      baseline_values=None,
                      baseline_color="black",
                      residual=False,
                      xlabel="", display_xlim=None,
                      ylabel="", display_ylim=None,
                      title=""):
  # Check dimensions
  check_dimensions(plot_xs, 'plot_xs', names, 'names')

  # Check that baseline values are well-defined
  if baseline_values is not None:
    if len(baseline_values) != len(names):
      print('The list of baseline values has the wrong '
            'dimension.  Baselines will not be plotted.')
      baseline_values = None

  # Check that names are in samples
  names = check_expectand_names(names, samples)

  # Extract function values
  fs = [ samples[name].flatten() for name in names ]

  N = len(fs)
  I = len(fs[0])

  if baseline_values is not None and residual:
    for i in range(I):
      for n in range(N):
        fs[n][i] -= baseline_values[n]

  # Configure ensemble of function realizations
  J = min(N_plots, I)
  plot_idx = [ int(I / J) * n for n in range(J) ]

  colors = [dark, dark_highlight, dark, mid_highlight,
            mid, light_highlight, light]
  cmap = LinearSegmentedColormap.from_list("reds", colors, N=J)

  # Plot
  if display_xlim is None:
    display_xlims = [ min(plot_xs), max(plot_xs) ]

  if display_ylim is None:
    if baseline_values is None:
      display_ylim = [ min([ min(f) for f in fs ]),
                       max([ max(f) for f in fs ]) ]
    else:
      if residual:
        display_ylim = [ min([0] + [ min(f) for f in fs ]),
                         max([0] + [ max(f) for f in fs ]) ]
      else:
        display_ylim = [ min(min([ min(f) for f in fs ]),
                             min(baseline_values)),
                         max(max([ max(f) for f in fs ]),
                             max(baseline_values)) ]
    delta = 0.05 * (display_ylim[1] - display_ylim[0])
    display_ylim[0] -= delta
    display_ylim[1] += delta

  if ylabel is None:
    if baseline_values is not None or not residual:
      ylabel = "Function Outputs"
    else:
      ylabel = "Function Outputs - Baselines"

  for j in range(J):
    r_fs = [ f[plot_idx[j]] for f in fs ]
    ax.plot(plot_xs, r_fs, color = cmap(j), linewidth=3)

  if baseline_values is not None:
    if residual:
      ax.axhline(y=0, linewidth=2, linestyle="dashed", color='#DDDDDD')
    else:
      ax.plot(plot_xs, baseline_values, color="white", linewidth=4)
      ax.plot(plot_xs, baseline_values, color=baseline_color, linewidth=2)

  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_xlim(display_xlim)
  ax.set_ylabel(ylabel)
  ax.set_ylim(display_ylim)


# Overlay nested quantile intervals to visualize an ensemble of empirical means
# within the computed bins.
# Individual quantiles are estimated as the average of the empirical quantiles
# across each Markov chain, a consistent quantile estimator for Markov chain
# Monte Carlo.
# @ax Matplotlib axis object
# @param samples A named list of two-dimensional arrays for
#                each expectand.  The first dimension of each element
#                indexes the Markov chains and the second dimension
#                indexes the sequential states within each Markov chain.
# @param names List of relevant variable names
# @param obs_xs One-dimensional array of observed x-values on which to condition
# @param bin_min Lower threshold for conditioning
# @param bin_max Upper threshold for conditioning
# @param bin_delta Bin width for conditioning
# @param baseline_values Baseline values; defaults to None
# @param baseline_color Color for plotting baseline value; defaults to "black"
# @params residual Boolean value indicating whether to overlay quantiles and
#                  baseline values or plot their differences
# @param xlabel Label for x-axis; defaults to empty string
# @param ylabel Label for y-axis; defaults to None
# @param display_ylim Plot limits for y-axis; defaults to None
# @param title Plot title; defaults to empty string
def plot_conditional_mean_quantiles(ax, samples, names, obs_xs,
                                    bin_min=None, bin_max=None, bin_delta=None,
                                    baseline_values=None,
                                    baseline_color="black",
                                    residual=False,
                                    xlabel="",
                                    ylabel=None, display_ylim=None,
                                    title=""):
  # Check dimensions
  check_dimensions(obs_xs, 'obs_xs', names, 'names')

  # Check that baseline values are well-defined
  if baseline_values is not None:
    if len(baseline_values) != len(names):
      print('The list of baseline values has the wrong '
            'dimension.  Baselines will not be plotted.')
      baseline_values = None

  # Check that names are in samples
  names = check_expectand_names(names, samples)

  # Construct binning configuration
  bin_min, bin_max, bin_delta = configure_bins(bin_min, bin_max,
                                               bin_delta, obs_xs)

  # Construct bins
  breaks = numpy.arange(bin_min, bin_max, bin_delta)
  plot_idxs, plot_xs = configure_bin_plotting(breaks)

  # Check bin containment
  check_bin_containment(bin_min, bin_max, obs_xs,
                        "conditioning value")

  # Construct quantiles for predictive bin contents
  B = len(breaks) - 1

  nonempty_bins = [ b for b in range(B)
                    if sum([ 1 for x in obs_xs
                             if breaks[b] <= x and x < breaks[b + 1] ]) > 0]

  baseline_cond_means = [numpy.nan] * B

  def cond_mean(ys, xs, b_low, b_high):
    bin_idxs = [ n for n, x in enumerate(xs)
                 if b_low <= x and x < b_high ]
    if len(bin_idxs):
      return numpy.mean([ ys[n] for n in bin_idxs ])
    else:
      return 0

  def cond_mean_residual(ys, xs, b_low, b_high, baseline):
    return cond_mean(ys, xs, b_low, b_high) - baseline

  if baseline_values is not None:
    for b in range(B):
      baseline_cond_means[b] = cond_mean(baseline_values, obs_xs,
                                         breaks[b], breaks[b + 1])

  if baseline_values is None or not residual:
    expectands = {}
    for b in range(B):
      expectands[b] = partial(cond_mean,
                              xs=obs_xs,
                              b_low=breaks[b],
                              b_high=breaks[b + 1])
  else:
    expectands = {}
    for b in range(B):
      baseline = baseline_cond_means[b]
      expectands[b] = partial(cond_mean_residual,
                              xs=obs_xs,
                              b_low=breaks[b],
                              b_high=breaks[b + 1],
                              baseline=baseline)

  cond_mean_samples = \
    util.eval_expectand_pushforwards(samples,
                                     expectands,
                                     {'ys': numpy.array(names)})

  probs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
  mean_quantiles = [ util.ensemble_mcmc_quantile_est(cond_mean_samples[b],
                                                     probs)
                     for b in range(B)                                    ]

  plot_quantiles = [ mean_quantiles[idx] for idx in plot_idxs ]

  # Plot
  if display_ylim is None:
    if baseline_values is None:
      display_ylim = [ min([ mean_quantiles[b][0]
                             for b in nonempty_bins ]),
                       max([ mean_quantiles[b][8]
                             for b in nonempty_bins ]) ]
    else:
      if residual:
        display_ylim = [ min([0] + [ mean_quantiles[b][0]
                                     for b in nonempty_bins ]),
                         max([0] + [ mean_quantiles[b][8]
                                     for b in nonempty_bins ]) ]
      else:
        display_ylim = [ min(min([ mean_quantiles[b][0]
                                   for b in nonempty_bins ]),
                             min([ baseline_cond_means[b]
                                   for b in nonempty_bins ])),
                         max(max([ mean_quantiles[b][8]
                                   for b in nonempty_bins ]),
                             max([ baseline_cond_means[b]
                                   for b in nonempty_bins ])) ]
    delta = 0.05 * (display_ylim[1] - display_ylim[0])
    display_ylim[0] -= delta
    display_ylim[1] += delta

  if ylabel is None:
    if baseline_values is not None or not residual:
      ylabel = "Marginal Quantiles of Conditional Means"
    else:
      ylabel = "Marginal Quantiles of Conditional Means - Baselines"

  for b in range(B):
    idx1 = 2 * b
    idx2 = 2 * b + 1
    w = plot_xs[idx2] - plot_xs[idx1]

    if b in nonempty_bins:
      h = plot_quantiles[idx1][8] - plot_quantiles[idx1][0]
      rect1 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][0]),
                             w, h, facecolor=light)
      ax.add_patch(rect1)

      h = plot_quantiles[idx1][7] - plot_quantiles[idx1][1]
      rect2 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][1]),
                             w, h, facecolor=light_highlight)
      ax.add_patch(rect2)

      h = plot_quantiles[idx1][6] - plot_quantiles[idx1][2]
      rect3 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][2]),
                             w, h, facecolor=mid)
      ax.add_patch(rect3)

      h = plot_quantiles[idx1][5] - plot_quantiles[idx1][3]
      rect4 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][3]),
                             w, h, facecolor=mid_highlight)
      ax.add_patch(rect4)

      ax.plot([ plot_xs[idx1], plot_xs[idx2] ],
            [ plot_quantiles[idx1][4], plot_quantiles[idx1][4] ],
            linewidth=1, color=dark)
    else:
      h = display_ylim[1] - display_ylim[0]
      rect = plot.Rectangle((plot_xs[idx1], display_ylim[0]),
                             w, h, facecolor="#EEEEEE")
      ax.add_patch(rect)

  if baseline_values is not None:
    if residual:
      ax.axhline(y=0, linewidth=2, linestyle="dashed", color='#DDDDDD')
    else:
      for b in range(B):
        if b in nonempty_bins:
          idx1 = 2 * b
          idx2 = 2 * b + 1
          ax.plot([ plot_xs[idx1], plot_xs[idx2] ],
                  [ baseline_cond_means[b], baseline_cond_means[b] ],
                  color="white", linewidth=4)
          ax.plot([ plot_xs[idx1], plot_xs[idx2] ],
                  [ baseline_cond_means[b], baseline_cond_means[b] ],
                  color=baseline_color, linewidth=2)

  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_xlim([bin_min, bin_max])
  ax.set_ylabel(ylabel)
  ax.set_ylim(display_ylim)

# Overlay nested quantile intervals to visualize an ensemble of empirical
# medians within the computed bins.
# Individual quantiles are estimated as the average of the empirical quantiles
# across each Markov chain, a consistent quantile estimator for Markov chain
# Monte Carlo.
# @ax Matplotlib axis object
# @param samples A named list of two-dimensional arrays for
#                each expectand.  The first dimension of each element
#                indexes the Markov chains and the second dimension
#                indexes the sequential states within each Markov chain.
# @param names List of relevant variable names
# @param obs_xs One-dimensional array of observed x-values on which to condition
# @param bin_min Lower threshold for conditioning
# @param bin_max Upper threshold for conditioning
# @param bin_delta Bin width for conditioning
# @param baseline_values Baseline values; defaults to None
# @param baseline_color Color for plotting baseline value; defaults to "black"
# @params residual Boolean value indicating whether to overlay quantiles and
#                  baseline values or plot their differences
# @param xlabel Label for x-axis; defaults to empty string
# @param ylabel Label for y-axis; defaults to None
# @param display_ylim Plot limits for y-axis; defaults to None
# @param title Plot title; defaults to empty string
def plot_conditional_median_quantiles(ax, samples, names, obs_xs,
                                      bin_min=None, bin_max=None,
                                      bin_delta=None,
                                      baseline_values=None,
                                      baseline_color="black",
                                      residual=False,
                                      xlabel="",
                                      ylabel=None, display_ylim=None,
                                      title=""):
  # Check dimensions
  check_dimensions(obs_xs, 'obs_xs', names, 'names')

  # Check that baseline values are well-defined
  if baseline_values is not None:
    if len(baseline_values) != len(names):
      print('The list of baseline values has the wrong '
            'dimension.  Baselines will not be plotted.')
      baseline_values = None

  # Check that names are in samples
  names = check_expectand_names(names, samples)

  # Construct binning configuration
  bin_min, bin_max, bin_delta = configure_bins(bin_min, bin_max,
                                               bin_delta, obs_xs)

  # Construct bins
  breaks = numpy.arange(bin_min, bin_max, bin_delta)
  plot_idxs, plot_xs = configure_bin_plotting(breaks)

  # Check bin containment
  check_bin_containment(bin_min, bin_max, obs_xs,
                        "conditioning value")

  # Construct quantiles for predictive bin contents
  B = len(breaks) - 1

  nonempty_bins = [ b for b in range(B)
                    if sum([ 1 for x in obs_xs
                             if breaks[b] <= x and x < breaks[b + 1] ]) > 0]

  baseline_cond_medians = [numpy.nan] * B

  def cond_median(ys, xs, b_low, b_high):
    bin_idxs = [ n for n, x in enumerate(xs)
                 if b_low <= x and x < b_high ]
    if len(bin_idxs):
      return numpy.median([ ys[n] for n in bin_idxs ])
    else:
      return 0

  def cond_median_residual(ys, xs, b_low, b_high, baseline):
    return cond_median(ys, xs, b_low, b_high) - baseline

  if baseline_values is not None:
    for b in range(B):
      baseline_cond_medians[b] = cond_median(baseline_values, obs_xs,
                                             breaks[b], breaks[b + 1])

  if baseline_values is None or not residual:
    expectands = {}
    for b in range(B):
      expectands[b] = partial(cond_median,
                              xs=obs_xs,
                              b_low=breaks[b],
                              b_high=breaks[b + 1])
  else:
    expectands = {}
    for b in range(B):
      baseline = baseline_cond_medians[b]
      expectands[b] = partial(cond_median_residual,
                              xs=obs_xs,
                              b_low=breaks[b],
                              b_high=breaks[b + 1],
                              baseline=baseline)

  cond_median_samples = \
    util.eval_expectand_pushforwards(samples,
                                     expectands,
                                     {'ys': numpy.array(names)})

  probs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
  median_quantiles = [ util.ensemble_mcmc_quantile_est(cond_median_samples[b],
                                                       probs)
                       for b in range(B)                                      ]

  plot_quantiles = [ median_quantiles[idx] for idx in plot_idxs ]

  # Plot
  if display_ylim is None:
    if baseline_values is None:
      display_ylim = [ min([ median_quantiles[b][0]
                             for b in nonempty_bins ]),
                       max([ median_quantiles[b][8]
                             for b in nonempty_bins ]) ]
    else:
      if residual:
        display_ylim = [ min([0] + [ median_quantiles[b][0]
                                     for b in nonempty_bins ]),
                         max([0] + [ median_quantiles[b][8]
                                     for b in nonempty_bins ]) ]
      else:
        display_ylim = [ min(min([ median_quantiles[b][0]
                                   for b in nonempty_bins ]),
                             min([ baseline_cond_medians[b]
                                   for b in nonempty_bins ])),
                         max(max([ median_quantiles[b][8]
                                   for b in nonempty_bins ]),
                             max([ baseline_cond_medians[b]
                                   for b in nonempty_bins ])) ]
    delta = 0.05 * (display_ylim[1] - display_ylim[0])
    display_ylim[0] -= delta
    display_ylim[1] += delta

  if ylabel is None:
    if baseline_values is not None or not residual:
      ylabel = "Marginal Quantiles of Conditional Medians"
    else:
      ylabel = "Marginal Quantiles of Conditional Medians - Baselines"

  for b in range(B):
    idx1 = 2 * b
    idx2 = 2 * b + 1
    w = plot_xs[idx2] - plot_xs[idx1]

    if b in nonempty_bins:
      h = plot_quantiles[idx1][8] - plot_quantiles[idx1][0]
      rect1 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][0]),
                             w, h, facecolor=light)
      ax.add_patch(rect1)

      h = plot_quantiles[idx1][7] - plot_quantiles[idx1][1]
      rect2 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][1]),
                             w, h, facecolor=light_highlight)
      ax.add_patch(rect2)

      h = plot_quantiles[idx1][6] - plot_quantiles[idx1][2]
      rect3 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][2]),
                             w, h, facecolor=mid)
      ax.add_patch(rect3)

      h = plot_quantiles[idx1][5] - plot_quantiles[idx1][3]
      rect4 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][3]),
                             w, h, facecolor=mid_highlight)
      ax.add_patch(rect4)

      ax.plot([ plot_xs[idx1], plot_xs[idx2] ],
            [ plot_quantiles[idx1][4], plot_quantiles[idx1][4] ],
            linewidth=1, color=dark)
    else:
      h = display_ylim[1] - display_ylim[0]
      rect = plot.Rectangle((plot_xs[idx1], display_ylim[0]),
                             w, h, facecolor="#EEEEEE")
      ax.add_patch(rect)

  if baseline_values is not None:
    if residual:
      ax.axhline(y=0, linewidth=2, linestyle="dashed", color='#DDDDDD')
    else:
      for b in range(B):
        if b in nonempty_bins:
          idx1 = 2 * b
          idx2 = 2 * b + 1
          ax.plot([ plot_xs[idx1], plot_xs[idx2] ],
                  [ baseline_cond_medians[b], baseline_cond_medians[b] ],
                  color="white", linewidth=4)
          ax.plot([ plot_xs[idx1], plot_xs[idx2] ],
                  [ baseline_cond_medians[b], baseline_cond_medians[b] ],
                  color=baseline_color, linewidth=2)

  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_xlim([bin_min, bin_max])
  ax.set_ylabel(ylabel)
  ax.set_ylim(display_ylim)
