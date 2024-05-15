################################################################################
#
# The code is copyright 2024 Michael Betancourt and licensed under the
# new BSD (3-clause) license:
#  https://opensource.org/licenses/BSD-3-Clause
#
# For more information see LINK.
#
################################################################################

# Load required libraries
import matplotlib
import matplotlib.pyplot as plot
from matplotlib.colors import LinearSegmentedColormap

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
    bin_min = bin_min - 0.5 * excess
    bin_max = bin_max + 0.5 * excess

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
# @param prob Boolean determining whether bin contents should be normalized so
#             that the histogram approximates a probability density function;
#             defaults to FALSE
# @param xlabel Label for x-axis; defaults to empty string.
# @param title Plot title; defaults to empty string.
def plot_line_hist(ax, values,
                   bin_min=None, bin_max=None, bin_delta=None,
                   prob=False, xlabel="", title=""):
  # Remove any nan values
  values = numpy.array(values)
  values = values[~numpy.isnan(values)]

  # Construct binning configuration
  bin_min, bin_max, bin_delta = configure_bins(bin_min, bin_max,
                                               bin_delta, values)

  # Construct bins
  breaks = numpy.arange(bin_min, bin_max, bin_delta)
  plot_idxs, plot_xs = configure_bin_plotting(breaks)

  # Check bin containment
  check_bin_containment(bin_min, bin_max, values)

  # Compute bin contents
  counts = numpy.histogram(values, bins=breaks)[0]

  ylabel = "Counts"
  if prob:
    ylabel = "Empirical Bin Probability / Bin Width"
    norm = delta * sum(counts)
    counts = [ c / norm for c in counts ]

  # Plot
  ax.plot(plot_xs, counts[plot_idxs], color="black")

  ax.set_title(title)
  ax.set_xlim([bin_min, bin_max])
  ax.set_xlabel(xlabel)
  ax.set_ylim([0, 1.1 * max(counts)])
  ax.set_ylabel(ylabel)
  ax.get_yaxis().set_visible(False)

# Plot the overlay of two line histograms.
# @ax Matplotlib axis object
# @param values1 Values that comprise the first histogram
# @param values2 Values that comprise the second histogram
# @param bin_min Lower threshold
# @param bin_max Upper threshold
# @param bin_delta Bin width
# @param prob Boolean determining whether bin contents should be normalized so
#             that the histogram approximates a probability density function;
#             defaults to FALSE
# @param xlabel Label for x-axis; defaults to empty string.
# @param title Plot title; defaults to empty string.
# @param col1 Color of first histogram; defaults to "black"
# @param col2 Color of second histogram; defaults to c_mid_teal
def plot_line_hists(ax, values1, values2,
                    bin_min=None, bin_max=None, bin_delta=None,
                    prob=False, xlabel="", title="",
                    col1="black", col2=mid_teal):
  # Remove any nan values
  values1 = numpy.array(values1)
  values1 = values1[~numpy.isnan(values1)]

  values2 = numpy.array(values2)
  values2 = values2[~numpy.isnan(values2)]

  # Construct binning configuration
  bin_min, bin_max, bin_delta = configure_bins(bin_min, bin_max, bin_delta,
                                               values1, values2)

  # Construct bins
  breaks = numpy.arange(bin_min, bin_max, bin_delta)
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
    counts1 = counts1 / (delta * sum(counts1))
    counts2 = counts2 / (delta * sum(counts1))

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
# @param baseline_values Baseline values for constructing a baseline histogram;
#                        defaults to None
# @param xlabel Label for x-axis; defaults to empty string
# @param display_ylim Plot limits for y-axis; defaults to None
# @param title Plot title; defaults to empty string
def plot_hist_quantiles(ax, samples, val_name_prefix,
                        bin_min=None, bin_max=None, bin_delta=None,
                        baseline_values=None,
                        xlabel="", display_ylim=None, title=""):
  # Construct relevant variable names and format corresponding values.
  # Order of the variables does not affect the shape of the histogram.
  names = [ key for key in samples.keys() if val_name_prefix + '[' in key ]
  collapsed_values = numpy.hstack([ samples[name].flatten()
                                    for name in names ])

  # Construct binning configuration
  if baseline_values is None:
    [bin_min, bin_max, bin_delta] = configure_bins(bin_min, bin_max, bin_delta,
                                                   collapsed_values)
  else:
    [bin_min, bin_max, bin_delta] = configure_bins(bin_min, bin_max, bin_delta,
                                                   collapsed_values,
                                                   baseline_values)

  # Construct bins
  breaks = numpy.arange(bin_min, bin_max, bin_delta)
  plot_idxs, plot_xs = configure_bin_plotting(breaks)

  # Check bin containment
  check_bin_containment(bin_min, bin_max, collapsed_values,
                        "predictive value")
  if baseline_values is not None:
    check_bin_containment(bin_min, bin_max, baseline_values,
                          "observed value")

  # Construct quantiles for bin contents
  B = len(breaks) - 1
  N = len(names)
  C = samples[names[0]].shape[0]
  S = samples[names[0]].shape[1]
  probs = [10, 20, 30, 40, 50, 60, 70, 80, 90]

  quantiles = numpy.full((B, 9), 0.0)

  for c in range(C):
    values = [ samples[name][c,:] for name in names ]
    counts = [ numpy.histogram([ v[s] for v in values ], bins=breaks)[0]
                               for s in range(S) ]
    quantiles += [ numpy.percentile([ c[b] for c in counts ], probs) / C
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
                           w, h, color=light)
    ax.add_patch(rect1)

    h = plot_quantiles[idx1][7] - plot_quantiles[idx1][1]
    rect2 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][1]),
                           w, h, color=light_highlight)
    ax.add_patch(rect2)

    h = plot_quantiles[idx1][6] - plot_quantiles[idx1][2]
    rect3 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][2]),
                           w, h, color=mid)
    ax.add_patch(rect3)

    h = plot_quantiles[idx1][5] - plot_quantiles[idx1][3]
    rect4 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][3]),
                           w, h, color=mid_highlight)
    ax.add_patch(rect4)

    h = plot_quantiles[idx1][4] - plot_quantiles[idx1][4]
    rect5 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][4]),
                           w, h, color=dark)
    ax.add_patch(rect5)

  if baseline_values is not None:
    baseline_counts = numpy.histogram(baseline_values, bins=breaks)[0]
    plot_counts = [ baseline_counts[idx] for idx in plot_idxs ]
    ax.plot(plot_xs, plot_counts, color="white", linewidth=4)
    ax.plot(plot_xs, plot_counts, color="black", linewidth=2)

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
# @params residual Boolean value indicating whether to overlay quantiles and
#                  baseline values or plot their differences
# @param xlabel Label for x-axis; defaults to empty string
# @param xticklabels Labels for x-axis tics; defaults to None
# @param ylabel Label for y-axis; defaults to None
# @param display_ylim Plot limits for y-axis; defaults to None
# @param title Plot title; defaults to empty string
def plot_disc_pushforward_quantiles(ax, samples, names,
                                    baseline_values=None,
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
  C = samples[names[0]].shape[0]
  probs = [10, 20, 30, 40, 50, 60, 70, 80, 90]

  quantiles = numpy.full((N, 9), 0.0)
  for c in range(C):
    values = [ samples[name][c,:] for name in names ]
    if baseline_values is not None and residual:
      quantiles += [ numpy.percentile(values[n] - baseline_values[n],
                                      probs) / C
                     for n in range(N) ]
    else:
      quantiles += [ numpy.percentile(values[n], probs) / C
                     for n in range(N) ]

  plot_quantiles = [ quantiles[idx] for idx in plot_idxs ]

  # Plot
  # Plot
  if display_ylim is None:
    if baseline_values is None:
      display_ylim = [ min([ q[0] for q in quantiles ]),
                       max([ q[8] for q in quantiles ]) ]
    else:
      if residual:
        display_ylim = [ min([ q[0] for q in quantiles ]),
                         max([ q[8] for q in quantiles ]) ]
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
                           w, h, color=light)
    ax.add_patch(rect1)

    h = plot_quantiles[idx1][7] - plot_quantiles[idx1][1]
    rect2 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][1]),
                           w, h, color=light_highlight)
    ax.add_patch(rect2)

    h = plot_quantiles[idx1][6] - plot_quantiles[idx1][2]
    rect3 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][2]),
                           w, h, color=mid)
    ax.add_patch(rect3)

    h = plot_quantiles[idx1][5] - plot_quantiles[idx1][3]
    rect4 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][3]),
                           w, h, color=mid_highlight)
    ax.add_patch(rect4)

    h = plot_quantiles[idx1][4] - plot_quantiles[idx1][4]
    rect5 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][4]),
                           w, h, color=dark)
    ax.add_patch(rect5)

  if baseline_values is not None:
    if residual:
      ax.axhline(y=0, linewidth=2, linestyle="dashed", color='#DDDDDD')
    else:
      for n in range(N):
        idx1 = 2 * n
        idx2 = 2 * n + 1
        ax.plot([plot_xs[idx1], plot_xs[idx2]],
                [baseline_values[n], baseline_values[n]],
                color = "white", linewidth=4)
        ax.plot([plot_xs[idx1], plot_xs[idx2]],
                [baseline_values[n], baseline_values[n]],
                color = "black", linewidth=2)

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
# @params residual Boolean value indicating whether to overlay quantiles and
#                  baseline values or plot their differences
# @param xlabel Label for x-axis; defaults to empty string
# @param display_xlim Plot limits for x-axis; defaults to None
# @param ylabel Label for y-axis; defaults to empty string
# @param display_ylim Plot limits for y-axis; defaults to None
# @param title Plot title; defaults to empty string
def plot_conn_pushforward_quantiles(ax, samples, names, plot_xs,
                                    baseline_values=None,
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
  C = samples[names[0]].shape[0]
  probs = [10, 20, 30, 40, 50, 60, 70, 80, 90]

  plot_quantiles = numpy.full((N, 9), 0.0)
  for c in range(C):
    values = [ samples[name][c,:] for name in names ]
    if baseline_values is not None and residual:
      plot_quantiles += [ numpy.percentile(values[n] - baseline_values[n],
                                           probs) / C
                          for n in range(N) ]
    else:
      plot_quantiles += [ numpy.percentile(values[n], probs) / C
                          for n in range(N) ]

  # Plot
  if display_xlim is None:
    display_xlim = [ min(plot_xs), max(plot_xs) ]

  if display_ylim is None:
    if baseline_values is None:
      display_ylim = [ min([ q[0] for q in plot_quantiles ]),
                       max([ q[8] for q in plot_quantiles ]) ]
    else:
      if residual:
        display_ylim = [ min([ q[0] for q in plot_quantiles ]),
                         max([ q[8] for q in plot_quantiles ]) ]
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
      ax.plot(plot_xs, baseline_values, color="black", linewidth=2)

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
# @params residual Boolean value indicating whether to overlay quantiles and
#                  baseline values or plot their differences
# @param xlab Label for x-axis; defaults to empty string
# @param display_xlim Plot limits for x-axis; defaults to None
# @param ylab Label for y-axis; defaults to None
# @param display_ylim Plot limits for y-axis; defaults to None
# @param main Plot title; defaults to empty string
def plot_realizations(ax, samples, names, plot_xs, N_plots=50,
                      baseline_values=None,
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
        display_ylim = [ min([ min(f) for f in fs ]),
                         max([ max(f) for f in fs ]) ]
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
      ax.plot(plot_xs, baseline_values, color="black", linewidth=2)

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
# @params residual Boolean value indicating whether to overlay quantiles and
#                  baseline values or plot their differences
# @param xlabel Label for x-axis; defaults to empty string
# @param ylabel Label for y-axis; defaults to None
# @param display_ylim Plot limits for y-axis; defaults to None
# @param title Plot title; defaults to empty string
def plot_conditional_mean_quantiles(ax, samples, names, obs_xs,
                                    bin_min=None, bin_max=None, bin_delta=None,
                                    baseline_values=None,
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
  N = len(names)
  C = samples[names[0]].shape[0]
  S = samples[names[0]].shape[1]
  probs = [10, 20, 30, 40, 50, 60, 70, 80, 90]

  obs_means = [numpy.nan] * B
  mean_quantiles = numpy.full((B, 9), 0.0)

  for b in range(B):
    bin_idx = [ n for n, obs in enumerate(obs_xs)
                if breaks[b] <= obs and obs < breaks[b + 1] ]
    if len(bin_idx) > 0:
      if baseline_values is not None:
        obs_means [b] = numpy.mean([ baseline_values[idx]
                                     for idx in bin_idx ] )

      for c in range(C):
        values = [ samples[name][c,:] for name in names ]

        if baseline_values is not None and residual:
          means = [ numpy.mean([ values[idx][s] for idx in bin_idx ])
                    - obs_means[b]
                    for s in range(S) ]
        else:
          means = [ numpy.mean([ values[idx][s] for idx in bin_idx ])
                    for s in range(S) ]

        mean_quantiles[b,:] += numpy.percentile(means, probs) / C
    else:
      mean_quantiles[b,:] = numpy.full((9), numpy.nan)

  plot_quantiles = [ mean_quantiles[idx] for idx in plot_idxs ]

  # Plot
  if display_ylim is None:
    if baseline_values is None:
      display_ylim = [ numpy.nanmin([ q[0] for q in mean_quantiles ]),
                       numpy.nanmax([ q[8] for q in mean_quantiles ]) ]
    else:
      if residual:
        display_ylim = [ numpy.nanmin([ q[0] for q in mean_quantiles ]),
                         numpy.nanmax([ q[8] for q in mean_quantiles ]) ]
      else:
        display_ylim = [ min(numpy.nanmin([ q[0] for q in mean_quantiles ]),
                             numpy.nanmin(obs_means)),
                         max(numpy.nanmax([ q[8] for q in mean_quantiles ]),
                             numpy.nanmax(obs_means)) ]
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

    bin_idx = [ n for n, obs in enumerate(obs_xs)
                if breaks[b] <= obs and obs < breaks[b + 1] ]
    if len(bin_idx) > 0:
      h = plot_quantiles[idx1][8] - plot_quantiles[idx1][0]
      rect1 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][0]),
                             w, h, color=light)
      ax.add_patch(rect1)

      h = plot_quantiles[idx1][7] - plot_quantiles[idx1][1]
      rect2 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][1]),
                             w, h, color=light_highlight)
      ax.add_patch(rect2)

      h = plot_quantiles[idx1][6] - plot_quantiles[idx1][2]
      rect3 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][2]),
                             w, h, color=mid)
      ax.add_patch(rect3)

      h = plot_quantiles[idx1][5] - plot_quantiles[idx1][3]
      rect4 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][3]),
                             w, h, color=mid_highlight)
      ax.add_patch(rect4)

      h = plot_quantiles[idx1][4] - plot_quantiles[idx1][4]
      rect5 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][4]),
                             w, h, color=dark)
      ax.add_patch(rect5)
    else:
      h = display_ylim[1] - display_ylim[0]
      rect = plot.Rectangle((plot_xs[idx1], display_ylim[0]),
                             w, h, color="#EEEEEE")
      ax.add_patch(rect)

  if baseline_values is not None:
    if residual:
      ax.axhline(y=0, linewidth=2, linestyle="dashed", color='#DDDDDD')
    else:
      for b in range(B):
        bin_idx = [ n for n, obs in enumerate(obs_xs)
                    if breaks[b] <= obs and obs < breaks[b + 1] ]
        if len(bin_idx) > 0:
          idx1 = 2 * b
          idx2 = 2 * b + 1
          ax.plot([plot_xs[idx1], plot_xs[idx2]],
                  [obs_means[b], obs_means[b]],
                  color="white", linewidth=4)
          ax.plot([plot_xs[idx1], plot_xs[idx2]],
                  [obs_means[b], obs_means[b]],
                  color="black", linewidth=2)

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
# @params residual Boolean value indicating whether to overlay quantiles and
#                  baseline values or plot their differences
# @param xlabel Label for x-axis; defaults to empty string
# @param ylabel Label for y-axis; defaults to None
# @param display_ylim Plot limits for y-axis; defaults to None
# @param title Plot title; defaults to empty string
def plot_conditional_median_quantiles(ax, samples, names, obs_xs,
                                      bin_min=None, bin_max=None,
                                      bin_delta=None,
                                      baseline_values=None, residual=False,
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
  N = len(names)
  C = samples[names[0]].shape[0]
  S = samples[names[0]].shape[1]
  probs = [10, 20, 30, 40, 50, 60, 70, 80, 90]

  obs_medians = [numpy.nan] * B
  median_quantiles = numpy.full((B, 9), 0.0)

  for b in range(B):
    bin_idx = [ n for n, obs in enumerate(obs_xs)
                if breaks[b] <= obs and obs < breaks[b + 1] ]
    if len(bin_idx) > 0:
      if baseline_values is not None:
        obs_medians[b] = numpy.median([ baseline_values[idx]
                                        for idx in bin_idx ] )

      for c in range(C):
        values = [ samples[name][c,:] for name in names ]

        if baseline_values is not None and residual:
          medians = [ numpy.median([ values[idx][s] for idx in bin_idx ])
                      - obs_medians[b]
                      for s in range(S) ]
        else:
          medians = [ numpy.median([ values[idx][s] for idx in bin_idx ])
                      for s in range(S) ]

        median_quantiles[b,:] += numpy.percentile(medians, probs) / C
    else:
      median_quantiles[b,:] = numpy.full((9), numpy.nan)

  plot_quantiles = [ median_quantiles[idx] for idx in plot_idxs ]

  # Plot
  if display_ylim is None:
    if baseline_values is None:
      display_ylim = [ numpy.nanmin([ q[0] for q in median_quantiles ]),
                       numpy.nanmax([ q[8] for q in median_quantiles ]) ]
    else:
      if residual:
        display_ylim = [ numpy.nanmin([ q[0] for q in median_quantiles ]),
                         numpy.nanmax([ q[8] for q in median_quantiles ]) ]
      else:
        display_ylim = [ min(numpy.nanmin([ q[0] for q in median_quantiles ]),
                             numpy.nanmin(obs_medians)),
                         max(numpy.nanmax([ q[8] for q in median_quantiles ]),
                             numpy.nanmax(obs_medians)) ]
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

    bin_idx = [ n for n, obs in enumerate(obs_xs)
                if breaks[b] <= obs and obs < breaks[b + 1] ]
    if len(bin_idx) > 0:
      h = plot_quantiles[idx1][8] - plot_quantiles[idx1][0]
      rect1 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][0]),
                             w, h, color=light)
      ax.add_patch(rect1)

      h = plot_quantiles[idx1][7] - plot_quantiles[idx1][1]
      rect2 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][1]),
                             w, h, color=light_highlight)
      ax.add_patch(rect2)

      h = plot_quantiles[idx1][6] - plot_quantiles[idx1][2]
      rect3 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][2]),
                             w, h, color=mid)
      ax.add_patch(rect3)

      h = plot_quantiles[idx1][5] - plot_quantiles[idx1][3]
      rect4 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][3]),
                             w, h, color=mid_highlight)
      ax.add_patch(rect4)

      h = plot_quantiles[idx1][4] - plot_quantiles[idx1][4]
      rect5 = plot.Rectangle((plot_xs[idx1], plot_quantiles[idx1][4]),
                             w, h, color=dark)
      ax.add_patch(rect5)
    else:
      h = display_ylim[1] - display_ylim[0]
      rect = plot.Rectangle((plot_xs[idx1], display_ylim[0]),
                             w, h, color="#EEEEEE")
      ax.add_patch(rect)

  if baseline_values is not None:
    if residual:
      ax.axhline(y=0, linewidth=2, linestyle="dashed", color='#DDDDDD')
    else:
      for b in range(B):
        bin_idx = [ n for n, obs in enumerate(obs_xs)
                    if breaks[b] <= obs and obs < breaks[b + 1] ]
        if len(bin_idx) > 0:
          idx1 = 2 * b
          idx2 = 2 * b + 1
          ax.plot([plot_xs[idx1], plot_xs[idx2]],
                  [obs_medians[b], obs_medians[b]],
                  color="white", linewidth=4)
          ax.plot([plot_xs[idx1], plot_xs[idx2]],
                  [obs_medians[b], obs_medians[b]],
                  color="black", linewidth=2)

  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_xlim([bin_min, bin_max])
  ax.set_ylabel(ylabel)
  ax.set_ylim(display_ylim)
