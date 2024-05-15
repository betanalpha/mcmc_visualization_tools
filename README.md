A collection of functions for visualizing Markov chain Monte Carlo output in
`R` with the base graphics library and `python` with `matplotlib`.  While the
functions are relatively general they are designed to facilitate the
implementation of Bayesian inference, including visual prior checks, visual
posterior retrodictive checks, and the visualization of marginal posterior
inferences.

Most of the visualization functions assume that Markov chain Monte Carlo output
is organized into named lists or dictionaries for `R` and `python`, each
consisting of two-dimensional arrays indexed by Markov chain and then iteration.
This format is also used in https://github.com/betanalpha/mcmc_diagnostics.

Each folder contains an extensive demonstration of the visualization functions
applied to Bayesian inference.