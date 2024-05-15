data {
  int<lower=1> N;
  vector[N] x; // Observed inputs
  vector[N] y; // Observed outputs
  
  int<lower=1> N_grid; // Number of grid points for quantifying functional behavior
  vector[N_grid] x_grid; // Grid points for quantifying functional behavior
}

parameters { 
  real alpha;
  real beta;
  real<lower=0> sigma;
}

model {
  alpha ~ normal(0, 3 / 2.32);
  beta ~ normal(0, 3 / 2.32);
  sigma ~ normal(0, 1 / 2.57);
  
  y ~ normal(alpha + beta * x, sigma);
}

generated quantities {
  vector[N_grid] f_grid = alpha + beta * x_grid;
  array[N_grid] real y_pred_grid = normal_rng(f_grid, sigma);
}
