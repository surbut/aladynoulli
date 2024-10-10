generate_tensor_data <- function(N = 1000, D = 5, T = 50, K = 3, num_covariates = 5) {
  library(mvtnorm)
  library(MASS)
  library(pracma)
  library(reshape2)
  library(dplyr)
  library(einsum)
  
  set.seed(123)
  
  # Create time differences matrix
  time_diff <- outer(seq_len(T), seq_len(T), "-")
  
  # Generate scales
  length_scales_lambda <- runif(K, T / 3, T / 2)
  var_scales_lambda <- runif(K, 0.8, 1.2)
  length_scales_phi <- runif(K, T / 3, T / 2)
  var_scales_phi <- runif(K, 0.8, 1.2)
  length_scales_mu <- runif(D, T / 3, T / 2)
  var_scales_mu <- runif(D, 0.8, 1.2)
  
  # Simulate mu
  mu_d <- array(NA, dim = c(D, T))
  for (d in 1:D) {
    cov_matrix_mu <- exp(-0.5 * var_scales_mu[d] * (time_diff ^ 2) / length_scales_mu[d] ^ 2)
    mu_d[d, ] <- mvrnorm(1, mu = rep(qlogis(0.10), T), Sigma = cov_matrix_mu)
  }
  
  # Generate lambda, phi matrices
  g_i <- array(rnorm(num_covariates * N), dim = c(N, num_covariates))
  lambda_ik <- array(NA, dim = c(N, K, T))
  phi_kd <- array(NA, dim = c(K, D, T))
  Gamma_k <- matrix(rnorm(num_covariates * K), nrow = K, ncol = num_covariates)
  
  # Simulate lambda
  for (k in 1:K) {
    cov_matrix <- exp(-0.5 * var_scales_lambda[k] * (time_diff ^ 2) / length_scales_lambda[k] ^ 2)
    for (i in 1:N) {
      mean_lambda <- g_i[i, ] %*% Gamma_k[k, ]
      lambda_ik[i, k, ] <- mvrnorm(1, mu = rep(mean_lambda, T), Sigma = cov_matrix)
    }
  }
  
  # Apply softmax to lambda
  theta <- apply(lambda_ik, c(1,3), function(x) exp(x) / sum(exp(x)))
  theta <- aperm(theta, c(2,1,3))  # Reorder dimensions to match original lambda_ik
  
  # Simulate phi
  for (k in 1:K) {
    cov_matrix <- exp(-0.5 * var_scales_phi[k] * (time_diff ^ 2) / length_scales_phi[k] ^ 2)
    for (d in 1:D) {
      phi_kd[k, d, ] <- mvrnorm(1, mu = mu_d[d,], Sigma = cov_matrix)
    }
  }
  
  eta <- plogis(phi_kd)
  
  # Generate pi and Y
  pi <- einsum('nkt,kdt->ndt', theta, eta)
  Y <- array(rbinom(n = N*D*T, size = 1, prob = pi), dim = c(N, D, T))
  
  # Return all generated data and parameters
  return(list(
    Y = Y,
    G = g_i,
    var_scales_lambda = var_scales_lambda,
    length_scales_lambda = length_scales_lambda,
    var_scales_phi = var_scales_phi,
    length_scales_phi = length_scales_phi,
    var_scales_mu = var_scales_mu,
    length_scales_mu = length_scales_mu,
    mu_d = mu_d,
    lambda_ik = lambda_ik,
    phi_kd = phi_kd,
    Gamma_k = Gamma_k,
    pi = pi,
    s = s,
    eta = eta
  ))
}