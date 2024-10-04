
library(MASS)


# Helper Functions
rbf_kernel <- function(t1, t2, length_scale, variance) {
  return(variance * exp(-0.5 * (t1 - t2)^2 / length_scale^2))
}

softmax <- function(x) {
  exp_x <- exp(x - max(x))  # Subtract max for numerical stability
  return(exp_x / sum(exp_x))
}

# Function to apply softmax to Lambda
apply_softmax_to_lambda <- function(Lambda) {
  N <- dim(Lambda)[1]
  K <- dim(Lambda)[2]
  T <- dim(Lambda)[3]
  
  theta <- array(0, dim = c(N, K, T))
  for (i in 1:N) {
    for (t in 1:T) {
      theta[i, , t] <- softmax(Lambda[i, , t])
    }
  }
  return(theta)
}


log_sum_exp <- function(x) {
  max_x <- max(x)
  max_x + log(sum(exp(x - max_x)))
}


precompute_K_inv <- function(T, length_scale, var_scale) {
  time_diff_matrix <- outer(1:T, 1:T, "-")^2
  Kern <- var_scale * exp(-0.5 * time_diff_matrix / length_scale^2)
  Kern <- Kern + diag(1e-6, T)  # Add small jitter for numerical stability
  K_inv <- solve(Kern)
  log_det_K <- determinant(Kern, logarithm = TRUE)$modulus
  cat("K_inv diagonal:", diag(K_inv)[1:5], "log_det_K:", log_det_K, "\n")  # Add this line
  return(list(K_inv = K_inv, log_det_K = log_det_K))
}

log_gp_prior_vec <- function(eta, mean, K_inv, log_det_K) {
  T <- length(eta)
  centered_eta <- eta - mean
  quad_form <- sum(centered_eta * (K_inv %*% centered_eta))
  log_prior <- -0.5 * (log_det_K + quad_form + T * log(2 * pi))
  cat("log_det_K:", log_det_K, "quad_form:", quad_form, "T:", T, "log_prior:", log_prior, "\n")
  return(log_prior)
}



logistic <- function(x) {
  return(1 / (1 + exp(-x)))
}
# 
# Initialization Function
initialize_mcmc <- function(y, g_i, n_topics, n_diseases, T,length_scales_lambda,length_scales_phi,var_scales_lambda,var_scales_phi,sigsmall) {
  N <- dim(y)[1]  # Number of individuals
  P <- ncol(g_i)  # Number of genetic covariates
  
  
  time_diff <- outer(seq_len(T), seq_len(T), "-")
  
  Gamma_init <- matrix(rnorm(n_topics * P, mean = 0, sd = 1),
                       nrow = n_topics,
                       ncol = P)
  
  lambda_init <- array(0, dim = c(N, n_topics, T))
  K=n_topics
  for (k in 1:K) {
    # Simulate lambda_ik(t) using a different covariance matrix for each topic
    cov_matrix <- exp(-0.5 * var_scales_lambda[k] * (time_diff^2) / length_scales_lambda[k]^
                        2)
    
    for (i in 1:N) {
      mean_lambda <- g_i[i, ] %*% Gamma_init[k, ]
      
      lambda_init[i, k, ] <- mvrnorm(
        1,
        mu = rep(mean_lambda, T), Sigma = cov_matrix
      )
    }
  }
  
  mudraw <- apply(y, c(2,3), mean)
  logmudraw <- logit(pmax(mudraw, 1e-10))  # Ensure no negative values before logit
  smoothlogmudraw <- t(apply(logmudraw, 1, function(x) predict(loess(x ~ seq_len(T)))))
  
  # Initialize Phi based on smoothed mu_d
  Phi_init <- array(0, dim = c(n_topics, n_diseases, T))
  for (k in 1:n_topics) {
    for (d in 1:n_diseases) {
      t <- seq_len(T)
      Sigma <- var_scales_phi[k] * exp(-0.5 * outer(t, t, "-")^2 / length_scales_phi[k]^2)
      Phi_init[k, d, ] <- smoothlogmudraw[d, ] + mvrnorm(1, mu = rep(0, T), Sigma = Sigma)
    }
  }
  
  mu_d_init <- smoothlogmudraw
  
  
  
  
  return(
    list(
      Lambda = lambda_init,
      Phi = Phi_init,
      Gamma = Gamma_init,
      mu_d = mu_d_init,
      length_scales_lambda = length_scales_lambda,
      var_scales_lambda = var_scales_lambda,
      length_scales_phi = length_scales_phi,
      var_scales_phi = var_scales_phi
    )
  )
}

# Log-likelihood function
log_likelihood <- function(y, Lambda, Phi) {
  n_individuals <- dim(Lambda)[1]
  n_topics <- dim(Lambda)[2]
  n_diseases <- dim(Phi)[2]
  T <- dim(Lambda)[3]
  
  theta <- apply_softmax_to_lambda(Lambda) # Apply softmax to Lambda
  pi <- array(0, dim = c(n_individuals, n_diseases, T))
  
  for (t in 1:T) {
    pi[, , t] <- theta[, , t] %*% logistic(Phi[, , t])
  }
  
  log_lik <- 0
  for (i in 1:n_individuals) {
    for (d in 1:n_diseases) {
      at_risk <- which(cumsum(y[i, d, ]) == 0)
      if (length(at_risk) > 0) {
        event_time <- max(at_risk) + 1
        if (event_time <= T) {
          log_lik <- log_lik + log(pi[i, d, event_time])
        }
        log_lik <- log_lik + sum(log(1 - pi[i, d, at_risk]))
      } else {
        log_lik <- log_lik + log(pi[i, d, 1])
      }
    }
  }
  return(log_lik)
}



mcmc_sampler_softmax <- function(y, g_i, n_iterations, initial_values) {
  current_state <- initial_values
  n_individuals <- dim(current_state$Lambda)[1]
  n_topics <- dim(current_state$Lambda)[2]
  T <- dim(current_state$Lambda)[3]
  n_diseases <- dim(current_state$Phi)[2]
  P <- ncol(g_i)
  
  # Initialize storage for samples and diagnostics
  samples <- list(
    Lambda = array(0, dim = c(
      n_iterations, dim(current_state$Lambda)
    )),
    Phi = array(0, dim = c(n_iterations, dim(
      current_state$Phi
    ))),
    Gamma = array(0, dim = c(
      n_iterations, dim(current_state$Gamma)
    ))
  )
  log_likelihoods <- numeric(n_iterations)
  log_posteriors <- numeric(n_iterations)
  acceptance_rates <- list(Lambda = 0, Phi = 0)
  
  # Initialize proposal standard deviations
  adapt_sd <- list(Lambda = array(0.01, dim = dim(current_state$Lambda)),
                   Phi = array(0.01, dim = dim(current_state$Phi)))
  
  # Precompute inverse covariance matrices
  K_inv_lambda <- lapply(1:n_topics, function(k)
    precompute_K_inv(
      T,
      current_state$length_scales_lambda[k],
      current_state$var_scales_lambda[k]
    ))
  K_inv_phi <- lapply(1:n_topics, function(k)
    precompute_K_inv(
      T,
      current_state$length_scales_phi[k],
      current_state$var_scales_phi[k]
    ))
  
  for (iter in 1:n_iterations) {
    # Update Lambda
    proposed_Lambda <- current_state$Lambda + array(rnorm(prod(dim(
      current_state$Lambda
    )), 0, adapt_sd$Lambda),
    dim = dim(current_state$Lambda))
    
    current_log_lik <- log_likelihood(y, current_state$Lambda, current_state$Phi)
    proposed_log_lik <- log_likelihood(y, proposed_Lambda, current_state$Phi)
    
    
    current_log_prior_lambda <- log_sum_exp(unlist(sapply(1:n_individuals, function(i) {
      sapply(1:n_topics, function(k) {
        log_gp_prior_vec(
          current_state$Lambda[i, k, ],
          rep(g_i[i, ] %*% current_state$Gamma[k, ], T),
          K_inv_lambda[[k]]$K_inv,
          K_inv_lambda[[k]]$log_det_K
        )
      })
    })))
    
    proposed_log_prior_lambda <- log_sum_exp(unlist(sapply(1:n_individuals, function(i) {
      sapply(1:n_topics, function(k) {
        log_gp_prior_vec(
          proposed_Lambda[i, k, ],
          rep(g_i[i, ] %*% current_state$Gamma[k, ], T),
          K_inv_lambda[[k]]$K_inv,
          K_inv_lambda[[k]]$log_det_K
        )
      })
    })))
    
    log_accept_ratio <- (proposed_log_lik + proposed_log_prior_lambda) -
      (current_log_lik + current_log_prior_lambda)
    
    if (log(runif(1)) < log_accept_ratio) {
      current_state$Lambda <- proposed_Lambda
      adapt_sd$Lambda <- adapt_sd$Lambda * 1.01
      acceptance_rates$Lambda <- acceptance_rates$Lambda + 1
    } else {
      adapt_sd$Lambda <- adapt_sd$Lambda * 0.99
    }
    
    # Update Phi
    proposed_Phi <- current_state$Phi + array(rnorm(prod(dim(
      current_state$Phi
    )), 0, adapt_sd$Phi), dim = dim(current_state$Phi))
    
    current_log_lik <- log_likelihood(y, current_state$Lambda, current_state$Phi)
    proposed_log_lik <- log_likelihood(y, current_state$Lambda, proposed_Phi)
    
    
    
    current_log_prior_phi <- log_sum_exp(unlist(sapply(1:n_topics, function(k) {
      sapply(1:n_diseases, function(d) {
        log_gp_prior_vec(
          current_state$Phi[k, d, ],
          current_state$mu_d[d, ],
          K_inv_phi[[k]]$K_inv,
          K_inv_phi[[k]]$log_det_K
        )
      })
    })))
    
    proposed_log_prior_phi <- log_sum_exp(unlist(sapply(1:n_topics, function(k) {
      sapply(1:n_diseases, function(d) {
        log_gp_prior_vec(
          proposed_Phi[k, d, ],
          current_state$mu_d[d, ],
          K_inv_phi[[k]]$K_inv,
          K_inv_phi[[k]]$log_det_K
        )
      })
    })))
    
    log_accept_ratio <- (proposed_log_lik + proposed_log_prior_phi) -
      (current_log_lik + current_log_prior_phi)
    
    if (log(runif(1)) < log_accept_ratio) {
      current_state$Phi <- proposed_Phi
      adapt_sd$Phi <- adapt_sd$Phi * 1.01
      acceptance_rates$Phi <- acceptance_rates$Phi + 1
    } else {
      adapt_sd$Phi <- adapt_sd$Phi * 0.99
    }
    
    # Update Gamma using Gibbs sampler
    for (k in 1:n_topics) {
      Lambda_k <- current_state$Lambda[, k, ]  # N x T matrix for topic k
      K_inv <- K_inv_lambda[[k]]$K_inv  # T x T inverse covariance matrix
      
      # Compute posterior precision (inverse covariance), see standard MVN derivatino using design matrix on X instead of N
      posterior_precision <- diag(1, P)  # Prior precision (assuming standard normal prior)
      posterior_mean <- rep(0, P)  # Prior mean
      
      for (i in 1:N) {
        Xi <- matrix(rep(g_i[i, ], T), nrow = T, byrow = TRUE)  # T x P matrix
        precision_contrib <- t(Xi) %*% K_inv %*% Xi
        posterior_precision <- posterior_precision + precision_contrib
        posterior_mean <- posterior_mean + t(Xi) %*% K_inv %*% Lambda_k[i, ]
      }
      
      # Compute posterior covariance and mean
      posterior_covariance <- solve(posterior_precision, tol = 1e-20)
      posterior_mean <- posterior_covariance %*% posterior_mean
      
      # Sample new Gamma_k
      current_state$Gamma[k, ] <- mvrnorm(1, mu = posterior_mean, Sigma = posterior_covariance)
    }
    
    # Store samples and diagnostics
    samples$Lambda[iter, , , ] <- current_state$Lambda
    samples$Phi[iter, , , ] <- current_state$Phi
    samples$Gamma[iter, , ] <- current_state$Gamma
    
    log_likelihoods[iter] <- current_log_lik
    log_posteriors[iter] <- current_log_lik + current_log_prior_lambda + current_log_prior_phi +
      sum(dnorm(current_state$Gamma, 0, 1, log = TRUE))
    
    cat("current_log_lik:", current_log_lik, "\n")
    cat("current_log_prior_lambda:",
        current_log_prior_lambda,
        "\n")
    cat("current_log_prior_phi:", current_log_prior_phi, "\n")
    cat("log_prior_gamma:", sum(dnorm(current_state$Gamma, 0, 1, log = TRUE)), "\n")
    
    
    # Print progress
    #if (iter %% 10 == 0) {
    cat(
      "Iteration",
      iter,
      "Log posterior:",
      log_posteriors[iter],
      "Log-likelihood:",
      log_likelihoods[iter],
      "\n"
    )
    cat(
      "Acceptance rates: Lambda =",
      acceptance_rates$Lambda / iter,
      "Phi =",
      acceptance_rates$Phi / iter,
      "\n"
    )
    #}
  }
  
  # Calculate final acceptance rates
  for (param in names(acceptance_rates)) {
    acceptance_rates[[param]] <- acceptance_rates[[param]] / n_iterations
  }
  
  return(
    list(
      samples = samples,
      log_likelihoods = log_likelihoods,
      acceptance_rates = acceptance_rates,
      log_posteriors = log_posteriors
    )
  )
}