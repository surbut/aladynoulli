mcmc_sampler_elliptical <- function(y, g_i, n_iterations, initial_values) {
  current_state <- initial_values
  n_individuals <- dim(current_state$Lambda)[1]
  n_topics <- dim(current_state$Lambda)[2]
  T <- dim(current_state$Lambda)[3]
  n_diseases <- dim(current_state$Phi)[2]
  P <- ncol(g_i)
  
  # Initialize storage for samples and diagnostics
  samples <- list(
    Lambda = array(0, dim = c(n_iterations, dim(current_state$Lambda))),
    Phi = array(0, dim = c(n_iterations, dim(current_state$Phi))),
    Gamma = array(0, dim = c(n_iterations, dim(current_state$Gamma)))
  )
  log_likelihoods <- numeric(n_iterations)
  log_posteriors <- numeric(n_iterations)
  
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
    # Update Lambda using elliptical slice sampling
    for (i in 1:n_individuals) {
      for (k in 1:n_topics) {
        prior_mean <- rep(g_i[i, ] %*% current_state$Gamma[k, ], T)
        prior_cov <- solve(K_inv_lambda[[k]]$K_inv)
        current_state$Lambda[i, k, ] <- elliptical_slice(
          current_state$Lambda[i, k, ],
          prior_mean,
          prior_cov,
          function(x, args) log_likelihood(args$y, update_lambda(args$Lambda, args$i, args$k, x), args$Phi),
          list(y = y, Lambda = current_state$Lambda, Phi = current_state$Phi, i = i, k = k)
        )
      }
    }
    
    # Update Phi using elliptical slice sampling
    for (k in 1:n_topics) {
      for (d in 1:n_diseases) {
        prior_mean <- current_state$mu_d[d, ]
        prior_cov <- solve(K_inv_phi[[k]]$K_inv)
        current_state$Phi[k, d, ] <- elliptical_slice(
          current_state$Phi[k, d, ],
          prior_mean,
          prior_cov,
          function(x, args) log_likelihood(args$y, args$Lambda, update_phi(args$Phi, args$k, args$d, x)),
          list(y = y, Lambda = current_state$Lambda, Phi = current_state$Phi, k = k, d = d)
        )
      }
    }
    
    # Update Gamma using Gibbs sampling
    for (k in 1:n_topics) {
      Lambda_k <- current_state$Lambda[, k, ]  # N x T matrix for topic k
      K_inv <- K_inv_lambda[[k]]$K_inv  # T x T inverse covariance matrix
      
      posterior_precision <- diag(1, P)  # Prior precision (assuming standard normal prior)
      posterior_mean <- rep(0, P)  # Prior mean
      
      for (i in 1:n_individuals) {
        Xi <- matrix(rep(g_i[i, ], T), nrow = T, byrow = TRUE)  # T x P matrix
        precision_contrib <- t(Xi) %*% K_inv %*% Xi
        posterior_precision <- posterior_precision + precision_contrib
        posterior_mean <- posterior_mean + t(Xi) %*% K_inv %*% Lambda_k[i, ]
      }
      
      posterior_covariance <- solve(posterior_precision)
      posterior_mean <- posterior_covariance %*% posterior_mean
      
      current_state$Gamma[k, ] <- mvrnorm(1, mu = posterior_mean, Sigma = posterior_covariance)
    }
    
    # Store samples and diagnostics
    samples$Lambda[iter, , , ] <- current_state$Lambda
    samples$Phi[iter, , , ] <- current_state$Phi
    samples$Gamma[iter, , ] <- current_state$Gamma
    
    log_likelihoods[iter] <- log_likelihood(y, current_state$Lambda, current_state$Phi)
    
    # Calculate log posterior
    log_prior_lambda <- sum(sapply(1:n_individuals, function(i) {
      sapply(1:n_topics, function(k) {
        log_gp_prior_vec(
          current_state$Lambda[i, k, ],
          rep(g_i[i, ] %*% current_state$Gamma[k, ], T),
          K_inv_lambda[[k]]$K_inv,
          K_inv_lambda[[k]]$log_det_K
        )
      })
    }))
    
    log_prior_phi <- sum(sapply(1:n_topics, function(k) {
      sapply(1:n_diseases, function(d) {
        log_gp_prior_vec(
          current_state$Phi[k, d, ],
          current_state$mu_d[d, ],
          K_inv_phi[[k]]$K_inv,
          K_inv_phi[[k]]$log_det_K
        )
      })
    }))
    
    log_prior_gamma <- sum(dnorm(current_state$Gamma, 0, 1, log = TRUE))
    
    log_posteriors[iter] <- log_likelihoods[iter] + log_prior_lambda + log_prior_phi + log_prior_gamma
    
    # Print progress
    if (iter %% 100 == 0) {
      cat("Iteration", iter, "of", n_iterations, "\n")
      cat("Log posterior:", log_posteriors[iter], "\n")
      cat("Log likelihood:", log_likelihoods[iter], "\n")
    }
  }
  
  return(list(
    samples = samples,
    log_likelihoods = log_likelihoods,
    log_posteriors = log_posteriors
  ))
}