library(parallel)

mcmc_sampler_elliptical <- function(y, g_i, n_iterations, initial_values, n_cores = 1) {
  current_state <- initial_values
  n_individuals <- dim(current_state$Lambda)[1]
  n_topics <- dim(current_state$Lambda)[2]
  T <- dim(current_state$Lambda)[3]
  n_diseases <- dim(current_state$Phi)[2]
  P <- ncol(g_i)
  
  samples <- list(
    Lambda = array(0, dim = c(n_iterations, dim(current_state$Lambda))),
    Phi = array(0, dim = c(n_iterations, dim(current_state$Phi))),
    Gamma = array(0, dim = c(n_iterations, dim(current_state$Gamma)))
  )
  
  log_likelihoods <- numeric(n_iterations)
  log_posteriors <- numeric(n_iterations)
  
  K_lambda <- lapply(1:n_topics, function(k)
    compute_kernel_matrix(T, current_state$length_scales_lambda[k], current_state$var_scales_lambda[k]))
  K_phi <- lapply(1:n_topics, function(k)
    compute_kernel_matrix(T, current_state$length_scales_phi[k], current_state$var_scales_phi[k]))
  
  K_inv_lambda <- lapply(K_lambda, solve)
  K_inv_phi <- lapply(K_phi, solve)
  
  update_lambda_block <- function(i, current_Lambda, current_Gamma) {
    new_Lambda <- current_Lambda
    for (k in 1:n_topics) {
      prior_mean <- rep(g_i[i,] %*% current_Gamma[k,], T)
      new_Lambda[i,k,] <- elliptical_slice(
        current_Lambda[i,k,],
        prior_mean,
        K_lambda[[k]],
        function(x) log_likelihood(y, update_lambda(current_Lambda, i, k, x), current_state$Phi)
      )
    }
    return(new_Lambda[i,,])
  }
  
  update_phi_block <- function(k, current_Phi) {
    new_Phi <- current_Phi
    for (d in 1:n_diseases) {
      prior_mean <- current_state$mu_d[d,]
      new_Phi[k,d,] <- elliptical_slice(
        current_Phi[k,d,],
        prior_mean,
        K_phi[[k]],
        function(x) log_likelihood(y, current_state$Lambda, update_phi(current_Phi, k, d, x))
      )
    }
    return(new_Phi[k,,])
  }
  
  cl <- makeCluster(n_cores)
  parallel::clusterExport(cl, c("elliptical_slice", "log_likelihood", "update_lambda", "update_phi", "rmvn"))
  for (iter in 1:n_iterations) {
    # Update Lambda using parallel processing
    lambda_updates <- parLapply(cl, 1:n_individuals, update_lambda_block, 
                                current_Lambda = current_state$Lambda, 
                                current_Gamma = current_state$Gamma)
    current_state$Lambda <- array(unlist(lambda_updates), dim = dim(current_state$Lambda))
    
    # Update Phi using parallel processing
    phi_updates <- parLapply(cl, 1:n_topics, update_phi_block, 
                             current_Phi = current_state$Phi)
    current_state$Phi <- array(unlist(phi_updates), dim = dim(current_state$Phi))
    
    # Update Gamma using Gibbs sampling (vectorized)
    for (k in 1:n_topics) {
      Lambda_k <- current_state$Lambda[,k,]  # N x T matrix for topic k
      K_inv <- K_inv_lambda[[k]]
      
      posterior_precision <- diag(1, P) + t(g_i) %*% K_inv %*% g_i
      posterior_mean <- solve(posterior_precision, t(g_i) %*% K_inv %*% t(Lambda_k))
      
      current_state$Gamma[k,] <- mvrnorm(1, posterior_mean, solve(posterior_precision))
    }
    
    # Store samples and compute diagnostics
    samples$Lambda[iter,,,] <- current_state$Lambda
    samples$Phi[iter,,,] <- current_state$Phi
    samples$Gamma[iter,,] <- current_state$Gamma
    
    log_likelihoods[iter] <- log_likelihood(y, current_state$Lambda, current_state$Phi)
    
    log_prior_lambda <- sum(sapply(1:n_individuals, function(i) {
      sapply(1:n_topics, function(k) {
        log_gp_prior_direct(current_state$Lambda[i,k,], rep(g_i[i,] %*% current_state$Gamma[k,], T), K_lambda[[k]])
      })
    }))
    
    log_prior_phi <- sum(sapply(1:n_topics, function(k) {
      sapply(1:n_diseases, function(d) {
        log_gp_prior_direct(current_state$Phi[k,d,], current_state$mu_d[d,], K_phi[[k]])
      })
    }))
    
    log_prior_gamma <- sum(dnorm(current_state$Gamma, 0, 1, log = TRUE))
    
    log_posteriors[iter] <- log_likelihoods[iter] + log_prior_lambda + log_prior_phi + log_prior_gamma
    
    if (iter %% 10 == 0) {
      cat("Iteration", iter, "of", n_iterations, "\n")
      cat("Log posterior:", log_posteriors[iter], "\n")
      cat("Log likelihood:", log_likelihoods[iter], "\n")
    }
  }
  
  stopCluster(cl)
  
  return(list(
    samples = samples,
    log_likelihoods = log_likelihoods,
    log_posteriors = log_posteriors
  ))
}