
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


