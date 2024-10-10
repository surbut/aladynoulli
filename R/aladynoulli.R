

aladynoulli <- function(Y, G, n_topics = 3, nsamples, nburnin){
  
  
  
  Ttot <- dim(Y)[3]
  
  
  # Matrix of indexed to ignore. 
  # Create a matrix of the indexes for the time-to-event for each patient-disease
  precomputed_indices <- precompute_likelihood_indices(Y)
  
  # Here you initialize the MCMC
  initial_values <- initialize_mcmc(Y, 
                                    G, 
                                    n_topics, 
                                    n_diseases, 
                                    Ttot,
                                    var_scales_phi = var_scales_phi,
                                    length_scales_lambda = length_scales_lambda,
                                    length_scales_phi = length_scales_phi,
                                    var_scales_lambda = var_scales_lambda,
                                    sigsmall = 0.01)

  
  # Here you run it
  
  
  
  
}


update_Lambda <- function(Y, Lambda, G, Gamma, s = 0.01){
  
  n_individuals <- dim(Lambda)[1]
  n_topics <- dim(Lambda)[2]
  Ttot <- dim(Lambda)[3]
  
  #pb <- txtProgressBar(style=3)
  for(i in 1:n_individuals){
    print(i)
    #setTxtProgressBar(pb, i/n_individuals)
    for(k in 1:n_topics){
      # Sample Lambda from past value
      lambda_new <- c(rmvnorm(1, Lambda[i, k, ], sigma = s * diag(Ttot)))
      # Evaluate log posterior
      Lambda_new <- Lambda
      Lambda_new[i, k, ] <- lambda_new
      
      lpost_new <- compute_log_likelihood(Lambda_new, Phi, precomputed_indices) + 
        log_gp_prior_vec(lambda_new, rep(G[i, ] %*% Gamma[k, ], Ttot), K_inv = K_inv_lambda[[k]]$K_inv, 
                         log_det_K = K_inv_lambda[[k]]$log_det_K)
      
      lpost_old <- compute_log_likelihood(Lambda, Phi, precomputed_indices) + 
        log_gp_prior_vec(Lambda[i, k, ], rep(G[i, ] %*% Gamma[k, ], Ttot), K_inv = K_inv_lambda[[k]]$K_inv, 
                         log_det_K = K_inv_lambda[[k]]$log_det_K)
      # Accept/reject
      log_acc <- lpost_new - lpost_old
      if(log(runif(1)) < log_acc){
        Lambda[i, k, ] <- lambda_new
        print("accepted! :)")
      }
    }
  }
}

for(t in 1:Ttot){
  print(t)
  #setTxtProgressBar(pb, i/n_individuals)
  for(k in 1:n_topics){
    # Sample Lambda from past value
    lambda_new <- c(rmvnorm(1, Lambda[, k, t], sigma = s * diag(n_individuals)))
    # Evaluate log posterior
    Lambda_new <- Lambda
    Lambda_new[, k, t] <- lambda_new
    
    lpost_new <- compute_log_likelihood(Lambda_new, Phi, precomputed_indices) + 
      log_gp_prior_vec(lambda_new, rep(G[i, ] %*% Gamma[k, ], Ttot), K_inv = K_inv_lambda[[k]]$K_inv, 
                       log_det_K = K_inv_lambda[[k]]$log_det_K)
    
    lpost_old <- compute_log_likelihood(Lambda, Phi, precomputed_indices) + 
      log_gp_prior_vec(Lambda[i, k, ], rep(G[i, ] %*% Gamma[k, ], Ttot), K_inv = K_inv_lambda[[k]]$K_inv, 
                       log_det_K = K_inv_lambda[[k]]$log_det_K)
    # Accept/reject
    log_acc <- lpost_new - lpost_old
    if(log(runif(1)) < log_acc){
      Lambda[i, k, ] <- lambda_new
      print("accepted! :)")
    }
  }
}
  
  









