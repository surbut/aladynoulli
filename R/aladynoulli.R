

aladynoulli <- function(y, G, n_topics = 3, nsamples, nburnin){
  
  
  
  Ttot <- dim(y)[3]
  
  
  # Matrix of indexed to ignore. 
  # Create a matrix of the indexes for the time-to-event for each patient-disease
  at_risk <- precompute_likelihood_indices(Y)
  
  # Here you initialize the MCMC
  initial_values <- initialize_mcmc(y, 
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









