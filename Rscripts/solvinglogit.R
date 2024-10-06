
# Load required libraries
set.seed(123)
source("simulations/simwithlogit.R")
source("utils.R")
source("mcmc_sampler.R")
source("utils/utils.R")
source("utils/model_functions.R")
source("utils/sampling_methods.R")
source("utils/initialization.R")


# main execution
# Assuming y and g_i are already loaded
n_topics <- 3  # Set this to your desired number of topics
n_diseases <- dim(y)[2]
T <- dim(y)[3]

initial_values <- initialize_mcmc(y, g_i, n_topics, n_diseases, T,var_scales_phi = var_scales_phi,length_scales_lambda = length_scales_lambda,length_scales_phi = length_scales_phi,var_scales_lambda = var_scales_lambda,sigsmall = 0.01)

n_iterations <- 2000
samples <- mcmc_sampler_softmax(y, g_i, n_iterations, initial_values)