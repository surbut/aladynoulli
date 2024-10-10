
# Load required libraries
set.seed(123)

### make simwithlogit a function to simulate

source("simulations/simwithlogit.R")
source("mcmc_with_elliptical.R")
source("mcmc_sampler.R")
source("utils/utils.R")
source("utils/model_functions.R")
source("utils/sampling_methods.R")
source("utils/initialization.R")


# main execution
# Assuming y and g_i are already loaded
n_topics <- 3  # Set this to your desired number of topics
n_diseases <- dim(Y)[2]
Ttot <- dim(Y)[3]

initial_values <- initialize_mcmc(y, g_i, n_topics, n_diseases, Ttot,
                                  var_scales_phi = var_scales_phi,
                                  length_scales_lambda = length_scales_lambda,
                                  length_scales_phi = length_scales_phi,
                                  var_scales_lambda = var_scales_lambda,sigsmall = 0.01)

n_iterations <- 100
#samples <- mcmc_sampler_elliptical(y, g_i, n_iterations, initial_values)
samples <- mcmc_sampler_softmax(y, g_i, n_iterations, initial_values)
