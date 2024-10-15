
# Load required libraries
set.seed(123)

### make simwithlogit a function to simulate

source("../R/newsim.R")
source("mcmc_with_elliptical.R")
source("mcmc_sampler.R")
source("utils/utils.R")
source("utils/model_functions.R")
source("utils/sampling_methods.R")
source("utils/initialization.R")


data <- generate_tensor_data(num_covariates = 5,K = 3,T = 20,D = 5,N = 100)

Y <- data$Y
G <- data$G
plot_individuals(data$S,num_individuals = 3)
# Here you initialize the MCMC
initial_values <- mcmc_init_two(y = Y, G = G)
a=aladynoulli(Y, G, n_topics = 3, nsamples=100, nburnin=100,niters=1000)

