
# Load required libraries
set.seed(123)

### make simwithlogit a function to simulate

source("../R/newsim.R")
source("utils/utils.R")
source("utils/model_functions.R")
source("utils/sampling_methods.R")
source("utils/initialization.R")
source("../R/aladynoulli.R")


data <- generate_tensor_data(num_covariates = 5,K = 3,T = 20,D = 5,N = 100)

Y <- data$Y
G <- data$G
plot_individuals(data$S,num_individuals = 3)
# Here you initialize the MCMC
initial_values <- mcmc_init_two(y = Y, G = G, n_topics = 3)
a=aladynoulli(Y, G, n_topics = 3, nsamples=100, nburnin=100,n_iters=1000)

