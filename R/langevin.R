library(rsvd)  # For fast randomized SVD
library(mgcv)  # For Gaussian Process utilities
library(MASS)  # For mvrnorm function

# Helper functions
softmax <- function(x) {
  exp_x <- exp(x - max(x))
  exp_x / sum(exp_x)
}

sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

# Initialization function (your existing function)
mcmc_init_two <- function(y, G, n_topics, length_scales_lambda, var_scales_lambda, length_scales_phi, var_scales_phi) {
  N <- dim(y)[1]  # Number of individuals
  D <- dim(y)[2]  # Number of diseases
  Ttot <- dim(y)[3]  # Number of time points
  P <- ncol(G)  # Number of genetic covariates
  K <- n_topics   # Number of topics
  
  # 1. Perform SVD on the time-averaged data
  Y_mean <- apply(y, c(1,2), mean)
  svd_result <- rsvd(Y_mean, k = K)
  A1 <- svd_result$u %*% diag(sqrt(svd_result$d[1:K]))
  A2 <- t(diag(sqrt(svd_result$d[1:K])) %*% t(svd_result$v))
  
  # 2. Create time basis (polynomial without intercept because we are using the genetics or disease prevalence)
  time_basis <- cbind(1, poly(seq_len(Ttot), degree = min(Ttot-1, 3), simple = TRUE))
  
  # 3. Initialize and project Lambda
  lambda_init <- array(0, dim = c(N, K, Ttot))
  
  Gamma_init <- matrix(0, nrow = K, ncol = P)
  for (k in 1:K) {
    Gamma_init[k, ] <- coef(lm(A1[, k] ~ G - 1)) ## because centered around genetics
    for (i in 1:N) {
      mean_lambda <- rep(G[i, ] %*% Gamma_init[k, ], Ttot)
      lambda_init[i, k, ] <- mean_lambda + 
        mvrnorm(1, mu = rep(0, Ttot), Sigma = var_scales_lambda[k] * exp(-0.5 * outer(seq_len(Ttot), seq_len(Ttot), "-")^2 / length_scales_lambda[k]^2))
    }
  }
  
  # 4. Calculate mu_d and initialize Phi
  mudraw <- apply(y, c(2,3), mean)
  logmudraw <- qlogis(pmax(pmin(mudraw, 1-1e-10), 1e-10))  # Ensure values are within (0,1) before logit
  mu_d_init <- t(apply(logmudraw, 1, function(x) predict(loess(x ~ seq_len(Ttot)))))
  
  Phi_init <- array(0, dim = c(K, D, Ttot))
  for (k in 1:K) {
    for (d in 1:D) {
      Sigma <- var_scales_phi[k] * exp(-0.5 * outer(seq_len(Ttot), seq_len(Ttot), "-")^2 / length_scales_phi[k]^2)
      Phi_init[k, d, ] <- mu_d_init[d, ] + mvrnorm(1, mu = rep(0, Ttot), Sigma = Sigma)
    }
  }
  
  return(
    list(
      Lambda = lambda_init,
      Phi = Phi_init,
      Gamma = Gamma_init,
      mu_d = mu_d_init
    )
  )
}

# Gradient computation functions
compute_gradient_log_likelihood_lambda <- function(Lambda, Phi, Y, i, k, t) {
  D <- dim(Phi)[2]
  K <- dim(Lambda)[2]
  
  gradient <- 0
  for (d in 1:D) {
    theta <- softmax(Lambda[i,,t])
    pi_idt <- sum(theta * sigmoid(Phi[,d,t]))
    dL_dpi <- Y[i,d,t]/pi_idt - (1-Y[i,d,t])/(1-pi_idt)
    dpi_dlambda <- sigmoid(Phi[k,d,t]) * theta[k] * (1 - theta[k])
    gradient <- gradient + dL_dpi * dpi_dlambda
  }
  
  return(gradient)
}

compute_gradient_log_likelihood_phi <- function(Lambda, Phi, Y, k, d, t) {
  N <- dim(Lambda)[1]
  
  gradient <- 0
  for (i in 1:N) {
    theta <- softmax(Lambda[i,,t])
    pi_idt <- sum(theta * sigmoid(Phi[,d,t]))
    dL_dpi <- Y[i,d,t]/pi_idt - (1-Y[i,d,t])/(1-pi_idt)
    dpi_dphi <- theta[k] * sigmoid(Phi[k,d,t]) * (1 - sigmoid(Phi[k,d,t]))
    gradient <- gradient + dL_dpi * dpi_dphi
  }
  
  return(gradient)
}

compute_gradient_log_prior_lambda <- function(Lambda_i, K_inv_lambda) {
  return(-K_inv_lambda %*% Lambda_i)
}

compute_gradient_log_prior_phi <- function(Phi_k, K_inv_phi) {
  return(-K_inv_phi %*% Phi_k)
}

# SGLD update functions
update_lambda <- function(Lambda, Phi, Y, K_inv_lambda, i, k, t, epsilon) {
  grad_likelihood <- compute_gradient_log_likelihood_lambda(Lambda, Phi, Y, i, k, t)
  grad_prior <- (K_inv_lambda %*% Lambda[i,k,])[t]
  full_grad <- grad_likelihood - grad_prior
  
  eta <- rnorm(1, 0, sqrt(epsilon))
  Lambda[i,k,t] <- Lambda[i,k,t] + 0.5 * epsilon * full_grad + eta
  
  return(Lambda)
}

update_phi <- function(Lambda, Phi, Y, K_inv_phi, k, d, t, epsilon) {
  grad_likelihood <- compute_gradient_log_likelihood_phi(Lambda, Phi, Y, k, d, t)
  grad_prior <- (K_inv_phi %*% Phi[k,d,])[t]
  full_grad <- grad_likelihood - grad_prior
  
  eta <- rnorm(1, 0, sqrt(epsilon))
  Phi[k,d,t] <- Phi[k,d,t] + 0.5 * epsilon * full_grad + eta
  
  return(Phi)
}
# Main Aladynoulli function
aladynoulli <- function(Y, G, n_topics = 3, n_iters = 1000, step_size_lambda = 0.01, step_size_phi = 0.01, 
                        length_scales_lambda, var_scales_lambda, length_scales_phi, var_scales_phi) {
  N <- dim(Y)[1]  # Number of individuals
  D <- dim(Y)[2]  # Number of diseases
  Ttot <- dim(Y)[3]  # Number of time points
  P <- ncol(G)  # Number of genetic covariates
  K <- n_topics   # Number of topics
  
  # Initialize parameters using mcmc_init_two
  init_state <- mcmc_init_two(Y, G, n_topics, length_scales_lambda, var_scales_lambda, length_scales_phi, var_scales_phi)
  
  Lambda <- init_state$Lambda
  Phi <- init_state$Phi
  Gamma <- init_state$Gamma
  
  # Set up Gaussian Process priors
  K_lambda <- lapply(1:K, function(k) {
    time_diff_matrix <- outer(1:Ttot, 1:Ttot, "-") ^ 2
    var_scales_lambda[k] * exp(-0.5 * time_diff_matrix / length_scales_lambda[k]^2) + diag(1e-6, Ttot)
  })
  K_phi <- lapply(1:K, function(k) {
    time_diff_matrix <- outer(1:Ttot, 1:Ttot, "-") ^ 2
    var_scales_phi[k] * exp(-0.5 * time_diff_matrix / length_scales_phi[k]^2) + diag(1e-6, Ttot)
  })
  
  K_inv_lambda <- lapply(K_lambda, solve)
  K_inv_phi <- lapply(K_phi, solve)
  
  # Storage for samples and diagnostics
  samples <- list(
    Lambda = array(0, dim = c(n_iters, dim(Lambda))),
    Phi = array(0, dim = c(n_iters, dim(Phi))),
    Gamma = array(0, dim = c(n_iters, dim(Gamma)))
  )
  log_likelihoods <- numeric(n_iters)
  log_priors_lambda <- numeric(n_iters)
  log_priors_phi <- numeric(n_iters)
  
  # Main MCMC loop
  for (iter in 1:n_iters) {
    # Compute log-priors
    log_prior_lambda <- 0
    log_prior_phi <- 0
    for (k in 1:K) {
      log_prior_lambda <- log_prior_lambda + dmvnorm(as.vector(Lambda[,,k]), mean = rep(0, N*Ttot), sigma = K_lambda[[k]], log = TRUE)
      log_prior_phi <- log_prior_phi + dmvnorm(as.vector(Phi[k,,]), mean = rep(0, D*Ttot), sigma = K_phi[[k]], log = TRUE)
    }
    log_priors_lambda[iter] <- log_prior_lambda
    log_priors_phi[iter] <- log_prior_phi

    # Update Lambda
    for (i in 1:N) {
      for (k in 1:K) {
        for (t in 1:Ttot) {
          Lambda <- update_lambda(Lambda, Phi, Y, K_inv_lambda[[k]], i, k, t, step_size_lambda)
        }
      }
    }
    
    # Update Phi
    for (k in 1:K) {
      for (d in 1:D) {
        for (t in 1:Ttot) {
          Phi <- update_phi(Lambda, Phi, Y, K_inv_phi[[k]], k, d, t, step_size_phi)
        }
      }
    }
    
    # Update Gamma using Gibbs sampler
    for (k in 1:K) {
      Lambda_k <- Lambda[, k, ]  # N x T matrix for topic k
      K_inv <- K_inv_lambda[[k]]  # T x T inverse covariance matrix
      
      posterior_precision <- diag(1, P) + t(G) %*% K_inv %*% G
      posterior_mean <- solve(posterior_precision, t(G) %*% K_inv %*% colMeans(Lambda_k))
      
      Gamma[k, ] <- mvrnorm(1, mu = posterior_mean, Sigma = solve(posterior_precision))
    }
    
    # Store samples
    samples$Lambda[iter, , , ] <- Lambda
    samples$Phi[iter, , , ] <- Phi
    samples$Gamma[iter, , ] <- Gamma
    
    # Compute log-likelihood
    log_lik <- 0
    for (i in 1:N) {
      for (d in 1:D) {
        for (t in 1:Ttot) {
          pi_idt <- sum(softmax(Lambda[i,,t]) * sigmoid(Phi[,d,t]))
          log_lik <- log_lik + Y[i,d,t] * log(pi_idt) + (1 - Y[i,d,t]) * log(1 - pi_idt)
        }
      }
    }
    log_likelihoods[iter] <- log_lik
    
    # Print progress
    if (iter %% 100 == 0) {
      cat("Iteration", iter, "Log-likelihood:", log_lik, 
          "Log-prior Lambda:", log_prior_lambda, 
          "Log-prior Phi:", log_prior_phi, "\n")
    }
  }
  
  return(list(
    samples = samples,
    log_likelihoods = log_likelihoods,
    log_priors_lambda = log_priors_lambda,
    log_priors_phi = log_priors_phi,
    init_state = init_state
  ))
}

# Example usage
set.seed(123)
N <- 100  # Number of individuals
D <- 5    # Number of diseases
Ttot <- 10  # Number of time points
P <- 3    # Number of genetic covariates
n_topics <- 3

# Generate synthetic data
Y <- array(rbinom(N * D * Ttot, 1, 0.5), dim = c(N, D, Ttot))
G <- matrix(rnorm(N * P), nrow = N, ncol = P)

# Define length scales and variance scales
length_scales_lambda <- rep(1, n_topics)
var_scales_lambda <- rep(1, n_topics)
length_scales_phi <- rep(1, n_topics)
var_scales_phi <- rep(1, n_topics)

# Run the model
result <- aladynoulli(Y, G, n_topics = n_topics, n_iters = 1000,
                      length_scales_lambda = length_scales_lambda,
                      var_scales_lambda = var_scales_lambda,
                      length_scales_phi = length_scales_phi,
                      var_scales_phi = var_scales_phi)

# Plot log-likelihood trace
plot(result$log_likelihoods, type = "l", xlab = "Iteration", ylab = "Log-likelihood")