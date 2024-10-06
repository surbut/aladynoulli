## model specific functions

# Log-likelihood function
log_likelihood <- function(y, Lambda, Phi) {
  n_individuals <- dim(Lambda)[1]
  n_topics <- dim(Lambda)[2]
  n_diseases <- dim(Phi)[2]
  T <- dim(Lambda)[3]
  
  theta <- apply_softmax_to_lambda(Lambda) # Apply softmax to Lambda
  pi <- array(0, dim = c(n_individuals, n_diseases, T))
  
  for (t in 1:T) {
    pi[, , t] <- theta[, , t] %*% logistic(Phi[, , t])
  }
  
  log_lik <- 0
  for (i in 1:n_individuals) {
    for (d in 1:n_diseases) {
      at_risk <- which(cumsum(y[i, d, ]) == 0)
      if (length(at_risk) > 0) {
        event_time <- max(at_risk) + 1
        if (event_time <= T) {
          log_lik <- log_lik + log(pi[i, d, event_time])
        }
        log_lik <- log_lik + sum(log(1 - pi[i, d, at_risk]))
      } else {
        log_lik <- log_lik + log(pi[i, d, 1])
      }
    }
  }
  return(log_lik)
}


precompute_K_inv <- function(T, length_scale, var_scale) {
  time_diff_matrix <- outer(1:T, 1:T, "-")^2
  Kern <- var_scale * exp(-0.5 * time_diff_matrix / length_scale^2)
  Kern <- Kern + diag(1e-6, T)  # Add small jitter for numerical stability
  K_inv <- solve(Kern)
  log_det_K <- determinant(Kern, logarithm = TRUE)$modulus
  cat("K_inv diagonal:", diag(K_inv)[1:5], "log_det_K:", log_det_K, "\n")  # Add this line
  return(list(K_inv = K_inv, log_det_K = log_det_K))
}

log_gp_prior_vec <- function(eta, mean, K_inv, log_det_K) {
  T <- length(eta)
  centered_eta <- eta - mean
  quad_form <- sum(centered_eta * (K_inv %*% centered_eta))
  log_prior <- -0.5 * (log_det_K + quad_form + T * log(2 * pi))
  cat("log_det_K:", log_det_K, "quad_form:", quad_form, "T:", T, "log_prior:", log_prior, "\n")
  return(log_prior)
}


update_phi <- function(Phi, k, d, new_values) {
  Phi_copy <- Phi
  Phi_copy[k, d, ] <- new_values
  return(Phi_copy)
}


update_lambda <- function(Lambda, i, k, new_values) {
  Lambda_copy <- Lambda
  Lambda_copy[i, k, ] <- new_values
  return(Lambda_copy)
}


