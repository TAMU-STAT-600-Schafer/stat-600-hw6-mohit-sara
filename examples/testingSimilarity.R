install.packages("Rcpp")
install.packages("RcppArmadillo")
Rcpp::sourceCpp("C:/Users/saraa/Desktop/stat-600-hw6-mohit-sara/src/LRMultiClass.cpp")
source("C:/Users/saraa/Desktop/stat-600-hw6-mohit-sara/R/LR_wrapper.R")


source("C:/Users/saraa/Desktop/stat-600-hw6-mohit-sara/examples/LRMulticlassO.R")

library(Rcpp)
library(RcppArmadillo)




# Generate synthetic data for testing
n <- 1000  # Dataset size
p <- 10    # Number of predictors (including intercept)
K <- 3     # Number of classes

# Design matrix with an intercept column (all ones in the first column)
X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))  
y <- sample(0:(K - 1), n, replace = TRUE)  # Random class labels
Xt <- X  # Use the same data as test data for simplicity
yt <- y

# Define parameters
numIter <- 50
eta <- 0.1
lambda <- 1
beta_init <- matrix(0, nrow = p, ncol = K)

# Run the functions with the same inputs
result_LRogFunction <- LRMultiClassO(X, y, Xt, yt, numIter = numIter, eta = eta, lambda = lambda, beta_init = beta_init)
result_LR_wrapper <- LRMultiClass(X, y, numIter = numIter, eta = eta, lambda = lambda, beta_init = beta_init)

# Print the beta matrices
cat("Beta from LRogFunction.R:\n")
print(result_LRogFunction$beta)
cat("\nBeta from LR_wrapper.R:\n")
print(result_LR_wrapper$beta)

# Print the objective values
cat("\nObjective values from LRogFunction.R:\n")
print(result_LRogFunction$objective)
cat("\nObjective values from LR_wrapper.R:\n")
print(result_LR_wrapper$objective)

# Compare beta and objective values
beta_diff <- max(abs(result_LRogFunction$beta - result_LR_wrapper$beta))
objective_diff <- max(abs(result_LRogFunction$objective - result_LR_wrapper$objective))

# Print max differences
cat("\nMax difference in beta matrices:", beta_diff, "\n")
cat("Max difference in objective values:", objective_diff, "\n")

# Check if the results are similar
if (beta_diff < 1e-5 && objective_diff < 1e-5) {
  cat("\nThe results from LRogFunction.R and LR_wrapper.R are similar.\n")
} else {
  cat("\nThe results from LRogFunction.R and LR_wrapper.R differ. Consider further investigation.\n")
}