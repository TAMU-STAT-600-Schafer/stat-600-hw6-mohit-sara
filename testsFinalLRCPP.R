install.packages("Rcpp")
install.packages("RcppArmadillo")

library(Rcpp)
library(RcppArmadillo)

path <- normalizePath("C:/Users/saraa/Desktop/stat-600-hw6-mohit-sara/src/LRMultiClass.cpp", mustWork = TRUE)
Rcpp::sourceCpp(path)


source("C:/Users/saraa/Desktop/stat-600-hw6-mohit-sara/R/LR_wrapper.R")

# Test 1: Simulated example with 3 classes and 2 predictors
set.seed(123)

# Generate data
n <- 100  # Number of samples
p <- 3    # Number of predictors (including intercept)
K <- 3    # Number of classes

# Design matrix with an intercept column (all ones in the first column)
X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))  
y <- sample(0:(K - 1), n, replace = TRUE)  # Random class labels

# Ensure that the data is correctly formatted
cat("Is X a matrix? ", is.matrix(X), "\n")  # Should return TRUE
cat("Is y a vector? ", is.vector(y), "\n")  # Should return TRUE

# Optional: Define beta_init explicitly (optional; function can also initialize it internally)
beta_init <- matrix(0, nrow = p, ncol = K)  # Initialize to zeros

# Check if beta_init is a matrix
cat("Is beta_init a matrix? ", is.matrix(beta_init), "\n")  # Should return TRUE

# Run the LRMultiClass function (with or without providing beta_init)
result <- LRMultiClass_c(X, y, numIter = 50, eta = 0.1, lambda = 1, beta_init = beta_init)

# Display the results
cat("Beta matrix:\n")
print(result$beta)

cat("\nObjective function values:\n")
print(result$objective)

# Plot the objective function values over iterations
plot(result$objective, type = "b", col = "blue", 
     xlab = "Iteration", ylab = "Objective Value", 
     main = "Objective Value Over Iterations")

# Check if the objective function values are decreasing
decreasing <- all(diff(result$objective) <= 0)
if (decreasing) {
  cat("The objective function decreases across iterations.\n")
} else {
  cat("The objective function does not consistently decrease!\n")
}



# Test 1
# Simulated example with 3 classes and 2 predictors
set.seed(123)

# Generate data
n <- 100  # Number of samples
p <- 3    # Number of predictors (including intercept)
K <- 3    # Number of classes

X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))  # Include intercept column
y <- sample(0:(K - 1), n, replace = TRUE)  # Random class labels

is.matrix(X)       # Should return TRUE
is.vector(y)       # Should return TRUE
is.matrix(beta_init) # Should return TRUE

X <- as.matrix(X)
beta_init <- as.matrix(beta_init)
y <- as.integer(y)  # Ensure 'y' is an integer vector



# Initialize beta (optional)
beta_init <- matrix(0, nrow = p, ncol = K)

# Fit the model
result <- LRMultiClass(X, y, numIter = 100, eta = 0.1, lambda = 1, beta_init = beta_init)
#result <- LRMultiClass_c(X, y, beta_init, numIter = 100, eta = 0.1, lambda = 1)

# Print results
print(result$beta)      # Coefficients
plot(result$objective)  # Objective function over iterations




#Test to reflect tests done in HW 3

# Creating sample data for testing
n = 100  # Number of observations per class
p = 2    # Number of features

# Class 1: Mean (2, 2)
X1 = matrix(rnorm(n * p, mean = 2), ncol = p)
y1 = rep(0, n)

# Class 2: Mean (-2, -2)
X2 = matrix(rnorm(n * p, mean = -2), ncol = p)
y2 = rep(1, n)

# Combine data
X = rbind(X1, X2)
y = c(y1, y2)

# Add intercept (column of 1's)
X = cbind(1, X)

# Run the multi-class logistic regression with modified function
result = LRMultiClass(X, y, numIter = 50, eta = 0.1, lambda = 1)

# Extract and display the results
cat("Beta matrix:\n")
print(result$beta)

cat("\nObjective function values:\n")
print(result$objective)

# Plot the objective function values over iterations
plot(result$objective, type = "b", col = "blue", 
     xlab = "Iteration", ylab = "Objective Value", 
     main = "Objective Value Over Iterations")

# Check if the objective function is decreasing
decreasing = all(diff(result$objective) <= 0)
if (decreasing) {
  cat("The objective function decreases across iterations.\n")
} else {
  cat("The objective function does not consistently decrease!\n")
}

# Additional example with different parameters
result2 = LRMultiClass(X, y, numIter = 50, eta = 0.05, lambda = 0.5)

cat("\nNew Beta matrix with eta = 0.05 and lambda = 0.5:\n")
print(result2$beta)

cat("\nNew Objective values with eta = 0.05 and lambda = 0.5:\n")
print(result2$objective)

# Plot the new objective function values for comparison
plot(result2$objective, type = "b", col = "green", 
     xlab = "Iteration", ylab = "Objective Value", 
     main = "Objective Value (eta=0.05, lambda=0.5)")



# Benchmarking check

install.packages("microbenchmark")

library(microbenchmark)

# Load necessary libraries and source your function
source("FunctionsLR.R")  # Assuming LRMultiClass is defined here

# Set seed for reproducibility
set.seed(123)

# Generate synthetic data for benchmarking
n <- 1000  # Larger dataset for benchmarking purposes
p <- 10    # Number of predictors (including intercept)
K <- 3     # Number of classes

# Design matrix with an intercept column (all ones in the first column)
X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))  
y <- sample(0:(K - 1), n, replace = TRUE)  # Random class labels

# Initialize beta matrix
beta_init <- matrix(0, nrow = p, ncol = K)  # Zero-initialization

# Benchmark the function with microbenchmark
benchmark_result <- microbenchmark(
  LRMultiClass(X, y, numIter = 50, eta = 0.1, lambda = 1, beta_init = beta_init),
  LRMultiClass(X, y, numIter = 100, eta = 0.05, lambda = 0.5, beta_init = beta_init),
  times = 10  # Number of repetitions for each configuration
)

# Display benchmark results
print(benchmark_result)


