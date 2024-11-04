install.packages("Rcpp")
install.packages("RcppArmadillo")
Rcpp::sourceCpp("C:/Users/saraa/Desktop/stat-600-hw6-mohit-sara/src/LRMultiClass.cpp")


library(Rcpp)
library(RcppArmadillo)

# Example usage in R
X <- matrix(runif(100), nrow = 10, ncol = 10)
y <- sample(0:2, 10, replace = TRUE)  # 3 classes
beta_init <- matrix(0, ncol = 3, nrow = 10)

result <- LRMultiClass_c(X, y, beta_init, numIter = 100, eta = 0.05, lambda = 0.1)
print(result)

#test 2

# Generate 1000 samples with 20 features
n_samples <- 1000
n_features <- 20
n_classes <- 5  # Multiclass problem with 5 classes

X <- matrix(rnorm(n_samples * n_features), nrow = n_samples, ncol = n_features)

# Add a column of ones for intercept (bias term)
X <- cbind(1, X)  # Now X has 21 columns

# Generate random class labels (0 to 4)
y <- sample(0:(n_classes - 1), n_samples, replace = TRUE)

# Initialize beta with zeros (21 x 5 matrix)
beta_init <- matrix(0, nrow = n_features + 1, ncol = n_classes)

# Run the logistic regression model
result <- LRMultiClass_c(X, y, beta_init, numIter = 100, eta = 0.01, lambda = 0.1)

# Print the results
print(result$beta)  # The optimized beta coefficients
print(result$objective)  # Objective values over iterations

# Plot the objective to see convergence
plot(result$objective, type = "l", main = "Objective Function Over Iterations",
     xlab = "Iteration", ylab = "Objective Value")

#test 3

# Set a random seed for reproducibility
set.seed(123)

# Creating sample data for testing
n = 100  # Number of samples per class
p = 2    # Number of features

# Class 1: Mean (2, 2)
X1 = matrix(rnorm(n * p, mean = 2), ncol = p)
y1 = rep(0, n)  # Class 0 labels

# Class 2: Mean (-2, -2)
X2 = matrix(rnorm(n * p, mean = -2), ncol = p)
y2 = rep(1, n)  # Class 1 labels

# Combine data into one dataset
X = rbind(X1, X2)
y = c(y1, y2)

# Add intercept (column of 1's) to X
X = cbind(1, X)

# Splitting train/test sets
trainIdx = sample(1:(2 * n), size = 0.7 * 2 * n)  # 70% train, 30% test
X_train = X[trainIdx, ]
y_train = y[trainIdx]

X_test = X[-trainIdx, ]
y_test = y[-trainIdx]

# Initialize beta (number of features + 1, number of classes)
beta_init = matrix(0, nrow = ncol(X_train), ncol = 2)  # 2 classes

# Run multi-class logistic regression using the C++ function
result = LRMultiClass_c(X_train, y_train, beta_init, numIter = 50, eta = 0.1, lambda = 1)

# Plot the objective function over iterations
plot(result$objective, type = "b", col = "blue", 
     xlab = "Iteration", ylab = "Objective Value", 
     main = "Objective Value Over Iterations")

# Prediction function
predict_class <- function(X, beta) {
  probs = exp(X %*% beta)  # Compute probabilities
  probs = probs / rowSums(probs)  # Normalize to sum to 1
  apply(probs, 1, which.max) - 1  # Predict class (subtract 1 for zero-indexed labels)
}

# Compute final training error
train_pred = predict_class(X_train, result$beta)
train_error = mean(train_pred != y_train) * 100
cat("Training error:", train_error, "%\n")

# Compute final test error
test_pred = predict_class(X_test, result$beta)
test_error = mean(test_pred != y_test) * 100
cat("Test error:", test_error, "%\n")

# Check if the objective function decreases over iterations
decreasing = all(diff(result$objective) <= 0)
if (decreasing) {
  cat("The objective function decreases across iterations.\n")
} else {
  cat("The objective function does not consistently decrease!\n")
}

# Additional test with different hyperparameters
result2 = LRMultiClass_c(X_train, y_train, beta_init, numIter = 50, eta = 0.05, lambda = 0.5)

# Display new objective values
cat("New Objective values with eta = 0.05 and lambda = 0.5:\n", result2$objective, "\n")

