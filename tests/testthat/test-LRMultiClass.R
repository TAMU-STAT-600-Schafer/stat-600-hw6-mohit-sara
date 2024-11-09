# Run test with only beta returned
test_that("Test with only beta returned", {
  X <- cbind(1, matrix(rnorm(100 * 4), nrow = 100))
  y <- as.integer(runif(100, 0, 3))
  beta_init <- matrix(0, nrow = ncol(X), ncol = length(unique(y)))
  result <- LRMultiClass(X, y, beta_init = beta_init)
  
  # Check that beta is present and has expected dimensions
  expect_equal(dim(result$beta), c(ncol(X), length(unique(y))))
})

# Run test with only objective returned
test_that("Test with only objective returned", {
  X <- cbind(1, matrix(rnorm(100 * 4), nrow = 100))
  y <- as.integer(runif(100, 0, 3))
  beta_init <- matrix(0, nrow = ncol(X), ncol = length(unique(y)))
  result <- LRMultiClass(X, y, beta_init = beta_init)
  
  # Check that objective is numeric and has multiple entries
  expect_true(is.numeric(result$objective))
  expect_true(length(result$objective) > 1)
})

# Run test with both beta and objective returned
test_that("Test with both beta and objective returned", {
  X <- cbind(1, matrix(rnorm(100 * 4), nrow = 100))
  y <- as.integer(runif(100, 0, 3))
  beta_init <- matrix(0, nrow = ncol(X), ncol = length(unique(y)))
  result <- LRMultiClass(X, y, beta_init = beta_init)
  
  # Ensure both components are as expected
  expect_true("beta" %in% names(result))
  expect_true("objective" %in% names(result))
  expect_equal(dim(result$beta), c(ncol(X), length(unique(y))))
  expect_true(length(result$objective) > 1)
})

# Test: LRMultiClass function returns correct dimensions and decreases objective
test_that("LRMultiClass Function returns correct dimensions and decreases objective", {
  X <- cbind(1, matrix(rnorm(100 * 4), nrow = 100))
  y <- as.integer(runif(100, 0, 3))
  beta_init <- matrix(0, nrow = ncol(X), ncol = length(unique(y)))
  
  result <- LRMultiClass(X, y, beta_init = beta_init)
  
  # Check that both beta and objective are in the output
  expect_true("beta" %in% names(result))
  expect_true("objective" %in% names(result))
  
  # Check dimensions of beta
  expect_equal(dim(result$beta), c(ncol(X), length(unique(y))))
  
  # Check that objective is non-increasing
  expect_true(all(diff(result$objective) <= 0))
})

# Test 1: Check if LRMultiClass runs and returns correct dimensions
test_that("Test 1: LRMultiClass runs and returns correct dimensions", {
  X <- matrix(c(1, 1, 1, 1, 1, 2, 3, 4, 5, 6), ncol = 2)
  y <- c(0, 1, 0, 1, 0)
  beta_init <- matrix(rnorm(2 * 2), nrow = 2)
  
  result <- LRMultiClass(X, y, numIter = 50, eta = 0.1, lambda = 1, beta_init = beta_init)
  
  expect_true(is.list(result))
  expect_true("beta" %in% names(result))
  expect_equal(dim(result$beta), c(ncol(X), length(unique(y))))
})

# Test 2: Objective function decreases with iterations
test_that("Test 2: Objective function decreases with iterations", {
  X <- matrix(c(1, 1, 1, 1, 1, 2, 3, 4, 5, 6), ncol = 2)
  y <- c(0, 1, 0, 1, 0)
  beta_init <- matrix(rnorm(2 * 2), nrow = 2)
  
  result <- LRMultiClass(X, y, numIter = 50, eta = 0.1, lambda = 1, beta_init = beta_init)
  
  expect_true(all(diff(result$objective) <= 0))
})

# Test 3: Regularization effect on beta values
test_that("Test 3: Regularization impact on beta", {
  X <- matrix(c(1, 1, 1, 1, 1, 2, 3, 4, 5, 6), ncol = 2)
  y <- c(0, 1, 0, 1, 0)
  beta_init <- matrix(rnorm(2 * 2), nrow = 2)
  
  result_no_reg <- LRMultiClass(X, y, numIter = 50, eta = 0.1, lambda = 0, beta_init = beta_init)
  result_with_reg <- LRMultiClass(X, y, numIter = 50, eta = 0.1, lambda = 1, beta_init = beta_init)
  
  expect_true(sum(abs(result_no_reg$beta)) > sum(abs(result_with_reg$beta)))
})

# Test 4: Input validation - intercept column
test_that("Test 4: Error if intercept column is missing", {
  X <- matrix(c(2, 3, 4, 5, 6), ncol = 1)
  y <- c(0, 1, 0, 1, 0)
  
  expect_error(LRMultiClass(X, y), "First column of X must be all ones for intercept.")
})

# Test 5: Input validation - dimensions of X and y
test_that("Test 5: Error if dimensions of X and y do not match", {
  X <- matrix(c(1, 1, 1, 1, 2, 3, 4, 5), ncol = 2)
  y <- c(0, 1, 0, 1, 0)
  
  expect_error(LRMultiClass(X, y), "Number of rows in X must match length of y.")
})

# Test 6: Input validation - positive learning rate
test_that("Test 6: Error if learning rate (eta) is not positive", {
  X <- matrix(c(1, 1, 1, 1, 1, 2, 3, 4, 5, 6), ncol = 2)
  y <- c(0, 1, 0, 1, 0)
  beta_init <- matrix(rnorm(2 * 2), nrow = 2)
  
  expect_error(LRMultiClass(X, y, eta = -0.1, beta_init = beta_init))
})

# Test 7: Input validation - non-negative regularization parameter
test_that("Test 7: Error if regularization parameter (lambda) is negative", {
  X <- matrix(c(1, 1, 1, 1, 1, 2, 3, 4, 5, 6), ncol = 2)
  y <- c(0, 1, 0, 1, 0)
  beta_init <- matrix(rnorm(2 * 2), nrow = 2)
  
  expect_error(LRMultiClass(X, y, lambda = -1, beta_init = beta_init))
})

# Test 8: Performance benchmark
test_that("Test 8: LRMultiClass performance benchmark", {
  X <- matrix(rnorm(500 * 10), nrow = 500)
  X[, 1] <- 1
  y <- sample(0:4, 500, replace = TRUE)
  beta_init <- matrix(rnorm(10 * 5), nrow = 10)
  
  time <- suppressWarnings(microbenchmark(
    LRMultiClass(X, y, numIter = 10, eta = 0.1, lambda = 1, beta_init = beta_init),
    times = 5
  ))
  
  expect_equal(length(time$time), 5)
  expect_equal(dim(time), c(5, 2))
})
