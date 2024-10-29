install.packages("Rcpp")
install.packages("RcppArmadillo")
install.packages("testthat")


Rcpp::sourceCpp("C:/Users/saraa/Desktop/stat-600-hw6-mohit-sara/src/LRMultiClass.cpp")


library(Rcpp)
library(RcppArmadillo)
library(testthat)


# Test 1: Basic functionality
test_that("Function runs without errors", {
  X <- matrix(rnorm(20), nrow = 5, ncol = 4)  # 5 data points, 4 features
  y <- as.integer(c(0, 1, 0, 2, 1))           # 3 classes (0, 1, 2)
  beta_init <- matrix(0, nrow = 4, ncol = 3)  # 4 features, 3 classes
  
  result <- LRMultiClass_c(X, y, beta_init)
  
  expect_true(is.list(result))
  expect_true("beta" %in% names(result))
  expect_true("objective" %in% names(result))
})




# Test 2: Correct output dimensions
test_that("Output dimensions are correct", {
  X <- matrix(rnorm(50), nrow = 10, ncol = 5)  # 10 data points, 5 features
  y <- as.integer(c(0, 1, 0, 2, 1, 0, 2, 2, 1, 0))  # 3 classes
  beta_init <- matrix(0, nrow = 5, ncol = 3)   # 5 features, 3 classes
  
  result <- LRMultiClass_c(X, y, beta_init)
  beta <- result$beta
  objective <- result$objective
  
  expect_equal(dim(beta), c(5, 3))  # Check if beta has correct dimensions
  expect_equal(length(objective), 51)  # Should be numIter + 1
})



# Test 3: Objective decreases over iterations
test_that("Objective function decreases", {
  X <- matrix(rnorm(30), nrow = 10, ncol = 3)
  y <- as.integer(c(0, 1, 2, 0, 1, 2, 0, 1, 2, 0))
  beta_init <- matrix(0, nrow = 3, ncol = 3)
  
  result <- LRMultiClass_c(X, y, beta_init, numIter = 100)
  objective <- result$objective
  
  # Check if the objective decreases or stays constant
  expect_true(all(diff(objective) <= 0))
})

