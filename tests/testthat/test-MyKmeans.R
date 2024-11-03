X <- matrix(rnorm(7000 * 256, 0, 5), nrow = 7000)
M1 <- matrix(rnorm(15 * 256, 0, 2), nrow = 15)
M2 <- matrix(rnorm(15 * 256, 1, 3), nrow = 15)
K <- 15
numIter <- 100

test_that("Function works!", {
  Y_MyKmeans_WithM1 <- MyKmeans(X, K, M1, numIter)
  expect_equal(length(Y_MyKmeans_WithM1), nrow(X))
  expect_equal(length(unique(Y_MyKmeans_WithM1)), K)
})

