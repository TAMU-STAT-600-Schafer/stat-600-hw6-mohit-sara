X <- matrix(rnorm(7000 * 256, 0, 5), nrow = 7000)
M3 <- matrix(rnorm(15 * 256, -1, 2), nrow = 15)
M4 <- matrix(runif(15 * 256), nrow = 15)
K <- 15
numIter <- 100

test_that("Test 1: MyKmeans_c Function works!", {
  Y_MyKmeans_WithM3 <- MyKmeans(X, K, M3, numIter)
  expect_equal(length(Y_MyKmeans_WithM3), nrow(X))
  expect_equal(length(unique(Y_MyKmeans_WithM3)), K)
})
