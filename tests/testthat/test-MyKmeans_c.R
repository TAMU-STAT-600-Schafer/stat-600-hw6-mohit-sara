X <- matrix(rnorm(7000 * 256, 0, 5), nrow = 7000)
M3 <- matrix(rnorm(15 * 256, -1, 2), nrow = 15)
M4 <- matrix(runif(15 * 256), nrow = 15)
K <- 15
numIter <- 100

test_that("Test 1: MyKmeans_c Function works!", {
  Y_MyKmeans_WithM3 <- MyKmeans_c(X, K, M3, numIter)
  expect_equal(length(Y_MyKmeans_WithM3), nrow(X))
  expect_equal(length(unique(Y_MyKmeans_WithM3)), K)
})

test_that("Test 1: MyKmeans_c Function works!", {
  Y_MyKmeans_WithM4 <- MyKmeans_c(X, K, M4, numIter)
  expect_equal(length(Y_MyKmeans_WithM4), nrow(X))
  expect_equal(length(unique(Y_MyKmeans_WithM4)), nrow(M4))
})

test_that("Test 3: MyKmeans_c Function works!", {
  time <- suppressWarnings(microbenchmark(
    MyKmeans_c(X, K, M3, numIter),
    MyKmeans_c(X, K, M4, numIter),
    times = 10
  ))
  expect_equal(length(time$time), 20)
  expect_equal(dim(time), c(20, 2))
})
