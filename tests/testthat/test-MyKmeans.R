X <- matrix(rnorm(7000 * 256, 0, 5), nrow = 7000)
M1 <- matrix(rnorm(15 * 256, 0, 2), nrow = 15)
M2 <- matrix(rnorm(15 * 256, 1, 3), nrow = 15)
K <- 15
numIter <- 100

test_that("Test 1: MyKmeans Function works!", {
  Y_MyKmeans_WithM1 <- MyKmeans(X, K, M1, numIter)
  expect_equal(length(Y_MyKmeans_WithM1), nrow(X))
  expect_equal(length(unique(Y_MyKmeans_WithM1)), K)
})

test_that("Test 2: MyKmeans Function works!", {
  Y_MyKmeans_WithM2 <- MyKmeans(X, K, M2, numIter)
  expect_equal(length(Y_MyKmeans_WithM2), nrow(X))
  expect_equal(length(unique(Y_MyKmeans_WithM2)), nrow(M2))
})

test_that("Test 3: MyKmeans Function works!", {
  Y_MyKmeans_WithoutM <- MyKmeans(X, K, M = NULL, numIter)
  expect_equal(length(Y_MyKmeans_WithoutM), nrow(X))
  expect_equal(length(unique(Y_MyKmeans_WithoutM)), K)
})

test_that("Test 4: MyKmeans Function works!", {
  time <- suppressWarnings(microbenchmark(
    MyKmeans(X, K, M1, numIter),
    MyKmeans(X, K, M2, numIter),
    MyKmeans(X, K, M = NULL, numIter),
    times = 10
  ))
  expect_equal(length(time$time), 30)
  expect_equal(dim(time), c(30, 2))
})

test_that("Test 5: MyKmeans Function throws error due to non-numeric element in X!", {
  X1 <- X
  X1[5000, 223] <- 'a'
  expect_error(MyKmeans(X1, K, M1, numIter))
})