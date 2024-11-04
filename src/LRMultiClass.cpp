
// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for RcppArmadillo
//
// [[Rcpp::depends(RcppArmadillo)]]

// For simplicity, no test data, only training data, and no error calculation.
// X - n x p data matrix
// y - n length vector of classes, from 0 to K-1
// numIter - number of iterations, default 50
// eta - damping parameter, default 0.1
// lambda - ridge parameter, default 1
// beta_init - p x K matrix of starting beta values (always supplied in correct format)

// [[Rcpp::export]]
Rcpp::List LRMultiClass_c(const arma::mat& X, const arma::uvec& y, const arma::mat& beta_init,
                          int numIter = 50, double eta = 0.1, double lambda = 1) {
  int K = arma::max(y) + 1;
  int p = X.n_cols;
  int n = X.n_rows;
  
  arma::mat beta = beta_init;
  arma::vec objective(numIter + 1, arma::fill::zeros);
  
  auto calculateProbs = [](const arma::mat& X, const arma::mat& beta) {
    arma::mat linearComb = X * beta;
    arma::vec maxVals = arma::max(linearComb, 1);
    linearComb.each_col() -= maxVals;
    arma::mat expXB = arma::exp(linearComb);
    return expXB.each_col() / arma::sum(expXB, 1);
  };
  
  auto calcObjective = [&](const arma::mat& P, const arma::uvec& y, const arma::mat& beta, double lambda) {
    double logLikelihood = 0;
    for (int i = 0; i < n; ++i) {
      logLikelihood -= std::log(P(i, y(i)) + 1e-10);
    }
    double regularization = 0.5 * lambda * arma::accu(beta % beta);
    return logLikelihood + regularization;
  };
  
  arma::mat prob_train = calculateProbs(X, beta);
  objective(0) = calcObjective(prob_train, y, beta, lambda);
  
  for (int t = 0; t < numIter; ++t) {
    for (int k = 0; k < K; ++k) {
      arma::vec Pk = prob_train.col(k);
      arma::vec gradient = X.t() * (Pk - arma::conv_to<arma::vec>::from(y == k)) + lambda * beta.col(k);
      arma::vec WkDiag = Pk % (1 - Pk);
      arma::mat hessian = X.t() * (X.each_col() % WkDiag) + lambda * arma::eye(p, p);
      hessian += 1e-6 * arma::eye(p, p);
      arma::vec deltaBeta = arma::solve(hessian, gradient, arma::solve_opts::fast);
      beta.col(k) -= eta * deltaBeta;
    }
    prob_train = calculateProbs(X, beta);
    objective(t + 1) = calcObjective(prob_train, y, beta, lambda);
  }
  
  // Debug output
  Rcpp::Rcout << "Beta dimensions: " << beta.n_rows << " x " << beta.n_cols << std::endl;
  Rcpp::Rcout << "Objective length: " << objective.size() << std::endl;
  
  // Testing return of each component
  Rcpp::NumericMatrix betaR = Rcpp::wrap(beta);
  Rcpp::NumericVector objectiveR = Rcpp::wrap(objective);
  
  // Step-by-step testing
  // return Rcpp::List::create(Rcpp::Named("beta") = betaR); // Uncomment for beta only
  // return Rcpp::List::create(Rcpp::Named("objective") = objectiveR); // Uncomment for objective only
  return Rcpp::List::create(Rcpp::Named("beta") = betaR,
                            Rcpp::Named("objective") = objectiveR); // Return both if both work independently
  }


