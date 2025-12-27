# ==============================================================================
# BHIFR Simulation Study 1: Homoscedastic Scenario
# Description: Exact Methodology Replication (Synced with Study 2)
#              Algorithm: Dense Matrix Kronecker + Logit Lambda + Component-wise Adaptive MALA
# ==============================================================================

rm(list=ls())
set.seed(123) # Global seed for reproducibility

# ==============================================================================
# 0. Environment Setup
# ==============================================================================
pkgs <- c("Rcpp", "RcppArmadillo", "splines2", "MASS", "dplyr", 
          "knitr", "foreach", "doSNOW", "parallel", "tidyr", 
          "kableExtra", "ggplot2", "patchwork", "ggsci", "stringr", "gridExtra")

# Install missing packages
new_pkgs <- pkgs[!(pkgs %in% installed.packages()[,"Package"])]
if(length(new_pkgs)) install.packages(new_pkgs)

# Load packages
invisible(lapply(pkgs, require, character.only = TRUE))

# Define plotting theme
theme_paper <- function() {
  theme_bw() +
    theme(
      text = element_text(family = "serif"),
      panel.grid.major = element_line(color = "grey90"),
      panel.grid.minor = element_blank(),
      axis.text = element_text(color="black", size=10),
      axis.title = element_text(color="black", size=12, face="bold"),
      legend.position = "bottom",
      legend.title = element_blank(),
      strip.background = element_rect(fill="grey95"),
      strip.text = element_text(face="bold", size=10),
      plot.title = element_blank() 
    )
}

# ==============================================================================
# 1. C++ Core Implementation (Synced with Study 2)
# ==============================================================================
cpp_code_string <- '
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <cmath>

using namespace arma;
using namespace Rcpp;

// Constants
const double nugget = 1e-6; 

// =============================================================================
// 1. Helper Functions 
// =============================================================================

double logit(double p) {
    if(p < 1e-10) p = 1e-10;
    if(p > 1.0 - 1e-10) p = 1.0 - 1e-10;
    return std::log(p / (1.0 - p));
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double fisher_z(double rho) { 
    if(rho > 0.995) rho = 0.995; if(rho < -0.995) rho = -0.995;
    return std::atanh(rho); 
}

double inv_fisher_z(double z) { 
    double r = std::tanh(z); if(r > 0.995) r = 0.995; if(r < -0.995) r = -0.995;
    return r;
}

// Calculate Marginal Log-Likelihood for Lambda (Collapsed Gibbs, Dense Version)
double calc_marginal_loglik_lambda_dense(const vec& lambda, const mat& X_L, const mat& X_U,
                                         const mat& BTWB, const mat& BTY_proj, const mat& Prior_Prec) {
    int n = X_L.n_rows; int q = X_L.n_cols;
    
    // Construct subject design matrix P
    mat P = ones(n, q + 1);
    for(int j=0; j<q; ++j) P.col(j+1) = (1.0 - lambda(j)) * X_L.col(j) + lambda(j) * X_U.col(j);
    
    mat PTP = P.t() * P;
    
    // Core: Kronecker Product Acceleration (Dense)
    mat Lambda_beta = kron(PTP, BTWB) + Prior_Prec; 
    
    // RHS = D^T Sigma^-1 Y = vec(BTY_proj * P)
    vec RHS = vectorise(BTY_proj * P);
    
    mat L;
    bool success = chol(L, Lambda_beta, "lower");
    if(!success) { // Fallback jitter if singular
         mat I(Lambda_beta.n_rows, Lambda_beta.n_cols, fill::eye);
         if(!chol(L, Lambda_beta + 1e-7 * I, "lower")) return -1e20;
    }
    
    // Log determinant: 2 * sum(log(diag(L)))
    double log_det = 2.0 * sum(log(L.diag()));
    
    // solve(L, RHS) solves L * y = RHS
    vec y = solve(trimatl(L), RHS);
    
    // Marginal LogLik = -0.5 * log|Lambda| + 0.5 * y^T y
    return -0.5 * log_det + 0.5 * dot(y, y);
}

double calc_joint_loglik(const mat& E_L, const mat& E_U, const vec& sL, const vec& sU, double rho) {
    int n = E_L.n_rows; int m = E_L.n_cols;
    double omr2 = 1.0 - rho*rho;
    
    double log_det = -n * (accu(log(sL)) + accu(log(sU))) - 0.5 * n * m * std::log(omr2);
    
    double quad = 0.0;
    for(int t=0; t<m; ++t) {
        double vL = sL(t); double vU = sU(t);
        double tL = accu(pow(E_L.col(t), 2)) / (vL*vL);
        double tU = accu(pow(E_U.col(t), 2)) / (vU*vU);
        double tLU = accu(E_L.col(t) % E_U.col(t)) / (vL*vU);
        quad += (tL + tU - 2.0 * rho * tLU);
    }
    return log_det - 0.5 / omr2 * quad;
}

// =============================================================================
// 2. Main BHIFR Homoscedastic Model Function
// =============================================================================
// [[Rcpp::export]]
List BHIFR_Homo_Cpp(mat Y_L, mat Y_U, mat X_L, mat X_U, mat B, mat Omega, 
                    mat X_L_test, mat X_U_test, // Internal prediction
                    int n_iter, int n_burnin) {
    
    int n = Y_L.n_rows; int m = Y_L.n_cols; int q = X_L.n_cols; int K = B.n_cols;
    int n_test = X_L_test.n_rows;
    int dim_beta = 2 * K * (q + 1);

    // --- Initialization ---
    vec lambda = randu(q);
    vec beta = zeros(dim_beta);
    
    mat P_init = ones(n, q + 1);
    for(int j=0; j<q; ++j) P_init.col(j+1) = 0.5 * X_L.col(j) + 0.5 * X_U.col(j);
    mat D_init = kron(P_init, B); 
    int dim_single = K * (q + 1);
    mat DTD = D_init.t() * D_init + 0.1 * eye(dim_single, dim_single);
    vec bL_init = solve(DTD, D_init.t() * vectorise(Y_L.t()));
    vec bU_init = solve(DTD, D_init.t() * vectorise(Y_U.t()));
    for(int j=0; j<=q; ++j) {
        beta.subvec(j*2*K, j*2*K+K-1)     = bL_init.subvec(j*K, j*K+K-1);
        beta.subvec(j*2*K+K, j*2*K+2*K-1) = bU_init.subvec(j*K, j*K+K-1);
    }
    
    double sigma2_L = 0.1, sigma2_U = 0.1, rho = 0.0;
    double nu_sigma_L = 1.0, nu_sigma_U = 1.0; 
    vec tau_beta = ones(2 * (q + 1)) * 10.0; 
    vec nu_beta = ones(2 * (q + 1));
    
    // --- Adaptive Step Size Variables ---
    vec step_l_vec = ones(q) * 1.0; 
    
    // --- Storage Containers ---
    int n_save = n_iter - n_burnin;
    mat res_beta(n_save, dim_beta); 
    mat res_lambda(n_save, q); 
    mat pred_L_sum = zeros(n_test, m); 
    mat pred_U_sum = zeros(n_test, m);
    
    // Pre-computed Matrices
    mat BTB = B.t()*B; mat BTY_L = B.t()*Y_L.t(); mat BTY_U = B.t()*Y_U.t();
    
    for(int it=0; it<n_iter; ++it) {
        // Robbins-Monro Adaptation Decay
        double gamma_t = std::min(1.0, 10.0 / (double(it) + 100.0));

        // Add Nugget
        double sL_val = std::sqrt(sigma2_L + nugget);
        double sU_val = std::sqrt(sigma2_U + nugget);
        
        double sc = 1.0/(1.0 - rho*rho);
        double w11=sc/(sL_val*sL_val); double w22=sc/(sU_val*sU_val); double w12=-sc*rho/(sL_val*sU_val);
        
        // Construct block-diagonal BTWB
        mat BTWB_blk = zeros(2*K, 2*K);
        BTWB_blk.submat(0,0,K-1,K-1)=w11*BTB; BTWB_blk.submat(K,K,2*K-1,2*K-1)=w22*BTB;
        BTWB_blk.submat(0,K,K-1,2*K-1)=w12*BTB; BTWB_blk.submat(K,0,2*K-1,K-1)=w12*BTB;
        
        mat BTY_proj = zeros(2*K, n);
        BTY_proj.rows(0,K-1) = w11*BTY_L + w12*BTY_U;
        BTY_proj.rows(K,2*K-1) = w12*BTY_L + w22*BTY_U;
        
        mat Prior_Prec = zeros(dim_beta, dim_beta);
        for(int g=0; g<2*(q+1); ++g) Prior_Prec.submat(g*K, g*K, g*K+K-1, g*K+K-1) = tau_beta(g) * Omega;
        
        double current_ll = calc_marginal_loglik_lambda_dense(lambda, X_L, X_U, BTWB_blk, BTY_proj, Prior_Prec);
        
        // ---------------------------------------------------------------------
        // 1. Update Lambda (Logit Transform + Adaptive MH)
        // ---------------------------------------------------------------------
        for(int j=0; j<q; ++j) {
            double eta_curr = logit(lambda(j));
            double eta_prop = eta_curr + step_l_vec(j) * randn();
            double lambda_prop_val = sigmoid(eta_prop);
            
            vec lambda_prop = lambda; 
            lambda_prop(j) = lambda_prop_val;
            
            double prop_ll = calc_marginal_loglik_lambda_dense(lambda_prop, X_L, X_U, BTWB_blk, BTY_proj, Prior_Prec);
            
            // Jacobian Adjustment: log(lambda*(1-lambda))
            double jacobian_diff = std::log(lambda_prop_val * (1.0 - lambda_prop_val)) - 
                                   std::log(lambda(j) * (1.0 - lambda(j)));
            
            double log_ratio = prop_ll - current_ll + jacobian_diff;
            bool accept = std::log(randu()) < log_ratio;
            
            if(accept) {
                lambda(j) = lambda_prop_val;
                current_ll = prop_ll;
            }
            
            // Adaptive Step Size
            if(it < n_burnin) {
                double alpha = accept ? 1.0 : 0.0;
                double log_step = std::log(step_l_vec(j)) + gamma_t * (alpha - 0.44);
                step_l_vec(j) = std::exp(log_step);
            }
        }
        
        // ---------------------------------------------------------------------
        // 2. Update Beta (Dense Kronecker Cholesky)
        // ---------------------------------------------------------------------
        mat P(n,q+1); P.col(0).fill(1.0); 
        for(int j=0; j<q; ++j) P.col(j+1)=(1-lambda(j))*X_L.col(j)+lambda(j)*X_U.col(j);
        mat PTP = P.t()*P;
        
        mat Prec = kron(PTP, BTWB_blk) + Prior_Prec;
        
        mat L_post;
        bool chol_ok = chol(L_post, Prec, "lower");
        if(!chol_ok) { 
             mat I(dim_beta, dim_beta, fill::eye);
             chol(L_post, Prec + 1e-7 * I, "lower");
        }
        
        vec RHS = vectorise(BTY_proj * P);
        vec z = randn(dim_beta);
        
        vec temp1 = solve(trimatl(L_post), RHS);
        vec mean_part = solve(trimatu(L_post.t()), temp1);
        vec temp2 = solve(trimatu(L_post.t()), z);
        beta = mean_part + temp2;
        
        // ---------------------------------------------------------------------
        // 3. Update Beta Hyperparameters (Scale Mixture)
        // ---------------------------------------------------------------------
        for(int g=0; g < 2*(q+1); ++g) {
             vec b_sub = beta.subvec(g*K, g*K+K-1);
             double quad = as_scalar(b_sub.t() * Omega * b_sub);
             tau_beta(g) = R::rgamma(0.5*K + 0.5, 1.0/(0.5*quad + 1.0/nu_beta(g)));
             
             double rate_nu_beta = 1.0/25.0 + tau_beta(g);
             nu_beta(g) = 1.0 / R::rgamma(1.0, 1.0 / rate_nu_beta); 
        }
        
        // ---------------------------------------------------------------------
        // 4. Update Sigma, Rho (Block MH) & Nu (Gibbs)
        // ---------------------------------------------------------------------
        // Calculate residuals
        mat YL_h(n,m), YU_h(n,m); YL_h.zeros(); YU_h.zeros();
        vec aL=beta.head(K); vec aU=beta.subvec(K, 2*K-1); 
        YL_h.each_row() += (B*aL).t(); YU_h.each_row() += (B*aU).t();
        for(int j=0;j<q;++j) { 
            vec bL=beta.subvec((j+1)*2*K, (j+1)*2*K+K-1); vec bU=beta.subvec((j+1)*2*K+K, (j+1)*2*K+2*K-1);
            for(int i=0;i<n;++i) { YL_h.row(i)+=P(i,j+1)*(B*bL).t(); YU_h.row(i)+=P(i,j+1)*(B*bU).t(); } 
        }
        mat EL=Y_L-YL_h; mat EU=Y_U-YU_h;
        
        vec theta_curr(3); 
        theta_curr(0)=log(sqrt(sigma2_L)); theta_curr(1)=log(sqrt(sigma2_U)); theta_curr(2)=fisher_z(rho);
        vec theta_prop = theta_curr + 0.02 * randn(3);
        
        double lsL_p=theta_prop(0); double lsU_p=theta_prop(1); double rho_p=inv_fisher_z(theta_prop(2));
        
        vec sL_c(m); sL_c.fill(std::sqrt(sigma2_L + nugget)); 
        vec sU_c(m); sU_c.fill(std::sqrt(sigma2_U + nugget));
        vec sL_p(m); sL_p.fill(std::sqrt(exp(2*lsL_p) + nugget)); 
        vec sU_p(m); sU_p.fill(std::sqrt(exp(2*lsU_p) + nugget));
        
        double lp_curr = calc_joint_loglik(EL,EU,sL_c,sU_c,rho) - theta_curr(0) - 1.0/(nu_sigma_L*exp(2*theta_curr(0))) - theta_curr(1) - 1.0/(nu_sigma_U*exp(2*theta_curr(1)));
        double lp_prop = calc_joint_loglik(EL,EU,sL_p,sU_p,rho_p) - theta_prop(0) - 1.0/(nu_sigma_L*exp(2*theta_prop(0))) - theta_prop(1) - 1.0/(nu_sigma_U*exp(2*theta_prop(1)));
        
        if(log(randu()) < (lp_prop - lp_curr + log(1-rho_p*rho_p) - log(1-rho*rho))) {
             sigma2_L = exp(2*lsL_p); sigma2_U = exp(2*lsU_p); rho = rho_p;
        }
        
        double rate_nu_sigma_L = 1.0/25.0 + 1.0/sigma2_L;
        nu_sigma_L = 1.0 / R::rgamma(1.0, 1.0 / rate_nu_sigma_L); 
        
        double rate_nu_sigma_U = 1.0/25.0 + 1.0/sigma2_U;
        nu_sigma_U = 1.0 / R::rgamma(1.0, 1.0 / rate_nu_sigma_U); 

        // ---------------------------------------------------------------------
        // 5. Store and Predict (Accumulate Prediction)
        // ---------------------------------------------------------------------
        if(it >= n_burnin) {
             res_beta.row(it-n_burnin) = beta.t(); 
             res_lambda.row(it-n_burnin) = lambda.t();
             
             // Calculate test set predictions
             mat P_test(n_test, q);
             for(int i=0; i<n_test; ++i) 
                for(int j=0; j<q; ++j) 
                    P_test(i,j) = (1-lambda(j))*X_L_test(i,j) + lambda(j)*X_U_test(i,j);
            
             mat M_L = zeros(m, n_test); mat M_U = zeros(m, n_test);
             vec aL_c = beta.subvec(0, K-1); vec aU_c = beta.subvec(K, 2*K-1);
             M_L += B * repmat(aL_c, 1, n_test);
             M_U += B * repmat(aU_c, 1, n_test);
             
             for(int j=0; j<q; ++j) {
                vec bL_c = beta.subvec((j+1)*2*K, (j+1)*2*K+K-1);
                vec bU_c = beta.subvec((j+1)*2*K+K, (j+1)*2*K+2*K-1);
                M_L += (B * bL_c) * P_test.col(j).t();
                M_U += (B * bU_c) * P_test.col(j).t();
             }
             pred_L_sum += M_L.t();
             pred_U_sum += M_U.t();
        }
    }
    
    // Calculate posterior mean predictions
    pred_L_sum /= n_save;
    pred_U_sum /= n_save;
    
    return List::create(Named("beta")=res_beta, 
                        Named("lambda")=res_lambda, 
                        Named("pred_L_mean")=pred_L_sum, 
                        Named("pred_U_mean")=pred_U_sum);
}
'

# ==============================================================================
# 2. Data Generation and Helper Functions
# ==============================================================================
create_penalty_matrix <- function(K) {
  D <- diff(diag(K), differences = 2)
  return(t(D) %*% D + diag(1e-6, K))
}

generate_data_study1 <- function(n, C_val, m = 51, q = 3, seed = NULL, true_params = NULL, lambda_values = NULL) {
  if(!is.null(seed)) set.seed(seed)
  t_grid <- seq(0, 1, length.out = m)
  
  c_ij <- matrix(runif(n * q, -10, 10), n, q) 
  r_ij <- matrix(runif(n * q, 0, 5), n, q)
  X_L <- c_ij - r_ij; X_U <- c_ij + r_ij
  
  if (is.null(true_params)) {
    if(!is.null(lambda_values)) {
      lambda_true <- lambda_values
    } else {
      lambda_true <- runif(q, 0, 1)
    }
    mu_L_true <- 1 + 2 * t_grid
    mu_U_true <- 2 + 2 * t_grid + 3*sin(pi * t_grid)
    
    beta_1L_true <- 2 * sin(2 * pi * t_grid)
    beta_1U_true <- 2 * sin(2 * pi * t_grid + pi / 4)
    
    poly_basis <- poly(t_grid, degree = 4, raw = FALSE)
    xi_L <- rnorm(4, 0, 1/(1:4)^2); xi_U <- rnorm(4, 0, 1/(1:4)^2)
    beta_2L_true <- as.numeric(poly_basis %*% xi_L) 
    beta_2U_true <- as.numeric(poly_basis %*% xi_U) 
    
    beta_3L_true <- rep(0, m); beta_3U_true <- rep(0, m)
    
    Beta_L_funcs <- list(beta_1L_true, beta_2L_true, beta_3L_true)
    Beta_U_funcs <- list(beta_1U_true, beta_2U_true, beta_3U_true)
    
    true_params <- list(lambda = lambda_true, mu_L = mu_L_true, mu_U = mu_U_true,
                        Beta_L = Beta_L_funcs, Beta_U = Beta_U_funcs, sigma = C_val)
  }
  
  lambda_true <- true_params$lambda
  P_star <- matrix(0, n, q)
  for(j in 1:q) P_star[, j] <- (1 - lambda_true[j]) * X_L[, j] + lambda_true[j] * X_U[, j]
  
  Y_L_mean <- matrix(true_params$mu_L, n, m, byrow=TRUE)
  Y_U_mean <- matrix(true_params$mu_U, n, m, byrow=TRUE)
  for(j in 1:q) {
    Y_L_mean <- Y_L_mean + P_star[, j] %*% t(true_params$Beta_L[[j]])
    Y_U_mean <- Y_U_mean + P_star[, j] %*% t(true_params$Beta_U[[j]])
  }
  
  Sigma_eps <- matrix(c(C_val^2, 0.5*C_val^2, 0.5*C_val^2, C_val^2), 2, 2)
  errors <- mvrnorm(n * m, mu = c(0,0), Sigma = Sigma_eps)
  Y_L <- Y_L_mean + matrix(errors[,1], n, m)
  Y_U <- Y_U_mean + matrix(errors[,2], n, m)
  
  return(list(Y_L=Y_L, Y_U=Y_U, X_L=X_L, X_U=X_U, t_grid=t_grid, true_params=true_params))
}

calc_imse <- function(true, est) mean((true - est)^2)

get_bs_basis <- function(t, df=15) bSpline(t, df=df, degree=3, intercept=TRUE)

fit_flm_benchmark <- function(Y, X, B) {
  Y_coef <- Y %*% B %*% solve(t(B) %*% B)
  dat_reg <- data.frame(X); colnames(dat_reg) <- paste0("V", 1:ncol(X))
  fit <- lm(Y_coef ~ ., data = dat_reg)
  coefs_mat <- coef(fit)
  mu_est <- as.vector(t(B %*% coefs_mat[1, ]))
  beta_est <- list()
  for(j in 1:ncol(X)) beta_est[[j]] <- as.vector(t(B %*% coefs_mat[j+1, ]))
  return(list(mu=mu_est, beta=beta_est))
}

predict_flm <- function(fit_obj, X_new) {
  n_test <- nrow(X_new); m <- length(fit_obj$mu)
  Y_pred <- matrix(fit_obj$mu, n_test, m, byrow=TRUE)
  for(j in 1:ncol(X_new)) Y_pred <- Y_pred + X_new[,j] %*% t(fit_obj$beta[[j]])
  return(Y_pred)
}

# ==============================================================================
# 3. Simulation Loop (Parallel)
# ==============================================================================
worker_init <- function(code) { 
  library(Rcpp); library(RcppArmadillo); library(splines2); library(MASS); 
  sourceCpp(code=code); 
  return(TRUE) 
}

# Simulation settings
n_set <- c(50, 100, 200)
C_set <- c(3, 5, 10)
n_reps <- 100 
sim_grid <- expand.grid(rep=1:n_reps, C=C_set, n=n_set)

# Setup parallel cluster
if(exists("cl")) { try(stopCluster(cl), silent=TRUE) }
cl <- makeCluster(detectCores() - 1)
registerDoSNOW(cl)
clusterExport(cl, c("worker_init", "generate_data_study1", "calc_imse", 
                    "fit_flm_benchmark", "predict_flm", "get_bs_basis", "create_penalty_matrix", "cpp_code_string"))
clusterEvalQ(cl, worker_init(cpp_code_string))

pb <- txtProgressBar(max=nrow(sim_grid), style=3)
progress <- function(n) setTxtProgressBar(pb, n)

cat("Running Simulation Study 1...\n")
results <- foreach(i=1:nrow(sim_grid), .combine=bind_rows, .packages=pkgs, .options.snow=list(progress=progress)) %dopar% {
  p <- sim_grid[i,]
  
  train_dat <- generate_data_study1(p$n, p$C, seed = 1000*p$n + 100*p$C + p$rep)
  test_dat <- generate_data_study1(50, p$C, seed = 2000*p$n + 200*p$C + p$rep, true_params = train_dat$true_params)
  B <- get_bs_basis(train_dat$t_grid)
  Omega <- create_penalty_matrix(ncol(B))
  
  res_rows <- list()
  
  # 1. S-BHIFR (Full Bayesian with Logit Lambda + Dense Kronecker)
  fit <- BHIFR_Homo_Cpp(train_dat$Y_L, train_dat$Y_U, train_dat$X_L, train_dat$X_U, 
                        B, Omega, 
                        test_dat$X_L, test_dat$X_U, # Input test data
                        5000, 2000)
  
  # Lambda Estimation Error
  l_est <- colMeans(fit$lambda)
  for(j in 1:3) {
    est_err <- l_est[j] - train_dat$true_params$lambda[j]
    res_rows[[length(res_rows)+1]] <- data.frame(method="S-BHIFR", param=paste0("lambda_",j), imse=est_err^2, est_error=est_err)
  }
  
  # Functions Estimation Error
  get_s <- function(idx) colMeans(fit$beta[, idx:(idx+14)]) %*% t(B)
  muL <- get_s(1); muU <- get_s(16)
  res_rows[[length(res_rows)+1]] <- data.frame(method="S-BHIFR", param="mu_L", imse=calc_imse(train_dat$true_params$mu_L, muL), est_error=NA)
  res_rows[[length(res_rows)+1]] <- data.frame(method="S-BHIFR", param="mu_U", imse=calc_imse(train_dat$true_params$mu_U, muU), est_error=NA)
  
  for(j in 1:3) {
    bL <- get_s(30+(j-1)*30+1); bU <- get_s(30+(j-1)*30+16)
    res_rows[[length(res_rows)+1]] <- data.frame(method="S-BHIFR", param=paste0("beta_",j,"L"), imse=calc_imse(train_dat$true_params$Beta_L[[j]], bL), est_error=NA)
    res_rows[[length(res_rows)+1]] <- data.frame(method="S-BHIFR", param=paste0("beta_",j,"U"), imse=calc_imse(train_dat$true_params$Beta_U[[j]], bU), est_error=NA)
  }
  
  # Prediction (Directly from C++ output)
  res_rows[[length(res_rows)+1]] <- data.frame(method="S-BHIFR", param="IMSPE_L", imse=calc_imse(test_dat$Y_L, fit$pred_L_mean), est_error=NA)
  res_rows[[length(res_rows)+1]] <- data.frame(method="S-BHIFR", param="IMSPE_U", imse=calc_imse(test_dat$Y_U, fit$pred_U_mean), est_error=NA)
  
  # 2. F-MinMax
  resL <- fit_flm_benchmark(train_dat$Y_L, train_dat$X_L, B)
  resU <- fit_flm_benchmark(train_dat$Y_U, train_dat$X_U, B)
  res_rows[[length(res_rows)+1]] <- data.frame(method="F-MinMax", param="mu_L", imse=calc_imse(train_dat$true_params$mu_L, resL$mu), est_error=NA)
  res_rows[[length(res_rows)+1]] <- data.frame(method="F-MinMax", param="mu_U", imse=calc_imse(train_dat$true_params$mu_U, resU$mu), est_error=NA)
  for(j in 1:3) {
    res_rows[[length(res_rows)+1]] <- data.frame(method="F-MinMax", param=paste0("beta_",j,"L"), imse=calc_imse(train_dat$true_params$Beta_L[[j]], resL$beta[[j]]), est_error=NA)
    res_rows[[length(res_rows)+1]] <- data.frame(method="F-MinMax", param=paste0("beta_",j,"U"), imse=calc_imse(train_dat$true_params$Beta_U[[j]], resU$beta[[j]]), est_error=NA)
  }
  pL <- predict_flm(resL, test_dat$X_L); pU <- predict_flm(resU, test_dat$X_U)
  res_rows[[length(res_rows)+1]] <- data.frame(method="F-MinMax", param="IMSPE_L", imse=calc_imse(test_dat$Y_L, pL), est_error=NA)
  res_rows[[length(res_rows)+1]] <- data.frame(method="F-MinMax", param="IMSPE_U", imse=calc_imse(test_dat$Y_U, pU), est_error=NA)
  
  # 3. F-CM
  Yc <- (train_dat$Y_L + train_dat$Y_U)/2; Xc <- (train_dat$X_L + train_dat$X_U)/2
  resc <- fit_flm_benchmark(Yc, Xc, B)
  res_rows[[length(res_rows)+1]] <- data.frame(method="F-CM", param="mu_L", imse=calc_imse(train_dat$true_params$mu_L, resc$mu), est_error=NA)
  res_rows[[length(res_rows)+1]] <- data.frame(method="F-CM", param="mu_U", imse=calc_imse(train_dat$true_params$mu_U, resc$mu), est_error=NA)
  for(j in 1:3) {
    res_rows[[length(res_rows)+1]] <- data.frame(method="F-CM", param=paste0("beta_",j,"L"), imse=calc_imse(train_dat$true_params$Beta_L[[j]], resc$beta[[j]]), est_error=NA)
    res_rows[[length(res_rows)+1]] <- data.frame(method="F-CM", param=paste0("beta_",j,"U"), imse=calc_imse(train_dat$true_params$Beta_U[[j]], resc$beta[[j]]), est_error=NA)
  }
  Xct <- (test_dat$X_L + test_dat$X_U)/2; pc <- predict_flm(resc, Xct)
  res_rows[[length(res_rows)+1]] <- data.frame(method="F-CM", param="IMSPE_L", imse=calc_imse(test_dat$Y_L, pc), est_error=NA)
  res_rows[[length(res_rows)+1]] <- data.frame(method="F-CM", param="IMSPE_U", imse=calc_imse(test_dat$Y_U, pc), est_error=NA)
  
  # 4. F-CRM
  Yr <- train_dat$Y_U - train_dat$Y_L; Xr <- train_dat$X_U - train_dat$X_L
  resr <- fit_flm_benchmark(Yr, Xr, B)
  muL_crm <- resc$mu - 0.5*resr$mu; muU_crm <- resc$mu + 0.5*resr$mu
  res_rows[[length(res_rows)+1]] <- data.frame(method="F-CRM", param="mu_L", imse=calc_imse(train_dat$true_params$mu_L, muL_crm), est_error=NA)
  res_rows[[length(res_rows)+1]] <- data.frame(method="F-CRM", param="mu_U", imse=calc_imse(train_dat$true_params$mu_U, muU_crm), est_error=NA)
  for(j in 1:3) {
    bL <- resc$beta[[j]] - 0.5*resr$beta[[j]]; bU <- resc$beta[[j]] + 0.5*resr$beta[[j]]
    res_rows[[length(res_rows)+1]] <- data.frame(method="F-CRM", param=paste0("beta_",j,"L"), imse=calc_imse(train_dat$true_params$Beta_L[[j]], bL), est_error=NA)
    res_rows[[length(res_rows)+1]] <- data.frame(method="F-CRM", param=paste0("beta_",j,"U"), imse=calc_imse(train_dat$true_params$Beta_U[[j]], bU), est_error=NA)
  }
  Xrt <- test_dat$X_U - test_dat$X_L; pr <- predict_flm(resr, Xrt)
  pL_crm <- pc - 0.5*pr; pU_crm <- pc + 0.5*pr
  res_rows[[length(res_rows)+1]] <- data.frame(method="F-CRM", param="IMSPE_L", imse=calc_imse(test_dat$Y_L, pL_crm), est_error=NA)
  res_rows[[length(res_rows)+1]] <- data.frame(method="F-CRM", param="IMSPE_U", imse=calc_imse(test_dat$Y_U, pU_crm), est_error=NA)
  
  d <- bind_rows(res_rows)
  d$n <- p$n; d$C <- p$C; d$rep <- p$rep
  d
}
stopCluster(cl); close(pb)

# ==============================================================================
# 4. Result Aggregation and Table Output (LaTeX)
# ==============================================================================
sourceCpp(code=cpp_code_string)

cat("\n\n--- Table 1: Prediction Performance Comparison (IMSPE) ---\n")
tab1_res <- results %>% 
  filter(param %in% c("IMSPE_L", "IMSPE_U")) %>%
  group_by(param, n, C, method) %>%
  summarise(Mean = mean(imse), SD = sd(imse), .groups='drop') %>%
  mutate(Val = sprintf("%.4f (%.4f)", Mean, SD)) %>%
  select(-Mean, -SD) %>%
  pivot_wider(names_from=method, values_from=Val) %>%
  arrange(param, n, C)

print(kbl(tab1_res, format="latex", booktabs=T, caption="Table 1: Prediction Performance (IMSPE_L and IMSPE_U)") %>% 
        kable_styling(latex_options=c("striped", "scale_down")))

cat("\n\n--- Table 2: Parameter Estimation Performance Part 1 (mu, beta1) ---\n")
tab2_params <- c("mu_L", "mu_U", "beta_1L", "beta_1U")
tab2_res <- results %>% 
  filter(param %in% tab2_params) %>%
  group_by(param, n, C, method) %>%
  summarise(Mean = mean(imse), SD = sd(imse), .groups='drop') %>%
  mutate(Val = sprintf("%.6f (%.6f)", Mean, SD)) %>%
  select(-Mean, -SD) %>%
  pivot_wider(names_from=method, values_from=Val) %>%
  arrange(param, n, C)

print(kbl(tab2_res, format="latex", booktabs=T, caption="Table 2: Parameter Estimation (Part 1: Intercept & Beta1)") %>% 
        kable_styling(latex_options=c("striped", "scale_down")))

cat("\n\n--- Table 3: Parameter Estimation Performance Part 2 (beta2, beta3) ---\n")
tab3_params <- c("beta_2L", "beta_2U", "beta_3L", "beta_3U")
tab3_res <- results %>% 
  filter(param %in% tab3_params) %>%
  group_by(param, n, C, method) %>%
  summarise(Mean = mean(imse), SD = sd(imse), .groups='drop') %>%
  mutate(Val = sprintf("%.6f (%.6f)", Mean, SD)) %>%
  select(-Mean, -SD) %>%
  pivot_wider(names_from=method, values_from=Val) %>%
  arrange(param, n, C)

print(kbl(tab3_res, format="latex", booktabs=T, caption="Table 3: Parameter Estimation (Part 2: Beta2 & Beta3)") %>% 
        kable_styling(latex_options=c("striped", "scale_down")))

cat("\n\n--- Table 4: Weight Parameter Estimation (Lambda MSE & Bias) ---\n")
tab4_res <- results %>% 
  filter(method == "S-BHIFR", grepl("lambda", param)) %>%
  group_by(param, n, C) %>%
  summarise(MSE = mean(imse), Bias = mean(est_error), .groups='drop') %>%
  pivot_wider(names_from=param, values_from=c(MSE, Bias))

print(kbl(tab4_res, format="latex", booktabs=T, digits=6, caption="Table 4: Weight Parameter Estimation (MSE & Bias)") %>% 
        kable_styling(latex_options="hold_position"))

# ==============================================================================
# 5. Plotting Output (Final: Single Row Compact Legend for Fig 1)
# ==============================================================================

# --- 0. Global Configuration ---
library(ggplot2)
library(patchwork)
library(ggsci)
library(dplyr)

mm_to_inch <- 0.0393701
WIDTH_DOUBLE <- 174 * mm_to_inch 

# Minimalist theme 
theme_clean_journal <- function() {
  theme_classic(base_family = "sans", base_size = 10) + 
    theme(
      text = element_text(color = "black"),
      axis.text = element_text(color = "black", size = 9),
      axis.title = element_text(face = "bold", size = 10),
      plot.title = element_text(face = "bold", size = 11, hjust = 0.5),
      panel.grid.major.y = element_line(color = "grey90", size = 0.3, linetype = "dashed"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", size = 0.5, fill = NA),
      axis.line = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = 10, margin = margin(b = 5)),
      
      # --- Legend Basic Settings ---
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "horizontal",      
      legend.title = element_blank(), 
      legend.background = element_blank(),
      legend.key = element_blank()
    )
}

# --- Figure 1: Curves ---
cat("Generating Figure 1 (Curves)...\n")

set.seed(999)
d_demo <- generate_data_study1(100, 5)
B_demo <- get_bs_basis(d_demo$t_grid)
Omega_demo <- create_penalty_matrix(ncol(B_demo))
fit_demo <- BHIFR_Homo_Cpp(d_demo$Y_L, d_demo$Y_U, d_demo$X_L, d_demo$X_U, B_demo, Omega_demo, d_demo$X_L, d_demo$X_U, 5000, 2000)
get_d <- function(idx) colMeans(fit_demo$beta[, idx:(idx+14)]) %*% t(B_demo)
get_ci <- function(idx) apply(fit_demo$beta[, idx:(idx+14)] %*% t(B_demo), 2, quantile, probs=c(0.025,0.975))

# Plotting Function
plot_curve_with_legend <- function(df, ylab_expr, show_legend = FALSE) {
  p <- ggplot(df, aes(x=t)) +
    # CI Ribbons
    geom_ribbon(aes(ymin=LL, ymax=UL, fill="Lower Bound"), alpha=0.15) +
    geom_ribbon(aes(ymin=LU, ymax=UU, fill="Upper Bound"), alpha=0.15) +
    
    # True Values (Dashed)
    geom_line(aes(y=TL, color="Lower Bound", linetype="True Value"), size=0.6) +
    geom_line(aes(y=TU, color="Upper Bound", linetype="True Value"), size=0.6) +
    
    # Estimates (Solid)
    geom_line(aes(y=EL, color="Lower Bound", linetype="Estimate"), size=0.6) +
    geom_line(aes(y=EU, color="Upper Bound", linetype="Estimate"), size=0.6) +
    
    # Scales
    scale_color_manual(values = c("Lower Bound"="#4DBBD5FF", "Upper Bound"="#E64B35FF")) +
    scale_fill_manual(values = c("Lower Bound"="#4DBBD5FF", "Upper Bound"="#E64B35FF")) +
    scale_linetype_manual(values = c("True Value"="dashed", "Estimate"="solid")) +
    
    # Labels
    labs(y=ylab_expr, x=expression(italic(t))) +
    theme_clean_journal()
  
  if (!show_legend) {
    p <- p + theme(legend.position = "none")
  }
  return(p)
}

# (a) Intercept
muL_est <- get_d(1); muL_ci <- get_ci(1); muU_est <- get_d(16); muU_ci <- get_ci(16)
df_mu <- data.frame(t=d_demo$t_grid, TL=d_demo$true_params$mu_L, EL=as.numeric(muL_est), LL=muL_ci[1,], UL=muL_ci[2,], TU=d_demo$true_params$mu_U, EU=as.numeric(muU_est), LU=muU_ci[1,], UU=muU_ci[2,])

# (b) Beta 1
b1L_est <- get_d(31); b1L_ci <- get_ci(31); b1U_est <- get_d(46); b1U_ci <- get_ci(46)
df_b1 <- data.frame(t=d_demo$t_grid, TL=d_demo$true_params$Beta_L[[1]], EL=as.numeric(b1L_est), LL=b1L_ci[1,], UL=b1L_ci[2,], TU=d_demo$true_params$Beta_U[[1]], EU=as.numeric(b1U_est), LU=b1U_ci[1,], UU=b1U_ci[2,])

# (c) Beta 2
b2L_est <- get_d(61); b2L_ci <- get_ci(61); b2U_est <- get_d(76); b2U_ci <- get_ci(76)
df_b2 <- data.frame(t=d_demo$t_grid, TL=d_demo$true_params$Beta_L[[2]], EL=as.numeric(b2L_est), LL=b2L_ci[1,], UL=b2L_ci[2,], TU=d_demo$true_params$Beta_U[[2]], EU=as.numeric(b2U_est), LU=b2U_ci[1,], UU=b2U_ci[2,])

# (d) Beta 3
b3L_est <- get_d(91); b3L_ci <- get_ci(91); b3U_est <- get_d(106); b3U_ci <- get_ci(106)
df_b3 <- data.frame(t=d_demo$t_grid, TL=d_demo$true_params$Beta_L[[3]], EL=as.numeric(b3L_est), LL=b3L_ci[1,], UL=b3L_ci[2,], TU=d_demo$true_params$Beta_U[[3]], EU=as.numeric(b3U_est), LU=b3U_ci[1,], UU=b3U_ci[2,])

# --- Combined Plot and Legend Optimization ---
p_combined_final <- (
  (plot_curve_with_legend(df_mu, expression(mu(t)), TRUE) + ggtitle("(a)")) + 
    (plot_curve_with_legend(df_b1, expression(beta[1](t)), TRUE) + ggtitle("(b)"))
) / (
  (plot_curve_with_legend(df_b2, expression(beta[2](t)), TRUE) + ggtitle("(c)")) + 
    (plot_curve_with_legend(df_b3, expression(beta[3](t)), TRUE) + ggtitle("(d)"))
) + 
  plot_layout(guides = "collect") & 
  theme(
    legend.position = "bottom",
    legend.box = "horizontal",       
    legend.spacing.x = unit(0.2, "cm"), 
    legend.margin = margin(t = 0, b = 5), 
    legend.text = element_text(size = 8), 
    legend.key.width = unit(0.5, "cm"),   
    legend.key.height = unit(0.3, "cm")   
  ) &
  guides(
    linetype = guide_legend(title = NULL, nrow = 1, order = 1), 
    color = guide_legend(title = NULL, nrow = 1, order = 2),
    fill = guide_legend(title = NULL, nrow = 1, order = 2)
  )

ggsave("Fig1.eps", plot=p_combined_final, width=WIDTH_DOUBLE, height=6, device=cairo_ps)

cat("Done! Figure 1 saved with a single-row, compact legend.\n")