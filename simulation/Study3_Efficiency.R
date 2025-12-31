# ==============================================================================
# BHIFR Simulation Study 3: Computational Efficiency & Convergence
# Description: Comparison of computational speed (Runtime) and sampling efficiency (ESS)
#              between the proposed Kronecker-based algorithm and standard naive implementation.
#              Includes: C++ implementation of both methods and benchmark execution.
# ==============================================================================

rm(list=ls())
set.seed(123) 

# ==============================================================================
# 1. Environment Loading
# ==============================================================================
pkgs <- c("Rcpp", "RcppArmadillo", "splines2", "MASS", "dplyr", 
          "knitr", "foreach", "doSNOW", "parallel", "tidyr", 
          "kableExtra", "ggplot2", "patchwork", "ggsci", "coda", "scales")

# Install missing packages
new_pkgs <- pkgs[!(pkgs %in% installed.packages()[,"Package"])]
if(length(new_pkgs)) install.packages(new_pkgs)

# Load packages
invisible(lapply(pkgs, require, character.only = TRUE))

# ==============================================================================
# 2. Helper Functions (R side)
# ==============================================================================

get_bs_basis <- function(t, df=15) {
  splines2::bSpline(t, df=df, degree=3, intercept=TRUE)
}

create_penalty_matrix <- function(K) {
  D <- diff(diag(K), differences = 2)
  t(D) %*% D + diag(1e-6, K)
}

# ==============================================================================
# 3. C++ Core Implementation: Fast (Kronecker) & Standard (Naive Loop)
# ==============================================================================
cpp_code_string <- '
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace arma;
using namespace Rcpp;

// Constants
const double nugget = 1e-6; 

// --- General Helper Functions ---
bool robust_chol(mat& L, const mat& A_in) {
    if(!A_in.is_finite()) return false;
    if(chol(L, A_in, "lower")) return true;
    mat A = A_in; 
    double diag_mean = mean(A.diag());
    double jitter = 1e-8 * diag_mean; if(jitter < 1e-10) jitter = 1e-10;
    A.diag() += jitter;
    if(chol(L, A, "lower")) return true;
    A.diag() += 100.0 * jitter;
    if(chol(L, A, "lower")) return true;
    return false;
}

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
double get_log_det_chol(const mat& L) { return 2.0 * sum(log(L.diag())); }

double calc_joint_loglik(const mat& E_L, const mat& E_U, double sL, double sU, double rho) {
    int n = E_L.n_rows; int m = E_L.n_cols;
    double omr2 = 1.0 - rho*rho;
    double log_det = -n * m * (log(sL) + log(sU)) - 0.5 * n * m * std::log(omr2);
    
    double quad = 0.0;
    for(int t=0; t<m; ++t) {
        double tL = accu(pow(E_L.col(t), 2)) / (sL*sL);
        double tU = accu(pow(E_U.col(t), 2)) / (sU*sU);
        double tLU = accu(E_L.col(t) % E_U.col(t)) / (sL*sU);
        quad += (tL + tU - 2.0 * rho * tLU);
    }
    return log_det - 0.5 / omr2 * quad;
}

// -----------------------------------------------------------------------------
// Helper 1: Fast Marginal Likelihood (Kronecker Algebra)
// -----------------------------------------------------------------------------
double calc_marginal_ll_fast(const vec& lambda, const mat& X_L, const mat& X_U,
                             const mat& BTWB_blk, const mat& BTY_proj, const mat& Prior_Prec) {
    int n = X_L.n_rows; int q = X_L.n_cols;
    mat P = ones(n, q+1);
    for(int j=0; j<q; ++j) P.col(j+1) = (1.0-lambda(j))*X_L.col(j) + lambda(j)*X_U.col(j);
    
    // Fast Construction: O(n*q^2) + O(K^2*q^2)
    mat PTP = P.t()*P;
    mat Prec = kron(PTP, BTWB_blk) + Prior_Prec;
    
    mat L_chol;
    if(!robust_chol(L_chol, Prec)) return -1e20;
    
    vec RHS = vectorise(BTY_proj * P);
    vec y = solve(trimatl(L_chol), RHS);
    return -0.5 * get_log_det_chol(L_chol) + 0.5 * dot(y,y);
}

// -----------------------------------------------------------------------------
// Helper 2: Standard/Naive Marginal Likelihood (Explicit Loop)
// -----------------------------------------------------------------------------
double calc_marginal_ll_standard(const vec& lambda, const mat& X_L, const mat& X_U,
                                 const mat& B_tilde, const mat& W_single, const mat& Prior_Prec,
                                 const mat& Y_L, const mat& Y_U) {
    int n = X_L.n_rows; int q = X_L.n_cols; int dim_beta = Prior_Prec.n_rows;
    
    mat P = ones(n, q+1);
    for(int j=0; j<q; ++j) P.col(j+1) = (1.0-lambda(j))*X_L.col(j) + lambda(j)*X_U.col(j);
    
    // Naive Construction: O(n * (m*K)^2) - Very Slow!
    mat Prec_Data = zeros(dim_beta, dim_beta);
    vec BTY_flat = zeros(dim_beta);
    
    for(int i=0; i<n; ++i) {
         // Construct Design Matrix for subject i explicitly
         mat D_i = kron(P.row(i), B_tilde); 
         // Accumulate Information
         Prec_Data += D_i.t() * W_single * D_i; 
         // Accumulate RHS
         vec y_i = join_cols(Y_L.row(i).t(), Y_U.row(i).t());
         BTY_flat += D_i.t() * W_single * y_i;
    }
    
    mat Prec = Prec_Data + Prior_Prec;
    mat L_chol;
    if(!robust_chol(L_chol, Prec)) return -1e20;
    
    vec y = solve(trimatl(L_chol), BTY_flat);
    return -0.5 * get_log_det_chol(L_chol) + 0.5 * dot(y,y);
}

// -----------------------------------------------------------------------------
// Model 1: BHIFR_Fast (Kronecker Accelerated)
// -----------------------------------------------------------------------------
// [[Rcpp::export]]
List BHIFR_Fast(mat Y_L, mat Y_U, mat X_L, mat X_U, mat B, mat Omega, int n_iter, int n_burnin, bool verbose=false) {
    int n = Y_L.n_rows; int m = Y_L.n_cols; int q = X_L.n_cols; int K = B.n_cols;
    int dim_beta = 2 * K * (q + 1);

    vec lambda = randu(q) * 0.8 + 0.1;
    vec beta = zeros(dim_beta);
    double sigma2_L = 0.1, sigma2_U = 0.1, rho = 0.0;
    double nu_sigma_L = 1.0, nu_sigma_U = 1.0; 
    vec tau_beta = ones(2 * (q + 1)) * 10.0; vec nu_beta = ones(2 * (q + 1));
    
    vec step_l_vec = ones(q) * 0.5;
    
    int n_save = (n_iter > n_burnin) ? (n_iter - n_burnin) : 1;
    if(n_burnin == 0 && n_iter > 0) n_save = n_iter; 
    
    mat res_lambda(n_save, q); 
    vec res_rho(n_save);
    
    mat BTB = B.t()*B; mat BTY_L = B.t()*Y_L.t(); mat BTY_U = B.t()*Y_U.t();
    
    for(int it=0; it<n_iter; ++it) {
        if(verbose && (it+1)%500 == 0) Rcout << "Fast Iter: " << it+1 << "\\r";
        double gamma_t = std::min(1.0, 10.0 / (double(it) + 100.0));

        double sL = sqrt(sigma2_L + nugget); double sU = sqrt(sigma2_U + nugget);
        double sc = 1.0/(1.0 - rho*rho);
        double w11=sc/(sL*sL); double w22=sc/(sU*sU); double w12=-sc*rho/(sL*sU);
        
        mat BTWB_blk = zeros(2*K, 2*K);
        BTWB_blk.submat(0,0,K-1,K-1)=w11*BTB; BTWB_blk.submat(K,K,2*K-1,2*K-1)=w22*BTB;
        BTWB_blk.submat(0,K,K-1,2*K-1)=w12*BTB; BTWB_blk.submat(K,0,2*K-1,K-1)=w12*BTB;
        
        mat BTY_proj = zeros(2*K, n);
        BTY_proj.rows(0,K-1) = w11*BTY_L + w12*BTY_U;
        BTY_proj.rows(K,2*K-1) = w12*BTY_L + w22*BTY_U;
        
        mat Prior_Prec = zeros(dim_beta, dim_beta);
        for(int g=0; g<2*(q+1); ++g) Prior_Prec.submat(g*K, g*K, g*K+K-1, g*K+K-1) = tau_beta(g) * Omega;
        
        double current_ll = calc_marginal_ll_fast(lambda, X_L, X_U, BTWB_blk, BTY_proj, Prior_Prec);
        
        // --- Update Lambda (Logit + Adaptive MH) ---
        for(int j=0; j<q; ++j) {
            double eta_curr = logit(lambda(j));
            double eta_prop = eta_curr + step_l_vec(j) * randn();
            double lambda_prop_val = sigmoid(eta_prop);
            
            vec lam_prop = lambda; lam_prop(j) = lambda_prop_val;
            double prop_ll = calc_marginal_ll_fast(lam_prop, X_L, X_U, BTWB_blk, BTY_proj, Prior_Prec);
            
            double jacobian_diff = std::log(lambda_prop_val * (1.0 - lambda_prop_val)) - 
                                   std::log(lambda(j) * (1.0 - lambda(j)));
            
            bool accept = std::log(randu()) < (prop_ll - current_ll + jacobian_diff);
            
            if(accept) {
                lambda(j) = lambda_prop_val;
                current_ll = prop_ll;
            }
            if(it < n_burnin) {
                double alpha = accept ? 1.0 : 0.0;
                double log_step = std::log(step_l_vec(j)) + gamma_t * (alpha - 0.44);
                step_l_vec(j) = std::exp(log_step);
            }
        }
        
        // --- Update Beta ---
        mat P(n,q+1); P.col(0).fill(1.0); 
        for(int j=0; j<q; ++j) P.col(j+1)=(1-lambda(j))*X_L.col(j)+lambda(j)*X_U.col(j);
        mat PTP = P.t()*P;
        mat Prec = kron(PTP, BTWB_blk) + Prior_Prec;
        mat L_post; robust_chol(L_post, Prec);
        vec z = randn(dim_beta);
        beta = solve(trimatu(L_post.t()), solve(trimatl(L_post), vectorise(BTY_proj * P))) + solve(trimatu(L_post.t()), z);

        // --- Other Updates (Hypers, Sigma, Rho) ---
        for(int g=0; g < 2*(q+1); ++g) {
             vec b_sub = beta.subvec(g*K, g*K+K-1);
             double quad = as_scalar(b_sub.t() * Omega * b_sub);
             tau_beta(g) = R::rgamma(0.5*K + 0.5, 1.0/(0.5*quad + 1.0/nu_beta(g)));
             // Inverse Gamma sampling via 1.0 / Gamma
             nu_beta(g) = 1.0 / R::rgamma(1.0, 1.0/(1.0/25.0 + tau_beta(g)));
        }
        
        mat YL_h(n,m), YU_h(n,m); YL_h.zeros(); YU_h.zeros();
        vec aL=beta.head(K); vec aU=beta.subvec(K, 2*K-1); 
        YL_h.each_row() += (B*aL).t(); YU_h.each_row() += (B*aU).t();
        for(int j=0;j<q;++j) { 
            vec bL=beta.subvec((j+1)*2*K, (j+1)*2*K+K-1); vec bU=beta.subvec((j+1)*2*K+K, (j+1)*2*K+2*K-1);
            for(int i=0;i<n;++i) { YL_h.row(i)+=P(i,j+1)*(B*bL).t(); YU_h.row(i)+=P(i,j+1)*(B*bU).t(); } 
        }
        mat EL=Y_L-YL_h; mat EU=Y_U-YU_h;
        
        vec theta_curr(3); theta_curr(0)=log(sqrt(sigma2_L)); theta_curr(1)=log(sqrt(sigma2_U)); theta_curr(2)=fisher_z(rho);
        vec theta_prop = theta_curr + 0.02 * randn(3); 
        double lsL_p=theta_prop(0); double lsU_p=theta_prop(1); double rho_p=inv_fisher_z(theta_prop(2));
        double lp_curr = calc_joint_loglik(EL,EU,exp(theta_curr(0)),exp(theta_curr(1)),rho) - theta_curr(0) - 1.0/(nu_sigma_L*exp(2*theta_curr(0))) - theta_curr(1) - 1.0/(nu_sigma_U*exp(2*theta_curr(1)));
        double lp_prop = calc_joint_loglik(EL,EU,exp(lsL_p),exp(lsU_p),rho_p) - theta_prop(0) - 1.0/(nu_sigma_L*exp(2*theta_prop(0))) - theta_prop(1) - 1.0/(nu_sigma_U*exp(2*theta_prop(1)));
        if(log(randu()) < (lp_prop - lp_curr + log(1-rho_p*rho_p) - log(1-rho*rho))) {
             sigma2_L = exp(2*lsL_p); sigma2_U = exp(2*lsU_p); rho = rho_p;
        }
        // Inverse Gamma sampling
        nu_sigma_L = 1.0 / R::rgamma(1.0, 1.0/(1.0/25.0 + 1.0/sigma2_L));
        nu_sigma_U = 1.0 / R::rgamma(1.0, 1.0/(1.0/25.0 + 1.0/sigma2_U));
        
        if(it >= n_burnin) {
            int idx = it - n_burnin;
            res_lambda.row(idx) = lambda.t();
            res_rho(idx) = rho;
        }
    }
    if(verbose) Rcout << "\\n";
    return List::create(Named("lambda")=res_lambda, Named("rho")=res_rho);
}

// -----------------------------------------------------------------------------
// Model 2: BHIFR_Standard (Naive Loop - Mathematically Equivalent but Slow)
// -----------------------------------------------------------------------------
// [[Rcpp::export]]
List BHIFR_Standard(mat Y_L, mat Y_U, mat X_L, mat X_U, mat B, mat Omega, int n_iter, int n_burnin, bool verbose=false) {
    int n = Y_L.n_rows; int m = Y_L.n_cols; int q = X_L.n_cols; int K = B.n_cols;
    int dim_beta = 2 * K * (q + 1);

    vec lambda = randu(q) * 0.8 + 0.1;
    vec beta = zeros(dim_beta);
    double sigma2_L = 0.1, sigma2_U = 0.1, rho = 0.0;
    double nu_sigma_L = 1.0, nu_sigma_U = 1.0; 
    vec tau_beta = ones(2 * (q + 1)) * 10.0; vec nu_beta = ones(2 * (q + 1));
    
    vec step_l_vec = ones(q) * 0.5;
    
    int n_save = (n_iter > n_burnin) ? (n_iter - n_burnin) : 1;
    if(n_burnin == 0 && n_iter > 0) n_save = n_iter; 
    
    mat res_lambda(n_save, q); 
    vec res_rho(n_save);
    
    mat B_tilde = zeros(2*m, 2*K);
    B_tilde.submat(0,0,m-1,K-1) = B; 
    B_tilde.submat(m,K,2*m-1,2*K-1) = B;
    
    for(int it=0; it<n_iter; ++it) {
        if(verbose && (it+1)%50 == 0) Rcout << "Standard Iter: " << it+1 << "\\r";
        double gamma_t = std::min(1.0, 10.0 / (double(it) + 100.0));

        double sL = sqrt(sigma2_L + nugget); double sU = sqrt(sigma2_U + nugget);
        double sc = 1.0/(1.0 - rho*rho);
        double w11=sc/(sL*sL); double w22=sc/(sU*sU); double w12=-sc*rho/(sL*sU);
        
        // Full Weight Matrix W for one subject (2m x 2m)
        mat W_single = zeros(2*m, 2*m);
        W_single.submat(0,0,m-1,m-1) = w11 * eye(m,m);
        W_single.submat(m,m,2*m-1,2*m-1) = w22 * eye(m,m);
        W_single.submat(0,m,m-1,2*m-1) = w12 * eye(m,m);
        W_single.submat(m,0,2*m-1,m-1) = w12 * eye(m,m);
        
        mat Prior_Prec = zeros(dim_beta, dim_beta);
        for(int g=0; g<2*(q+1); ++g) Prior_Prec.submat(g*K, g*K, g*K+K-1, g*K+K-1) = tau_beta(g) * Omega;

        // --- Update Lambda (Explicit Loop for Likelihood - Naive) ---
        double current_ll = calc_marginal_ll_standard(lambda, X_L, X_U, B_tilde, W_single, Prior_Prec, Y_L, Y_U);
        
        for(int j=0; j<q; ++j) {
            double eta_curr = logit(lambda(j));
            double eta_prop = eta_curr + step_l_vec(j) * randn();
            double lambda_prop_val = sigmoid(eta_prop);
            
            vec lam_prop = lambda; lam_prop(j) = lambda_prop_val;
            double prop_ll = calc_marginal_ll_standard(lam_prop, X_L, X_U, B_tilde, W_single, Prior_Prec, Y_L, Y_U);
            
            double jacobian_diff = std::log(lambda_prop_val * (1.0 - lambda_prop_val)) - 
                                   std::log(lambda(j) * (1.0 - lambda(j)));
            
            bool accept = std::log(randu()) < (prop_ll - current_ll + jacobian_diff);
            
            if(accept) {
                lambda(j) = lambda_prop_val;
                current_ll = prop_ll;
            }
            if(it < n_burnin) {
                double alpha = accept ? 1.0 : 0.0;
                double log_step = std::log(step_l_vec(j)) + gamma_t * (alpha - 0.44);
                step_l_vec(j) = std::exp(log_step);
            }
        }
        
        // --- Update Beta (Explicit Loop construction) ---
        mat P(n,q+1); P.col(0).fill(1.0); for(int j=0; j<q; ++j) P.col(j+1)=(1-lambda(j))*X_L.col(j)+lambda(j)*X_U.col(j);
        
        mat Prec_Data = zeros(dim_beta, dim_beta);
        vec BTY_flat = zeros(dim_beta);
        for(int i=0; i<n; ++i) {
             mat D_i = kron(P.row(i), B_tilde); 
             Prec_Data += D_i.t() * W_single * D_i; 
             vec y_i = join_cols(Y_L.row(i).t(), Y_U.row(i).t());
             BTY_flat += D_i.t() * W_single * y_i;
        }
        
        mat Prec = Prec_Data + Prior_Prec;
        mat L_post; robust_chol(L_post, Prec);
        vec z = randn(dim_beta);
        beta = solve(trimatu(L_post.t()), solve(trimatl(L_post), BTY_flat)) + solve(trimatu(L_post.t()), z);
        
        // --- Hyperparams & Variance (Same as Fast) ---
        for(int g=0; g < 2*(q+1); ++g) {
             vec b_sub = beta.subvec(g*K, g*K+K-1);
             double quad = as_scalar(b_sub.t() * Omega * b_sub);
             tau_beta(g) = R::rgamma(0.5*K + 0.5, 1.0/(0.5*quad + 1.0/nu_beta(g)));
             // Inverse Gamma sampling via 1.0 / Gamma
             nu_beta(g) = 1.0 / R::rgamma(1.0, 1.0/(1.0/25.0 + tau_beta(g)));
        }
        
        mat YL_h(n,m), YU_h(n,m); YL_h.zeros(); YU_h.zeros();
        vec aL=beta.head(K); vec aU=beta.subvec(K, 2*K-1); 
        YL_h.each_row() += (B*aL).t(); YU_h.each_row() += (B*aU).t();
        for(int j=0;j<q;++j) { 
            vec bL=beta.subvec((j+1)*2*K, (j+1)*2*K+K-1); vec bU=beta.subvec((j+1)*2*K+K, (j+1)*2*K+2*K-1);
            for(int i=0;i<n;++i) { YL_h.row(i)+=P(i,j+1)*(B*bL).t(); YU_h.row(i)+=P(i,j+1)*(B*bU).t(); } 
        }
        mat EL=Y_L-YL_h; mat EU=Y_U-YU_h;
        
        vec theta_curr(3); theta_curr(0)=log(sqrt(sigma2_L)); theta_curr(1)=log(sqrt(sigma2_U)); theta_curr(2)=fisher_z(rho);
        vec theta_prop = theta_curr + 0.02 * randn(3); 
        double lsL_p=theta_prop(0); double lsU_p=theta_prop(1); double rho_p=inv_fisher_z(theta_prop(2));
        double lp_curr = calc_joint_loglik(EL,EU,exp(theta_curr(0)),exp(theta_curr(1)),rho) - theta_curr(0) - 1.0/(nu_sigma_L*exp(2*theta_curr(0))) - theta_curr(1) - 1.0/(nu_sigma_U*exp(2*theta_curr(1)));
        double lp_prop = calc_joint_loglik(EL,EU,exp(lsL_p),exp(lsU_p),rho_p) - theta_prop(0) - 1.0/(nu_sigma_L*exp(2*theta_prop(0))) - theta_prop(1) - 1.0/(nu_sigma_U*exp(2*theta_prop(1)));
        if(log(randu()) < (lp_prop - lp_curr + log(1-rho_p*rho_p) - log(1-rho*rho))) {
             sigma2_L = exp(2*lsL_p); sigma2_U = exp(2*lsU_p); rho = rho_p;
        }
        // Inverse Gamma sampling
        nu_sigma_L = 1.0 / R::rgamma(1.0, 1.0/(1.0/25.0 + 1.0/sigma2_L));
        nu_sigma_U = 1.0 / R::rgamma(1.0, 1.0/(1.0/25.0 + 1.0/sigma2_U));
        
        if(it >= n_burnin) {
            int idx = it - n_burnin;
            res_lambda.row(idx) = lambda.t();
            res_rho(idx) = rho;
        }
    }
    if(verbose) Rcout << "\\n";
    return List::create(Named("lambda")=res_lambda, Named("rho")=res_rho);
}
'

sourceCpp(code=cpp_code_string)

# ==============================================================================
# 4. Data Generation Function
# ==============================================================================
generate_data_s3 <- function(n, m = 20, q = 3, seed = NULL) {
  if(!is.null(seed)) set.seed(seed)
  t_grid <- seq(0, 1, length.out = m)
  
  c_ij <- matrix(runif(n * q, -10, 10), n, q) 
  r_ij <- matrix(runif(n * q, 0, 5), n, q)
  X_L <- c_ij - r_ij; X_U <- c_ij + r_ij
  
  lambda_true <- c(0.2, 0.5, 0.8)
  
  # True coefficient functions
  beta_L_list <- list(
    function(t) 3 * sin(2 * pi * t),
    function(t) 4 * (t - 0.5)^2,
    function(t) 2 * cos(3 * pi * t)
  )
  beta_U_list <- list(
    function(t) 3 * sin(2 * pi * t + pi / 4),
    function(t) 4 * (t - 0.5)^2 + 1,
    function(t) 2 * cos(3 * pi * t) + 0.5
  )
  
  mu_L <- 1 + 2 * t_grid
  mu_U <- 2 + 2 * t_grid + 3*sin(pi * t_grid)
  
  P_star <- matrix(0, n, q)
  for(j in 1:q) P_star[, j] <- (1 - lambda_true[j]) * X_L[, j] + lambda_true[j] * X_U[, j]
  
  Y_L_mean <- matrix(mu_L, n, m, byrow=TRUE)
  Y_U_mean <- matrix(mu_U, n, m, byrow=TRUE)
  for(j in 1:q) {
    Y_L_mean <- Y_L_mean + P_star[, j] %*% t(beta_L_list[[j]](t_grid))
    Y_U_mean <- Y_U_mean + P_star[, j] %*% t(beta_U_list[[j]](t_grid))
  }
  
  Sigma_eps <- matrix(c(4, 2, 2, 4), 2, 2)
  errors <- mvrnorm(n * m, mu = c(0,0), Sigma = Sigma_eps)
  Y_L <- Y_L_mean + matrix(errors[,1], n, m)
  Y_U <- Y_U_mean + matrix(errors[,2], n, m)
  
  return(list(Y_L=Y_L, Y_U=Y_U, X_L=X_L, X_U=X_U, t=t_grid))
}

# ==============================================================================
# Part A: Convergence Diagnostics (Using Fast Method on Large Sample)
# ==============================================================================
cat("\n========================================================\n")
cat(" PART A: Convergence Diagnostics (n=200, 10,000 iters)\n")
cat("========================================================\n")

dat_conv <- generate_data_s3(200, m=20, seed=123)
B <- get_bs_basis(dat_conv$t)
Omega <- create_penalty_matrix(ncol(B))

chains <- list()
n_iter_conv <- 10000
n_burn_conv <- 5000

# Run 3 Chains for robust diagnostics
for(i in 1:3) {
  cat(sprintf("Running Chain %d...\n", i))
  set.seed(100+i)
  fit <- BHIFR_Fast(dat_conv$Y_L, dat_conv$Y_U, dat_conv$X_L, dat_conv$X_U, B, Omega, n_iter_conv, n_burn_conv, verbose=TRUE)
  df <- data.frame(fit$lambda); colnames(df) <- paste0("lambda", 1:3)
  df$rho <- fit$rho
  chains[[i]] <- as.mcmc(df)
}
mcmc_l <- as.mcmc.list(chains)

# Diagnostics
psrf <- gelman.diag(mcmc_l)$psrf[,1]
ess_conv <- effectiveSize(mcmc_l)
cat("\nConvergence Results:\n")
cat("Max PSRF:", max(psrf), "\n")
print(ess_conv)

# ==============================================================================
# Part B: Computational Efficiency Benchmark (Full Standard vs Fast)
# ==============================================================================
cat("\n========================================================\n")
cat(" PART B: Computational Efficiency (Full N=50...1000)\n")
cat("========================================================\n")

# Settings for Benchmark
n_vec <- c(50, 100, 200, 500, 1000)
n_iter_bench <- 1000 
n_burn_bench <- 500  

results_df <- data.frame()

for(n in n_vec) {
  cat(sprintf("\n--- Processing Sample Size n = %d ---\n", n))
  d <- generate_data_s3(n, m=20, seed=999)
  B <- get_bs_basis(d$t)
  Om <- create_penalty_matrix(ncol(B))
  
  # 1. Run Fast (Proposed)
  cat("  [BHIFR] Running...\n")
  t1_start <- Sys.time()
  fit1 <- BHIFR_Fast(d$Y_L, d$Y_U, d$X_L, d$X_U, B, Om, n_iter_bench, n_burn_bench, verbose=FALSE)
  t1_end <- Sys.time()
  t1_dur <- as.numeric(difftime(t1_end, t1_start, units="secs"))
  ess1 <- mean(effectiveSize(as.mcmc(fit1$lambda))) 
  cat(sprintf("  [BHIFR] Time: %.2f s, ESS: %.1f\n", t1_dur, ess1))
  
  # 2. Run Standard (Benchmark)
  cat("  [Standard] Running...\n")
  t2_start <- Sys.time()
  fit2 <- BHIFR_Standard(d$Y_L, d$Y_U, d$X_L, d$X_U, B, Om, n_iter_bench, n_burn_bench, verbose=TRUE)
  t2_end <- Sys.time()
  t2_dur <- as.numeric(difftime(t2_end, t2_start, units="secs"))
  ess2 <- mean(effectiveSize(as.mcmc(fit2$lambda)))
  cat(sprintf("  [Standard] Time: %.2f s, ESS: %.1f\n", t2_dur, ess2))
  
  # Store results
  results_df <- rbind(results_df, data.frame(
    N = n, Method = "BHIFR", Time = t1_dur, ESS = ess1, ESS_per_Sec = ess1/t1_dur
  ))
  results_df <- rbind(results_df, data.frame(
    N = n, Method = "Standard", Time = t2_dur, ESS = ess2, ESS_per_Sec = ess2/t2_dur
  ))
}

# --- Table 4 Generation ---
tab4_dat <- results_df %>% 
  pivot_wider(id_cols=N, names_from=Method, values_from=c(Time, ESS, ESS_per_Sec)) %>%
  mutate(
    Speedup_Ratio = Time_Standard / `Time_BHIFR`,
    ESS_Efficiency_Gain = `ESS_per_Sec_BHIFR` / `ESS_per_Sec_Standard`
  ) %>%
  select(N, `Time_BHIFR`, Time_Standard, Speedup_Ratio, `ESS_BHIFR`, ESS_Standard, ESS_Efficiency_Gain)

print(
  kbl(tab4_dat, format="latex", digits=2, booktabs=T, caption="Computational Efficiency Comparison") %>%
    kable_styling(latex_options = c("striped", "hold_position")) %>%
    add_header_above(c(" " = 1, "Runtime (sec)" = 2, " " = 1, "Avg ESS" = 2, " " = 1))
)