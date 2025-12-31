# ==============================================================================
# BHIFR Sensitivity Analysis: Prior (A), Basis Dimension (K), and Nugget (delta)
# 
# Description:
#   Conducts a sensitivity analysis to assess the robustness of the BHIFR model 
#   against variations in hyperparameters:
#     1. Prior scale parameter A (Half-Cauchy)
#     2. Number of B-spline basis functions K
#     3. Nugget term delta (for numerical stability)
#
# Outputs: 
#   - Sensitivity Table: Comparison of posterior estimates and IMSPE
#   - Comparison Plot: Visualizing functional coefficient beta(t) robustness
# ==============================================================================

rm(list=ls())
set.seed(123) # Ensure global reproducibility

# ==============================================================================
# PART 0: Environment Setup & Helper Functions
# ==============================================================================
pkgs <- c("Rcpp", "RcppArmadillo", "splines2", "MASS", "dplyr", 
          "foreach", "doSNOW", "parallel", "tidyr", "ggplot2", "ggsci", "lubridate", "kableExtra")
new_pkgs <- pkgs[!(pkgs %in% installed.packages()[,"Package"])]
if(length(new_pkgs)) install.packages(new_pkgs)
invisible(lapply(pkgs, require, character.only = TRUE))

# ==============================================================================
# PART 1: C++ Core Algorithm Definition
# ==============================================================================
cpp_code_string <- '
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <cmath>

using namespace arma;
using namespace Rcpp;

// --- Helper Functions ---

// Robust Cholesky Decomposition
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

// Calculate Marginal Log-Likelihood for Lambda (Collapsed Gibbs)
double calc_marginal_loglik_lambda_dense(const vec& lambda, const mat& X_L, const mat& X_U,
                                         const mat& BTWB, const mat& BTY_proj, const mat& Prior_Prec) {
    int n = X_L.n_rows; int q = X_L.n_cols;
    mat P = ones(n, q + 1);
    for(int j=0; j<q; ++j) P.col(j+1) = (1.0 - lambda(j)) * X_L.col(j) + lambda(j) * X_U.col(j);
    
    mat PTP = P.t() * P;
    mat Lambda_beta = kron(PTP, BTWB) + Prior_Prec; 
    vec RHS = vectorise(BTY_proj * P);
    
    mat L;
    if(!robust_chol(L, Lambda_beta)) return -1e20;
    
    double log_det = 2.0 * sum(log(L.diag()));
    vec y = solve(trimatl(L), RHS);
    return -0.5 * log_det + 0.5 * dot(y, y);
}

// Calculate Joint Log-Likelihood for Variance Parameters
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

// Gradient for MALA Update
vec calc_gradient_gamma(const mat& E_L, const mat& E_U, const vec& sL, const vec& sU, double rho,
                        const mat& B_sig, const mat& Omega_sig, const vec& gamma_curr, double tau, bool is_lower) {
    int m = sL.n_elem; int n = E_L.n_rows;
    vec v_vec(m);
    double denom = 1.0 / (1.0 - rho*rho);
    for(int t=0; t<m; ++t) {
        double vL = sL(t); double vU = sU(t);
        double SSE_L = accu(pow(E_L.col(t), 2));
        double SSE_U = accu(pow(E_U.col(t), 2));
        double SCP = accu(E_L.col(t) % E_U.col(t));
        
        if(is_lower) v_vec(t) = denom * (SSE_L/(vL*vL) - rho*SCP/(vL*vU)) - n;
        else         v_vec(t) = denom * (SSE_U/(vU*vU) - rho*SCP/(vL*vU)) - n;
    }
    return B_sig.t() * v_vec - tau * Omega_sig * gamma_curr;
}

// Main MCMC Algorithm (Heteroscedastic Model) with Configurable Hyperparameters
// [[Rcpp::export]]
List BHIFR_Hetero_Cpp(mat Y_L, mat Y_U, mat X_L, mat X_U, mat B, mat Omega, mat B_sig, mat Omega_sig, 
                      int n_iter, int n_burnin, double prior_A_scale, double nugget_val) {
    
    int n=Y_L.n_rows; int m=Y_L.n_cols; int q=X_L.n_cols; int K=B.n_cols; int K_sig=B_sig.n_cols;
    int dim_beta = 2 * K * (q + 1);
    
    // Initializations
    vec lambda = randu(q) * 0.8 + 0.1; 
    vec beta = zeros(dim_beta);
    
    mat P_init=ones(n,q+1); for(int j=0;j<q;++j) P_init.col(j+1)=0.5*X_L.col(j)+0.5*X_U.col(j);
    mat D_init = kron(P_init, B); 
    int dim_single = K * (q + 1); 
    mat DTD = D_init.t() * D_init + 0.1 * eye(dim_single, dim_single); 
    vec bL_init = solve(DTD, D_init.t() * vectorise(Y_L.t()));
    vec bU_init = solve(DTD, D_init.t() * vectorise(Y_U.t()));
    for(int j=0; j<=q; ++j) {
        beta.subvec(j*2*K, j*2*K+K-1)        = bL_init.subvec(j*K, j*K+K-1);
        beta.subvec(j*2*K+K, j*2*K+2*K-1) = bU_init.subvec(j*K, j*K+K-1);
    }
    
    mat YL_h(n,m), YU_h(n,m); YL_h.zeros(); YU_h.zeros();
    vec aL=beta.head(K); vec aU=beta.subvec(K, 2*K-1); 
    YL_h.each_row() += (B*aL).t(); YU_h.each_row() += (B*aU).t();
    for(int j=0;j<q;++j) { 
        vec bL=beta.subvec((j+1)*2*K, (j+1)*2*K+K-1); vec bU=beta.subvec((j+1)*2*K+K, (j+1)*2*K+2*K-1);
        for(int i=0;i<n;++i) { YL_h.row(i)+=P_init(i,j+1)*(B*bL).t(); YU_h.row(i)+=P_init(i,j+1)*(B*bU).t(); } 
    }
    
    vec mean_log = arma::log(mean(pow(Y_L-YL_h, 2) + pow(Y_U-YU_h, 2), 0).t()/2.0 + 1e-6);
    vec gamma_L = solve(B_sig.t()*B_sig + 0.01*eye(K_sig,K_sig), B_sig.t()*mean_log);
    vec gamma_U = gamma_L; double rho=0.5;
    
    vec tau_beta = ones(2 * (q + 1)) * 10.0; vec nu_beta = ones(2 * (q + 1));
    double tau_gamma_L = 10.0, nu_gamma_L = 1.0;
    double tau_gamma_U = 10.0, nu_gamma_U = 1.0;
    
    vec step_l_vec = ones(q) * 0.1; double step_MALA = 0.002; double step_rho = 0.02;
    
    mat res_b(n_iter-n_burnin, dim_beta); mat res_l(n_iter-n_burnin, q); 
    mat res_gL(n_iter-n_burnin, K_sig); mat res_gU(n_iter-n_burnin, K_sig); vec res_r(n_iter-n_burnin);
    
    mat BTB=B.t()*B; mat BTY_L=B.t()*Y_L.t(); mat BTY_U=B.t()*Y_U.t(); mat BT=B.t();
    
    double A_sq = prior_A_scale * prior_A_scale;

    // MCMC Loop
    for(int it=0; it<n_iter; ++it) {
        double gamma_t = std::min(1.0, 10.0 / (double(it) + 100.0));
        vec sL = exp(B_sig*gamma_L); vec sU = exp(B_sig*gamma_U);
        vec sL_nug = sqrt(square(sL) + nugget_val); vec sU_nug = sqrt(square(sU) + nugget_val);
        
        double sc=1.0/(1.0-rho*rho);
        mat BTWB_acc = zeros(2*K,2*K); mat BTY_p = zeros(2*K,n);
        
        for(int t=0; t<m; ++t) {
             double vL = sL_nug(t); double vU = sU_nug(t);
             double w11=sc/(vL*vL); double w22=sc/(vU*vU); double w12=-sc*rho/(vL*vU);
             vec b=BT.col(t); mat bbt=b*b.t();
             BTWB_acc.submat(0,0,K-1,K-1)     += w11*bbt; BTWB_acc.submat(K,K,2*K-1,2*K-1) += w22*bbt;
             BTWB_acc.submat(0,K,K-1,2*K-1)   += w12*bbt; BTWB_acc.submat(K,0,2*K-1,K-1)   += w12*bbt;
             for(int k=0; k<K; ++k) { 
                 BTY_p.row(k)   += b(k)*(w11*Y_L.col(t).t()+w12*Y_U.col(t).t()); 
                 BTY_p.row(K+k) += b(k)*(w12*Y_L.col(t).t()+w22*Y_U.col(t).t()); 
             }
        }
        
        mat Prior_Prec = zeros(dim_beta, dim_beta);
        for(int g=0; g<2*(q+1); ++g) Prior_Prec.submat(g*K, g*K, g*K+K-1, g*K+K-1) = tau_beta(g) * Omega;
        
        double current_ll = calc_marginal_loglik_lambda_dense(lambda, X_L, X_U, BTWB_acc, BTY_p, Prior_Prec);
        
        // --- Step 1: Update Lambda ---
        for(int j=0; j<q; ++j) {
            double eta_curr = logit(lambda(j)); double eta_prop = eta_curr + step_l_vec(j) * randn();
            double lambda_prop_val = sigmoid(eta_prop);
            vec lambda_prop = lambda; lambda_prop(j) = lambda_prop_val;
            double prop_ll = calc_marginal_loglik_lambda_dense(lambda_prop, X_L, X_U, BTWB_acc, BTY_p, Prior_Prec);
            double jacobian_diff = std::log(lambda_prop_val * (1.0 - lambda_prop_val)) - std::log(lambda(j) * (1.0 - lambda(j)));
            bool accept = std::log(randu()) < (prop_ll - current_ll + jacobian_diff);
            if(accept) { lambda(j) = lambda_prop_val; current_ll = prop_ll; }
            if(it < n_burnin) {
                 double alpha = accept ? 1.0 : 0.0;
                 double log_step = std::log(step_l_vec(j)) + gamma_t * (alpha - 0.44);
                 step_l_vec(j) = std::exp(log_step);
            }
        }
        
        // --- Step 2: Update Beta ---
        mat P(n,q+1); P.col(0).fill(1.0); for(int j=0;j<q;++j) P.col(j+1)=(1-lambda(j))*X_L.col(j)+lambda(j)*X_U.col(j);
        mat PTP = P.t()*P;
        mat Prec_Beta = kron(PTP, BTWB_acc) + Prior_Prec;
        mat L_post;
        if(robust_chol(L_post, Prec_Beta)) {
             vec z = randn(dim_beta);
             vec RHS = vectorise(BTY_p * P);
             vec mean_part = solve(trimatu(L_post.t()), solve(trimatl(L_post), RHS));
             beta = mean_part + solve(trimatu(L_post.t()), z);
        }

        // --- Step 3: Update Hyperparameters ---
        for(int g=0; g < 2*(q+1); ++g) {
             vec b_sub = beta.subvec(g*K, g*K+K-1); double quad = as_scalar(b_sub.t() * Omega * b_sub);
             tau_beta(g) = R::rgamma(0.5*K+0.5, 1.0/(0.5*quad + 1.0/nu_beta(g)));
             nu_beta(g) = 1.0 / R::rgamma(1.0, 1.0 / (tau_beta(g) + 1.0/A_sq));
        }
        
        YL_h.zeros(); YU_h.zeros();
        aL=beta.head(K); aU=beta.subvec(K, 2*K-1); 
        YL_h.each_row() += (B*aL).t(); YU_h.each_row() += (B*aU).t();
        for(int j=0;j<q;++j) { 
            vec bL=beta.subvec((j+1)*2*K, (j+1)*2*K+K-1); vec bU=beta.subvec((j+1)*2*K+K, (j+1)*2*K+2*K-1);
            for(int i=0;i<n;++i) { YL_h.row(i)+=P(i,j+1)*(B*bL).t(); YU_h.row(i)+=P(i,j+1)*(B*bU).t(); } 
        }
        mat EL=Y_L-YL_h; mat EU=Y_U-YU_h;
        
        bool accepted_gL = false; bool accepted_gU = false;
        
        // --- Step 4: Update Variance Parameters (MALA) ---
        {
            double quad = as_scalar(gamma_L.t() * Omega_sig * gamma_L);
            tau_gamma_L = R::rgamma(0.5*K_sig+0.5, 1.0/(0.5*quad + 1.0/nu_gamma_L));
            nu_gamma_L = 1.0 / R::rgamma(1.0, 1.0 / (tau_gamma_L + 1.0/A_sq));
            
            vec g_L=calc_gradient_gamma(EL,EU,sL_nug,sU_nug,rho,B_sig,Omega_sig,gamma_L,tau_gamma_L,true);
            vec mu_L=gamma_L+0.5*step_MALA*step_MALA*g_L; vec gL_p=mu_L+step_MALA*randn(K_sig);
            
            vec sL_p = exp(B_sig*gL_p); vec sL_p_nug = sqrt(square(sL_p) + nugget_val); 
            
            vec grad_p=calc_gradient_gamma(EL,EU,sL_p_nug,sU_nug,rho,B_sig,Omega_sig,gL_p,tau_gamma_L,true);
            vec mu_L_rev=gL_p+0.5*step_MALA*step_MALA*grad_p;
            
            double lp_curr=calc_joint_loglik(EL,EU,sL_nug,sU_nug,rho)-0.5*tau_gamma_L*as_scalar(gamma_L.t()*Omega_sig*gamma_L);
            double lp_prop=calc_joint_loglik(EL,EU,sL_p_nug,sU_nug,rho)-0.5*tau_gamma_L*as_scalar(gL_p.t()*Omega_sig*gL_p);
            double lq_fwd= -0.5*dot(gL_p-mu_L,gL_p-mu_L)/(step_MALA*step_MALA);
            double lq_rev= -0.5*dot(gamma_L-mu_L_rev,gamma_L-mu_L_rev)/(step_MALA*step_MALA);
            
            if(std::log(randu()) < lp_prop - lp_curr + lq_rev - lq_fwd) { gamma_L=gL_p; sL=sL_p; sL_nug=sL_p_nug; accepted_gL = true; }
        }
        
        {
            double quad = as_scalar(gamma_U.t() * Omega_sig * gamma_U);
            tau_gamma_U = R::rgamma(0.5*K_sig+0.5, 1.0/(0.5*quad + 1.0/nu_gamma_U));
            nu_gamma_U = 1.0 / R::rgamma(1.0, 1.0 / (tau_gamma_U + 1.0/A_sq));
            
            vec g_U=calc_gradient_gamma(EL,EU,sL_nug,sU_nug,rho,B_sig,Omega_sig,gamma_U,tau_gamma_U,false);
            vec mu_U=gamma_U+0.5*step_MALA*step_MALA*g_U; vec gU_p=mu_U+step_MALA*randn(K_sig);
            
            vec sU_p = exp(B_sig*gU_p); vec sU_p_nug = sqrt(square(sU_p) + nugget_val);
            
            vec grad_p=calc_gradient_gamma(EL,EU,sL_nug,sU_p_nug,rho,B_sig,Omega_sig,gU_p,tau_gamma_U,false);
            vec mu_U_rev=gU_p+0.5*step_MALA*step_MALA*grad_p;
            
            double lp_curr=calc_joint_loglik(EL,EU,sL_nug,sU_nug,rho)-0.5*tau_gamma_U*as_scalar(gamma_U.t()*Omega_sig*gamma_U);
            double lp_prop=calc_joint_loglik(EL,EU,sL_nug,sU_p_nug,rho)-0.5*tau_gamma_U*as_scalar(gU_p.t()*Omega_sig*gU_p);
            double lq_fwd= -0.5*dot(gU_p-mu_U,gU_p-mu_U)/(step_MALA*step_MALA);
            double lq_rev= -0.5*dot(gamma_U-mu_U_rev,gamma_U-mu_U_rev)/(step_MALA*step_MALA);
            
            if(std::log(randu()) < lp_prop - lp_curr + lq_rev - lq_fwd) { gamma_U=gU_p; sU=sU_p; sU_nug=sU_p_nug; accepted_gU = true; }
        }
        
        if(it < n_burnin) {
            double alpha = (double)(accepted_gL + accepted_gU) / 2.0;
            double log_step = std::log(step_MALA) + gamma_t * (alpha - 0.574);
            step_MALA = std::exp(log_step);
        }
        
        // --- Step 5: Update Correlation (Rho) ---
        double z=fisher_z(rho); double z_p=z+step_rho*randn(); double r_p=inv_fisher_z(z_p);
        if(std::log(randu()) < calc_joint_loglik(EL,EU,sL_nug,sU_nug,r_p) - calc_joint_loglik(EL,EU,sL_nug,sU_nug,rho) + std::log(1-r_p*r_p) - std::log(1-rho*rho)) rho=r_p;
        
        // Storage
        if(it>=n_burnin) { 
            res_l.row(it-n_burnin)=lambda.t(); res_b.row(it-n_burnin)=beta.t(); 
            res_gL.row(it-n_burnin)=gamma_L.t(); res_gU.row(it-n_burnin)=gamma_U.t(); 
            res_r(it-n_burnin)=rho; 
        }
    }
    return List::create(Named("beta")=res_b, Named("lambda")=res_l, Named("alpha_L")=res_gL, Named("alpha_U")=res_gU, Named("rho")=res_r);
}
'

# Compile C++ code
cat("Compiling C++ Code...\n")
sourceCpp(code = cpp_code_string)

# ==============================================================================
# PART 2: Data Processing (Ireland Data) 
# ==============================================================================
cat("PART 2: Processing Ireland Data...\n")

cat("Current Working Directory:", getwd(), "\n")


Sys.setlocale("LC_TIME", "C")

zip_path <- file.path("..", "data", "aggregated_hourly_data.zip")
csv_name <- "aggregated_hourly_data.csv"

if(!file.exists(zip_path)) {
  stop(paste0("ERROR: Data file NOT found at: ", normalizePath(zip_path, mustWork = FALSE)))
}

raw_data <- read.csv(unz(zip_path, csv_name), stringsAsFactors = FALSE)
cat("Raw rows:", nrow(raw_data), "\n")

library(lubridate)
if(!"zoo" %in% installed.packages()[,"Package"]) install.packages("zoo")
library(zoo)

data_cleaned <- raw_data %>%
  mutate(
    # Include 'ymd' to correctly parse '2018/1/1' (midnight)
    datetime = parse_date_time(date, orders = c("ymd HMS", "ymd HM", "dmy HMS", "dmy HM", "ymd", "dmy")),
    day = as.Date(datetime), 
    hour = hour(datetime)
  ) %>%
  filter(!is.na(datetime))

cat("Rows after date parsing:", nrow(data_cleaned), "\n")

# 5. Data Filtering & Interpolation
data_interpolated <- data_cleaned %>%
  arrange(datetime) %>%
  dplyr::select(datetime, day, hour, temp, rhum, msl, wdsp) %>%
  mutate(
    across(where(is.numeric), ~zoo::na.approx(., na.rm = FALSE, rule = 2))
  ) %>%
  filter(!is.na(temp))
response_data <- data_interpolated %>%
  group_by(day, hour) %>%
  summarise(Temp_L = min(temp), Temp_U = max(temp), .groups = 'drop') %>%
  mutate(Temp_L=ifelse(is.infinite(Temp_L),NA,Temp_L), 
         Temp_U=ifelse(is.infinite(Temp_U),NA,Temp_U))
predictor_data <- data_interpolated %>%
  group_by(day) %>%
  summarise(
    rhum_L=min(rhum, na.rm=TRUE), rhum_U=max(rhum, na.rm=TRUE), 
    msl_L=min(msl, na.rm=TRUE), msl_U=max(msl, na.rm=TRUE), 
    wdsp_L=min(wdsp, na.rm=TRUE), wdsp_U=max(wdsp, na.rm=TRUE), 
    .groups='drop'
  ) %>%
  mutate(across(where(is.numeric), ~ifelse(is.infinite(.), NA, .)))

model_data <- inner_join(response_data, predictor_data, by = "day") %>%
  filter(year(day) >= 2018 & year(day) <= 2022) %>%
  na.omit() 

cat("Rows after merging & processing:", nrow(model_data), "\n")

complete_days <- model_data %>% group_by(day) %>% filter(n() == 24) %>% ungroup()
unique_days <- complete_days %>% distinct(day) %>% pull(day)

if(length(unique_days) == 0) {
  stop("ERROR: No days have complete 24-hour records.")
}

Y_L_final <- complete_days %>% filter(day %in% unique_days) %>% dplyr::select(day,hour,Temp_L) %>% pivot_wider(names_from=hour, values_from=Temp_L) %>% dplyr::select(-day) %>% as.matrix()
Y_U_final <- complete_days %>% filter(day %in% unique_days) %>% dplyr::select(day,hour,Temp_U) %>% pivot_wider(names_from=hour, values_from=Temp_U) %>% dplyr::select(-day) %>% as.matrix()

X_pred <- model_data %>% filter(day %in% unique_days) %>% dplyr::select(-hour,-Temp_L,-Temp_U) %>% distinct(day, .keep_all=TRUE) %>% arrange(day) %>% dplyr::select(-day)

# Standardizing
X_scaled <- scale(X_pred)
X_L_final_scaled <- X_scaled[, grep("_L$", colnames(X_scaled))]
X_U_final_scaled <- X_scaled[, grep("_U$", colnames(X_scaled))]

cat("Data Ready for Sensitivity Analysis: N =", nrow(Y_L_final), "days.\n")

set.seed(123)
idx_all <- 1:nrow(Y_L_final)

idx_test <- sample(idx_all, size = floor(0.2 * length(idx_all)))
idx_train <- setdiff(idx_all, idx_test)

YL_tr <- Y_L_final[idx_train,]; YL_te <- Y_L_final[idx_test,]
YU_tr <- Y_U_final[idx_train,]; YU_te <- Y_U_final[idx_test,]
XL_tr <- X_L_final_scaled[idx_train,]; XL_te <- X_L_final_scaled[idx_test,]
XU_tr <- X_U_final_scaled[idx_train,]; XU_te <- X_U_final_scaled[idx_test,]

# ==============================================================================
# PART 3: Automated Sensitivity Loop
# ==============================================================================

# Construct penalty matrix
create_penalty <- function(K) {
  D <- diff(diag(K), differences = 2)
  return(t(D) %*% D + diag(1e-6, K))
}

# Function to compute predictions
predict_val <- function(res, X_L_new, X_U_new, B) {
  K <- ncol(B); q <- ncol(X_L_new)
  l_hat <- colMeans(res$lambda)
  beta_mean <- colMeans(res$beta)
  muL <- beta_mean[1:K] %*% t(B); muU <- beta_mean[(K+1):(2*K)] %*% t(B)
  Y_L <- matrix(muL, nrow(X_L_new), ncol(muL), byrow=T)
  Y_U <- matrix(muU, nrow(X_L_new), ncol(muU), byrow=T)
  for(j in 1:q) {
    bL <- beta_mean[((j)*2*K+1):((j)*2*K+K)] %*% t(B)
    bU <- beta_mean[((j)*2*K+K+1):((j)*2*K+2*K)] %*% t(B)
    P <- (1-l_hat[j])*X_L_new[,j] + l_hat[j]*X_U_new[,j]
    Y_L <- Y_L + P%*%bL; Y_U <- Y_U + P%*%bU
  }
  return(list(L=Y_L, U=Y_U))
}

# Define sensitivity analysis scenarios
# Baseline: A=5, K=15, Nugget=1e-6
scenarios <- list(
  list(name="Baseline", K=15, A=5, nug=1e-6),
  
  # Prior Sensitivity
  list(name="Prior A=1", K=15, A=1, nug=1e-6),
  list(name="Prior A=25", K=15, A=25, nug=1e-6),
  
  # Basis Dimension Sensitivity
  list(name="Basis K=10", K=10, A=5, nug=1e-6),
  list(name="Basis K=20", K=20, A=5, nug=1e-6),
  
  # Nugget Sensitivity (New)
  list(name="Nugget 1e-4", K=15, A=5, nug=1e-4),
  list(name="Nugget 1e-8", K=15, A=5, nug=1e-8)
)

results_list <- list()
saved_beta_curves <- list() 

t_grid <- seq(0, 1, length.out=24)

# Main sensitivity analysis loop
for(scen in scenarios) {
  cat("\nRunning Scenario:", scen$name, "| K =", scen$K, "| A =", scen$A, "| Nug =", scen$nug, "...\n")
  
  # 1. Construct B-spline basis and penalty
  B_curr <- bSpline(t_grid, df=scen$K, degree=3, intercept=TRUE)
  O_curr <- create_penalty(ncol(B_curr))
  
  # 2. Run BHIFR MCMC algorithm
  set.seed(999)
  fit <- BHIFR_Hetero_Cpp(YL_tr, YU_tr, XL_tr, XU_tr, 
                          B_curr, O_curr, B_curr, O_curr, 
                          10000, 5000, 
                          as.double(scen$A), 
                          as.double(scen$nug)) # Pass nugget here
  
  # 3. Compute posterior means and prediction errors
  lam_mean <- colMeans(fit$lambda)
  pred <- predict_val(fit, XL_te, XU_te, B_curr)
  imspe_L <- mean((YL_te - pred$L)^2)
  imspe_U <- mean((YU_te - pred$U)^2)
  
  # 4. Extract functional coefficient for plotting (Baseline & K scenarios only)
  if(scen$name %in% c("Baseline", "Basis K=10", "Basis K=20")) {
    K <- scen$K
    idx_start <- 4*K + 1
    idx_end <- 5*K
    beta_pres_L_coefs <- colMeans(fit$beta[, idx_start:idx_end])
    beta_pres_curve <- as.vector(beta_pres_L_coefs %*% t(B_curr))
    saved_beta_curves[[scen$name]] <- beta_pres_curve
  }
  
  # Store results
  res_row <- data.frame(
    Scenario = scen$name,
    Lam_Hum = lam_mean[1],
    Lam_Pres = lam_mean[2],
    Lam_Wind = lam_mean[3],
    IMSPE_L = imspe_L,
    IMSPE_U = imspe_U
  )
  results_list[[scen$name]] <- res_row
  
  cat("  -> Finished. IMSPE_L:", round(imspe_L, 3), "\n")
}

# ==============================================================================
# PART 4: Generate Output Table (LaTeX)
# ==============================================================================
final_table <- do.call(rbind, results_list)
rownames(final_table) <- NULL

cat("\n=== Sensitivity Analysis Result Table ===\n")
print(final_table)

# Generate LaTeX Code
lat_tab <- kbl(final_table, format="latex", booktabs=T, digits=4, 
               caption="Sensitivity analysis results. Comparison of posterior weight estimates and prediction errors across different hyperparameter settings (Prior Scale A, Basis Dimension K, Nugget Term delta).") %>%
  kable_styling(latex_options = "striped")

print(lat_tab)

# ==============================================================================
# PART 5: Generate Comparison Plot (Beta Pressure)
# ==============================================================================
plot_df <- data.frame()
for(name in names(saved_beta_curves)) {
  tmp <- data.frame(
    Time = 0:23,
    Value = saved_beta_curves[[name]],
    Scenario = name
  )
  plot_df <- rbind(plot_df, tmp)
}

# Ensure factor order for legend
plot_df$Scenario <- factor(plot_df$Scenario, levels=c("Baseline", "Basis K=10", "Basis K=20"))

p_sens <- ggplot(plot_df, aes(x=Time, y=Value, color=Scenario, linetype=Scenario)) +
  geom_line(size=1) +
  scale_color_npg() +
  labs(x="Hour", y=expression(beta(t)~~(Pressure~Lower)), 
       title="Robustness of Functional Coefficient Estimation") +
  theme_classic() +
  theme(legend.position = "bottom")

ggsave("Sensitivity_Beta_Curve.pdf", p_sens, width=6, height=4)
ggsave("Sensitivity_Beta_Curve.eps", p_sens, width=6, height=4, device=cairo_ps)

cat("\nAll Done! Check 'Sensitivity_Beta_Curve.pdf' and console output for table.\n")