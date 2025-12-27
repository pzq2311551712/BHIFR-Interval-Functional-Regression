# BHIFR: Efficient Bayesian Hierarchical Regression for Interval-Valued Functional Data

This repository contains the R implementation and data to reproduce the results of the paper:

> **Title:** Efficient Bayesian Hierarchical Regression for Interval-Valued Functional Data
> **Authors:** Chunjing Li, Zhenqiang Pang, Xiaohui Yuan
> **Journal:** [Target: Statistics and Computing]

## 🚀 Overview

**BHIFR** (Bayesian Hierarchical Interval-Valued Functional Regression) is a unified framework for modeling interval-valued functional response data against interval-valued scalar covariates. This approach effectively captures boundary dependencies and heteroscedasticity while addressing multicollinearity through hierarchical regularization.

### Key Methodological Contributions:
1.  **Parametrized Method**: Data-driven weight learning ($\lambda$) for interval covariates, overcoming limitations of fixed Center/Min-Max methods.
2.  **Heteroscedasticity**: Explicit modeling of time-varying variance $\sigma^2(t)$ using log-linear B-splines to capture dynamic volatility.
3.  **Computational Efficiency**: Implements a hybrid MCMC algorithm with **Kronecker Product Acceleration** and **Collapsed Gibbs Sampling**, reducing computational complexity from cubic $O(n^3)$ to linear scale.

## 📂 Repository Structure

The repository is organized as follows:

```text
BHIFR-Interval-Functional-Regression/
├── application/              # Empirical analysis on Ireland weather data
│   └── Ireland_Application.R # Main script for real data application
├── data/                     # Dataset required for the application
│   └── aggregated_hourly_data.zip # Processed weather data (see Data Note below)
├── simulation/               # Simulation studies verifying the methodology
│   ├── Study1_Homoscedastic.R   # Method comparison under homoscedasticity
│   ├── Study2_Heteroscedastic.R # Validation of heteroscedastic model & MALA
│   ├── Study3_Efficiency.R      # Computational benchmarking (Fast vs Standard)
│   └── Sensitivity_Analysis.R   # Robustness check for hyperparameters
├── LICENSE                   # MIT License
└── README.md                 # Project documentation

```

## 💻 How to Run

### 1. Prerequisites

The code is written in **R**. It relies on `Rcpp` and `RcppArmadillo` for high-performance C++ integration. Please ensure the following packages are installed:

```r
install.packages(c("Rcpp", "RcppArmadillo", "splines2", "MASS", "dplyr", 
                   "knitr", "foreach", "doSNOW", "parallel", "tidyr", 
                   "kableExtra", "ggplot2", "patchwork", "ggsci", "lubridate", 
                   "scales", "bayesplot", "coda"))

```

### 2. Reproducing Simulation Results

To reproduce the tables and figures from the simulation section of the paper:

* **Study 1 (Homoscedasticity)**: Run `simulation/Study1_Homoscedastic.R`.
* **Study 2 (Heteroscedasticity)**: Run `simulation/Study2_Heteroscedastic.R`.
* **Efficiency Benchmark**: Run `simulation/Study3_Efficiency.R`.
* **Sensitivity Analysis**: Run `simulation/Sensitivity_Analysis.R`.

### 3. Reproducing Empirical Analysis

To reproduce the application results (Ireland Weather Data):

1. Navigate to the `application/` folder or open the R project.
2. Run `Ireland_Application.R`.
* *Note: The script is designed to automatically read the compressed dataset (`.zip`) located in the `data/` folder without manual extraction.*



## ⚠️ Data Availability Note

The dataset provided in `data/aggregated_hourly_data.zip` is a **subset** of the original Irish Weather Data.

* **Time Range**: 2018-01-01 to 2022-12-31.
* **Preprocessing**: Historical records prior to 2018 have been removed to:
1. Comply with GitHub's file size limits (< 25MB).
2. Match the specific analysis period focused on in the paper.


* **Original Source**: The full dataset is publicly available from Met Éireann or the [Kaggle Repository](https://www.kaggle.com/datasets/conorrot/irish-weather-hourly-data).

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## 📝 Citation

If you use this code or data in your research, please cite our paper:

> Li, C., Pang, Z., & Yuan, X. (2025). Efficient Bayesian Hierarchical Regression for Interval-Valued Functional Data. *Statistics and Computing* (Under Review).
