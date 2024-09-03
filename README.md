# Risk Analysis with MATLAB: Portfolio VaR Estimation and Risk Parity Portfolio

This repository contains MATLAB code for performing statistical analysis, Value at Risk (VaR) estimation, and constructing a Risk Parity Portfolio as part of the SMM272 - Risk Analysis coursework in the MSc in Financial Mathematics, Mathematical Trading & Finance, and Quantitative Finance programs at Bayes Business School, City, University of London. The project was completed under the guidance of Prof. Gianluca Fusai.

## Project Overview
### Part 1: Statistical Analysis and Value at Risk (VaR) Estimation
Objective: Analyze an equally weighted portfolio composed of five stocks—Intel (INTC), JPMorgan Chase (JPM), Alcoa (AA), Procter & Gamble (PG), and Microsoft (MSFT)—using historical data from January 1, 2014, to December 31, 2023. The analysis includes:

1.) Statistical Analysis:

- Descriptive Statistics: Calculation of mean, variance, standard deviation, skewness, kurtosis, and other statistical measures of the portfolio's returns.
- Normality Testing: Examination of the distribution of portfolio returns using histograms, QQ plots, and the Jarque-Bera test for normality.
- Autocorrelation Analysis: Evaluation of autocorrelation in returns and squared returns to identify potential serial dependencies.

2.) Value at Risk (VaR) Estimation:

- Methods Used:
  - Gaussian Parametric Approach
  - Historical Simulation with Bootstrapping
  - Student's T-distribution using Maximum Likelihood Estimation (MLE)

- Backtesting:
  - Number of VaR Violations: Comparison of actual losses to estimated VaR at 90% and 99% confidence levels.
  - Kupiec Test: Likelihood ratio test to assess the accuracy of the VaR models.
  - Conditional Coverage Test: Evaluation of both the independence of VaR violations and the adequacy of the number of violations.
  - Kuiper Test: Used to check the uniformity of transformed probabilities in different VaR models.

### Part 2: Risk Parity Portfolio Construction and Evaluation
Objective: Construct a Risk Parity Portfolio using the parametric approach and evaluate its performance compared to an equally weighted portfolio. The analysis includes:

1.) Risk Parity Portfolio Construction:

- Method: The portfolio is constructed by minimizing the dispersion of Component VaRs (CVaR) across assets using the sample covariance matrix of the first half of the data (January 1, 2014, to mid-2018).
- Output: Optimal portfolio weights that equalize the CVaR of each asset.

2.) Performance Comparison:

- Return Series: Comparison of the cumulative returns of the Risk Parity Portfolio and an equally weighted portfolio using the second half of the data (mid-2018 to December 31, 2023).
- Performance Metrics:
  - Sharpe Ratio: Measures the risk-adjusted return of each portfolio.
  - Maximum Drawdown: The largest peak-to-trough decline in the portfolio value.
  - VaR Violations: Number of instances where actual losses exceeded the estimated VaR at a 95% confidence level.

### Repository Contents
SMM272_part_1.m: MATLAB code for Part 1, including statistical analysis, VaR estimation, and backtesting.
SMM272_part_2.m: MATLAB code for Part 2, covering the construction of a Risk Parity Portfolio and its performance comparison with an equally weighted portfolio.
