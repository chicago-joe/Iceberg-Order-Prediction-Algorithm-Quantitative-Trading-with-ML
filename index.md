---
title: "Machine Learning for Quantitative Trading"
author: "Joseph Loss"
date: "2025-05-01"
abstract: |
    This research collection presents two complementary papers on advanced machine learning 
    applications in quantitative trading. The first paper introduces a novel approach for 
    predicting iceberg order execution using XGBoost models, achieving 79% precision by 
    analyzing market microstructure patterns. The second paper systematically explores 
    hyperparameter optimization strategies for these prediction models, revealing that shorter 
    training windows (just two time periods) consistently outperform longer historical datasets 
    across all tested algorithms. Together, these works demonstrate how carefully optimized 
    machine learning models can extract predictive signals from market data, quantify execution 
    uncertainty, and create adaptive trading strategies that respond dynamically to changing 
    market conditions. The included model comparison reports provide detailed optimization 
    results for XGBoost, LightGBM, Random Forest, and Logistic Regression, with the surprising 
    finding that simpler Logistic Regression models achieved the highest overall performance 
    (0.6899) when properly regularized.
---


This paper presents a machine learning approach for predicting iceberg order execution in quantitative trading. We analyze market microstructure patterns to predict whether detected iceberg orders will be filled or canceled, providing valuable signals for algorithmic trading strategies.

### [![Iceberg Order Prediction](assets/complete_iceberg.png)](./iceberg-prediction-whitepaper-v2.md)

### [Hyperparameter Optimization](./hyperparameter-optimization-whitepaper.md)

This comprehensive study examines hyperparameter optimization for machine learning models that predict iceberg order execution. The paper includes model comparison and detailed optimization results for several algorithms:
- [XGBoost](./XGBoost_hpo_report.md)
- [LightGBM](./LightGBM_hpo_report.md)
- [Random Forest](./Random_Forest_hpo_report.md)
- [Logistic Regression](./Logistic_Regression_hpo_report.md)
