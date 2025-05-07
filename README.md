# 📊 Machine Learning for Quantitative Trading: Iceberg Order Prediction

[![View Live Site](https://img.shields.io/badge/Live%20Site-Visit-blue)](https://chicago-joe.github.io/Iceberg-Order-Prediction-Algorithm-Quantitative-Trading-with-ML/) 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/chicago-joe/Iceberg-Order-Prediction-Algorithm-Quantitative-Trading-with-ML/HEAD?urlpath=lab)
[![GitHub Stars](https://img.shields.io/github/stars/chicago-joe/Iceberg-Order-Prediction-Algorithm-Quantitative-Trading-with-ML?style=social)](https://github.com/chicago-joe/Iceberg-Order-Prediction-Algorithm-Quantitative-Trading-with-ML/stargazers)
<!-- 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
-->

## 📋 Overview

This repository contains comprehensive research on applying machine learning to predict iceberg order execution in quantitative trading. My work demonstrates how ML can transform trading strategies by extracting predictive signals from market microstructure data.

### 🔍 Key Research Findings

- **Predictive Power**: My models achieve 67% precision in predicting whether detected iceberg orders will be filled or canceled
- **Counter-Intuitive Discovery**: Shorter training windows (just two time periods) consistently outperform longer historical datasets across all tested algorithms
- **Model Comparison**: Systematic evaluation of XGBoost, LightGBM, Random Forest, and Logistic Regression models with surprising results

## 📑 Research Papers

### [1. Iceberg Order Prediction: A Machine Learning Approach](./iceberg-prediction-whitepaper-v2.md)

This paper presents a novel method for predicting iceberg order execution using an XGBoost-based model:

- Analysis of market microstructure patterns to identify execution signals
- Implementation of time-series cross-validation to prevent look-ahead bias
- Translation of prediction probabilities into actionable trading decisions
- Feature importance analysis revealing key market predictors

### [2. Hyperparameter Optimization for Trading Models](./hyperparameter-optimization-whitepaper.md)

My newest research challenges conventional wisdom in financial machine learning:

- Systematic approach to hyperparameter optimization for ML models in trading
- Comprehensive comparison of model types across extensive parameter spaces
- Surprising discovery that shorter training windows consistently outperform longer datasets
- Evidence that recent market data contains more predictive value than extended history
- Practical optimization strategies for trading applications

## 📊 Model Reports

Detailed optimization results for various algorithms:
- [XGBoost Optimization Results](./XGBoost_hpo_report.md)
- [LightGBM Optimization Results](./LightGBM_hpo_report.md)
- [Random Forest Optimization Results](./Random_Forest_hpo_report.md)
- [Logistic Regression Optimization Results](./Logistic_Regression_hpo_report.md)

## 🧠 Key Innovations

- **Side-Relative Feature Transformations**: Converting raw market features into consistent signals for both buy and sell orders
- **Custom Trading Metrics**: Trading-specific performance measures that balance precision and minimum recall thresholds
- **Time-Series Validation**: Implementation that respects temporal boundaries in financial data
- **Adaptive Confidence Bands**: Translating probability outputs into dynamic trading decisions

## 🚀 Getting Started

- Use the Binder links above to explore the notebooks interactively without local setup
- Visit the [full documentation site](https://chicago-joe.github.io/Iceberg-Order-Prediction-Algorithm-Quantitative-Trading-with-ML/) for comprehensive details

## 📈 Trading Implications

My research offers significant implications for trading system design:

- **Frequent Retraining**: Prioritize frequent model retraining on recent data over accumulating larger historical datasets
- **Regularization Focus**: Parameters controlling model complexity are critical for robust performance
- **Market Microstructure**: Order book positioning relative to key levels provides powerful predictive signals
- **Confidence-Based Execution**: Leverage prediction probabilities to make nuanced trading decisions

## 👨‍💻 About the Author

[Joseph Loss](https://github.com/chicago-joe) - Quantitative Developer

<!-- 
## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
-->

## 🔗 Citation

If you use this research in your work, please cite:

```
Loss, J. (2025). Machine Learning for Quantitative Trading: Iceberg Order Prediction.
https://github.com/chicago-joe/Iceberg-Order-Prediction-Algorithm-Quantitative-Trading-with-ML
```

---

*Star this repo and join me in exploring the future of quantitative trading!*
[![View Live Site](https://img.shields.io/badge/Live%20Site-Visit-blue)](https://chicago-joe.github.io/Iceberg-Order-Prediction-Algorithm-Quantitative-Trading-with-ML/) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/chicago-joe/Iceberg-Order-Prediction-Algorithm-Quantitative-Trading-with-ML/HEAD?urlpath=lab)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/chicago-joe/Iceberg-Order-Prediction-Algorithm-Quantitative-Trading-with-ML/main)
