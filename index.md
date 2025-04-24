---
title: Hyperparameter Optimization and Model Evaluation - Iceberg Order Execution Model
abstract: |
    This HPO Tuning paper brings together reports on XGBoost, LightGBM,
    Random Forest, and Logistic Regression hyperparameter optimizations,
    along with accompanying notebook analyses. We compare performance
    metrics, runtime trade‑offs, and best practices for grid vs. Bayesian
    search approaches.
parts:
  supplemental_materials: |
    The following HPO reports provide detailed analyses of hyperparameter optimization 
    for various machine learning algorithms. Each report contains the best trial parameters, 
    performance metrics, and visualizations of the optimization process.
exports:
  - format: pdf
    template: lapreprint-typst
    output: exports/HPO-Iceberg-Order-Execution-Model-Joseph-Loss.pdf
    id: my-paper
    articles:
      - file: hyperparameter-optimization-whitepaper.md
        title: "Hyperparameter Optimization Whitepaper"
      - file: XGBoost_hpo_report.md
        title: "Appendix A: XGBoost HPO Report"
      - file: LightGBM_hpo_report.md
        title: "Appendix B: LightGBM HPO Report"
      - file: Random Forest_hpo_report.md
        title: "Appendix C: Random Forest HPO Report"
      - file: Logistic Regression_hpo_report.md
        title: "Appendix D: Logistic Regression HPO Report"
---

{toctree}
:caption: "HPO Tuning Paper Contents"
:maxdepth: 2
- hyperparameter-optimization-whitepaper.md
- XGBoost_hpo_report.md
- LightGBM_hpo_report.md
- Random Forest_hpo_report.md
- Logistic Regression_hpo_report.md
- hpo_plots.ipynb
