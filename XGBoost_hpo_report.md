# HPO Report: `XGBoost`

## Best Trial
```json
{
  "_number": 21,
  "state": 1,
  "_values": [
    0.6746555095729683
  ],
  "_datetime_start": "2023-11-17 23:02:38.875387",
  "datetime_complete": "2023-11-17 23:08:38.863139",
  "_user_attrs": {},
  "_system_attrs": {},
  "intermediate_values": {},
  "_distributions": {
    "eval_metric": "CategoricalDistribution(choices=('logloss', 'error@0.7', 'error@0.5'))",
    "learning_rate": "FloatDistribution(high=0.05, log=False, low=0.01, step=0.01)",
    "n_estimators": "CategoricalDistribution(choices=(100, 250, 500, 1000))",
    "max_depth": "IntDistribution(high=5, log=False, low=3, step=1)",
    "min_child_weight": "IntDistribution(high=10, log=False, low=5, step=1)",
    "gamma": "FloatDistribution(high=0.2, log=False, low=0.1, step=0.05)",
    "subsample": "FloatDistribution(high=1.0, log=False, low=0.8, step=0.1)",
    "colsample_bytree": "FloatDistribution(high=1.0, log=False, low=0.8, step=0.1)",
    "reg_alpha": "FloatDistribution(high=0.2, log=False, low=0.1, step=0.1)",
    "reg_lambda": "IntDistribution(high=3, log=False, low=1, step=1)",
    "train_size": "CategoricalDistribution(choices=(2, 3, 4, 5, 6, 7, 8, 9, 10))"
  },
  "_trial_id": 122
}
```

## Best Parameters
```json
{
  "eval_metric": "error@0.5",
  "learning_rate": 0.03,
  "n_estimators": 250,
  "max_depth": 4,
  "min_child_weight": 8,
  "gamma": 0.2,
  "subsample": 1.0,
  "colsample_bytree": 0.8,
  "reg_alpha": 0.2,
  "reg_lambda": 2,
  "train_size": 2
}
```

## Embedded Visualizations

```{include} "./data/hyperparameter-optimization/XGBoost/visualizations/xgboost_plot_param_importances.html"

```

```{include} "./data/hyperparameter-optimization/XGBoost/visualizations/xgboost_plot_plot_slice.html"

```

```{include} "./data/hyperparameter-optimization/XGBoost/visualizations/xgboost_plot_parallel_coordinates.html"

```

```{include} "./data/hyperparameter-optimization/XGBoost/visualizations/xgboost_plot_rank.html"

```

```{include} "./data/hyperparameter-optimization/XGBoost/visualizations/xgboost_plot_contour.html"

```

```{include} "./data/hyperparameter-optimization/XGBoost/visualizations/xgboost_plot_edf.html"

```

```{include} "./data/hyperparameter-optimization/XGBoost/visualizations/xgboost_plot_optimization_history.html"

```
