# HPO Report: `Logistic Regression`

## Best Trial
```json
{
  "_number": 26,
  "state": 1,
  "_values": [
    0.6899342878280169
  ],
  "_datetime_start": "2023-11-17 15:55:06.658366",
  "datetime_complete": "2023-11-17 15:56:22.403761",
  "_user_attrs": {},
  "_system_attrs": {},
  "intermediate_values": {},
  "_distributions": {
    "penalty": "CategoricalDistribution(choices=('l1', 'l2', 'elasticnet'))",
    "C": "CategoricalDistribution(choices=(0.01, 0.1, 1, 10, 100))",
    "solver": "CategoricalDistribution(choices=('saga',))",
    "max_iter": "CategoricalDistribution(choices=(100, 500, 1000))",
    "l1_ratio": "CategoricalDistribution(choices=(0, 0.5, 1))",
    "train_size": "CategoricalDistribution(choices=(2, 3, 4, 5, 6, 7, 8, 9, 10))"
  },
  "_trial_id": 177
}
```

## Best Parameters
```json
{
  "penalty": "elasticnet",
  "C": 0.01,
  "solver": "saga",
  "max_iter": 1000,
  "l1_ratio": 0.5,
  "train_size": 2
}
```

## Embedded Visualizations

```{include} "data/hyperparameter-optimization/Logistic Regression/visualizations/logistic_regression_plot_param_importances.html"

```

```{include} "data/hyperparameter-optimization/Logistic Regression/visualizations/logistic_regression_plot_plot_slice.html"

```

```{include} "data/hyperparameter-optimization/Logistic Regression/visualizations/logistic_regression_plot_parallel_coordinates.html"

```

```{include} "data/hyperparameter-optimization/Logistic Regression/visualizations/logistic_regression_plot_rank.html"

```

```{include} "data/hyperparameter-optimization/Logistic Regression/visualizations/logistic_regression_plot_contour.html"

```

```{include} "data/hyperparameter-optimization/Logistic Regression/visualizations/logistic_regression_plot_edf.html"

```

```{include} "data/hyperparameter-optimization/Logistic Regression/visualizations/logistic_regression_plot_optimization_history.html"

```
