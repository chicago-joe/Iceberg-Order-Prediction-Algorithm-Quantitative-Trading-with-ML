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

+++

## Embedded Visualizations

### Parameter Importances
![logistic_regression_param_importances](/assets/logistic_regression_plot_param_importances.png)


### Slice Plot
![logistic_regression_slice_plot](/assets/logistic_regression_plot_slice.png)


### Parallel Coordinates Plot
![logistic_regression_parallel_coordinates](/assets/logistic_regression_plot_parallel_coordinate.png)


### Rank Plot
![logistic_regression_rank_plot](/assets/logistic_regression_plot_rank.png)


### Contour Plot
![logistic_regression_contour_plot](/assets/logistic_regression_plot_contour.png)


### EDF Plot
![logistic_regression_edf_plot](/assets/logistic_regression_plot_edf.png)


### Optimization History
![logistic_regression_optimization_history](/assets/logistic_regression_plot_optimization_history.png)
