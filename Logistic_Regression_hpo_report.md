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

### Parameter Importances
:::{figure} _references/hyperparameter-tuning/LogisticRegression/logistic_regression_plot_param_importances.png
:width: 80%
:class: with-shadow
:name: logistic_regression_param_importances

Parameter importance visualization
:::

[](xref:spec/logistic_regression_hpo#Parameter_Importances)
```{iframe} _references/hyperparameter-tuning/LogisticRegression/logistic_regression_plot_param_importances.html
:width: 100%
```

### Slice Plot
:::{figure} _references/hyperparameter-tuning/LogisticRegression/logistic_regression_plot_slice.png
:width: 80%
:class: with-shadow
:name: logistic_regression_slice_plot

Slice plot visualization
:::

[](xref:spec/logistic_regression_hpo#Slice_Plot)
```{iframe} _references/hyperparameter-tuning/LogisticRegression/logistic_regression_plot_slice.html
:width: 100%
```

### Parallel Coordinates Plot
:::{figure} _references/hyperparameter-tuning/LogisticRegression/logistic_regression_plot_parallel_coordinate.png
:width: 80%
:class: with-shadow
:name: logistic_regression_parallel_coordinates

Parallel coordinates visualization
:::

[](xref:spec/logistic_regression_hpo#Parallel_Coordinate)
```{iframe} _references/hyperparameter-tuning/LogisticRegression/logistic_regression_plot_parallel_coordinate.html
:width: 100%
```

### Rank Plot
:::{figure} _references/hyperparameter-tuning/LogisticRegression/logistic_regression_plot_rank.png
:width: 80%
:class: with-shadow
:name: logistic_regression_rank_plot

Rank plot visualization
:::

[](xref:spec/logistic_regression_hpo#Rank_Plot)
```{iframe} _references/hyperparameter-tuning/LogisticRegression/logistic_regression_plot_rank.html
:width: 100%
```

### Contour Plot
:::{figure} _references/hyperparameter-tuning/LogisticRegression/logistic_regression_plot_contour.png
:width: 80%
:class: with-shadow
:name: logistic_regression_contour_plot

Contour plot visualization
:::

[](xref:spec/logistic_regression_hpo#Contour_Plot)
```{iframe} _references/hyperparameter-tuning/LogisticRegression/logistic_regression_plot_contour.html
:width: 100%
```

### EDF Plot
:::{figure} _references/hyperparameter-tuning/LogisticRegression/logistic_regression_plot_edf.png
:width: 80%
:class: with-shadow
:name: logistic_regression_edf_plot

EDF plot visualization
:::

[](xref:spec/logistic_regression_hpo#EDF_Plot)
```{iframe} _references/hyperparameter-tuning/LogisticRegression/logistic_regression_plot_edf.html
:width: 100%
```

### Optimization History
:::{figure} _references/hyperparameter-tuning/LogisticRegression/logistic_regression_plot_optimization_history.png
:width: 80%
:class: with-shadow
:name: logistic_regression_optimization_history

Optimization history visualization
:::

[](xref:spec/logistic_regression_hpo#Optimization_History)
```{iframe} _references/hyperparameter-tuning/LogisticRegression/logistic_regression_plot_optimization_history.html
:width: 100%
```
