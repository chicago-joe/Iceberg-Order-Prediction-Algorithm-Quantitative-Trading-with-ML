# HPO Report: `LightGBM`

## Best Trial
```json
{
  "_number": 49,
  "state": 1,
  "_values": [
    0.6745654203529987
  ],
  "_datetime_start": "2023-11-18 01:42:25.240274",
  "datetime_complete": "2023-11-18 01:42:59.724692",
  "_user_attrs": {},
  "_system_attrs": {},
  "intermediate_values": {},
  "_distributions": {
    "objective": "CategoricalDistribution(choices=('binary', 'regression'))",
    "learning_rate": "FloatDistribution(high=0.05, log=False, low=0.01, step=0.01)",
    "n_estimators": "CategoricalDistribution(choices=(100, 250, 500, 1000))",
    "max_depth": "IntDistribution(high=5, log=False, low=3, step=1)",
    "num_leaves": "CategoricalDistribution(choices=(2, 3, 7, 15, 31))",
    "min_sum_hessian_in_leaf": "CategoricalDistribution(choices=(0.001, 0.01, 0.1, 1, 10))",
    "extra_trees": "CategoricalDistribution(choices=(True, False))",
    "min_data_in_leaf": "IntDistribution(high=100, log=False, low=25, step=25)",
    "feature_fraction": "FloatDistribution(high=1.0, log=False, low=0.6, step=0.2)",
    "bagging_fraction": "FloatDistribution(high=1.0, log=False, low=0.6, step=0.2)",
    "bagging_freq": "CategoricalDistribution(choices=(0, 5, 10))",
    "lambda_l1": "CategoricalDistribution(choices=(0, 0.1, 1, 2))",
    "lambda_l2": "CategoricalDistribution(choices=(0, 0.1, 1, 2))",
    "min_gain_to_split": "CategoricalDistribution(choices=(0, 0.1, 0.5))",
    "train_size": "CategoricalDistribution(choices=(2, 3, 4, 5, 6, 7, 8, 9, 10))"
  },
  "_trial_id": 50
}
```

## Best Parameters
```json
{
  "objective": "regression",
  "learning_rate": 0.05,
  "n_estimators": 100,
  "max_depth": 4,
  "num_leaves": 31,
  "min_sum_hessian_in_leaf": 10,
  "extra_trees": true,
  "min_data_in_leaf": 100,
  "feature_fraction": 1.0,
  "bagging_fraction": 0.8,
  "bagging_freq": 0,
  "lambda_l1": 2,
  "lambda_l2": 0,
  "min_gain_to_split": 0.1,
  "train_size": 2
}
```

## Embedded Visualizations

### Parameter Importances
:::{figure} _references/hyperparameter-tuning/LightGBM/lightgbm_plot_param_importances.png
:width: 80%
:class: with-shadow
:name: lightgbm_param_importances

Parameter importance visualization
:::

[](xref:spec/lightgbm_hpo#Parameter_Importances)
```{iframe} _references/hyperparameter-tuning/LightGBM/lightgbm_plot_param_importances.html
:width: 100%
```

### Slice Plot
:::{figure} _references/hyperparameter-tuning/LightGBM/lightgbm_plot_slice.png
:width: 80%
:class: with-shadow
:name: lightgbm_slice_plot

Slice plot visualization
:::

[](xref:spec/lightgbm_hpo#Slice_Plot)
```{iframe} _references/hyperparameter-tuning/LightGBM/lightgbm_plot_slice.html
:width: 100%
```

### Parallel Coordinates Plot
:::{figure} _references/hyperparameter-tuning/LightGBM/lightgbm_plot_parallel_coordinate.png
:width: 80%
:class: with-shadow
:name: lightgbm_parallel_coordinates

Parallel coordinates visualization
:::

[](xref:spec/lightgbm_hpo#Parallel_Coordinate)
```{iframe} _references/hyperparameter-tuning/LightGBM/lightgbm_plot_parallel_coordinate.html
:width: 100%
```

### Rank Plot
:::{figure} _references/hyperparameter-tuning/LightGBM/lightgbm_plot_rank.png
:width: 80%
:class: with-shadow
:name: lightgbm_rank_plot

Rank plot visualization
:::

[](xref:spec/lightgbm_hpo#Rank_Plot)
```{iframe} _references/hyperparameter-tuning/LightGBM/lightgbm_plot_rank.html
:width: 100%
```

### Contour Plot
:::{figure} _references/hyperparameter-tuning/LightGBM/lightgbm_plot_contour.png
:width: 80%
:class: with-shadow
:name: lightgbm_contour_plot

Contour plot visualization
:::

[](xref:spec/lightgbm_hpo#Contour_Plot)
```{iframe} _references/hyperparameter-tuning/LightGBM/lightgbm_plot_contour.html
:width: 100%
```

### EDF Plot
:::{figure} _references/hyperparameter-tuning/LightGBM/lightgbm_plot_edf.png
:width: 80%
:class: with-shadow
:name: lightgbm_edf_plot

EDF plot visualization
:::

[](xref:spec/lightgbm_hpo#EDF_Plot)
```{iframe} _references/hyperparameter-tuning/LightGBM/lightgbm_plot_edf.html
:width: 100%
```

### Optimization History
:::{figure} _references/hyperparameter-tuning/LightGBM/lightgbm_plot_optimization_history.png
:width: 80%
:class: with-shadow
:name: lightgbm_optimization_history

Optimization history visualization
:::

[](xref:spec/lightgbm_hpo#Optimization_History)
```{iframe} _references/hyperparameter-tuning/LightGBM/lightgbm_plot_optimization_history.html
:width: 100%
```
