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

### Parameter Importances
:::{figure} _references/hyperparameter-tuning/XGBoost/xgboost_plot_param_importances.png
:width: 80%
:class: with-shadow
:name: xgboost_param_importances

Parameter importance visualization
:::

[](xref:spec/xgboost_hpo#Parameter_Importances)
```{iframe} _references/hyperparameter-tuning/XGBoost/xgboost_plot_param_importances.html
:width: 100%
```

### Slice Plot
:::{figure} _references/hyperparameter-tuning/XGBoost/xgboost_plot_slice.png
:width: 80%
:class: with-shadow
:name: xgboost_slice_plot

Slice plot visualization
:::

[](xref:spec/xgboost_hpo#Slice_Plot)
```{iframe} _references/hyperparameter-tuning/XGBoost/xgboost_plot_slice.html
:width: 100%
```

### Parallel Coordinates Plot
:::{figure} _references/hyperparameter-tuning/XGBoost/xgboost_plot_parallel_coordinate.png
:width: 80%
:class: with-shadow
:name: xgboost_parallel_coordinates

Parallel coordinates visualization
:::

[](xref:spec/xgboost_hpo#Parallel_Coordinate)
```{iframe} _references/hyperparameter-tuning/XGBoost/xgboost_plot_parallel_coordinate.html
:width: 100%
```

### Rank Plot
:::{figure} _references/hyperparameter-tuning/XGBoost/xgboost_plot_rank.png
:width: 80%
:class: with-shadow
:name: xgboost_rank_plot

Rank plot visualization
:::

[](xref:spec/xgboost_hpo#Rank_Plot)
```{iframe} _references/hyperparameter-tuning/XGBoost/xgboost_plot_rank.html
:width: 100%
```

### Contour Plot
:::{figure} _references/hyperparameter-tuning/XGBoost/xgboost_plot_contour.png
:width: 80%
:class: with-shadow
:name: xgboost_contour_plot

Contour plot visualization
:::

[](xref:spec/xgboost_hpo#Contour_Plot)
```{iframe} _references/hyperparameter-tuning/XGBoost/xgboost_plot_contour.html
:width: 100%
```

### EDF Plot
:::{figure} _references/hyperparameter-tuning/XGBoost/xgboost_plot_edf.png
:width: 80%
:class: with-shadow
:name: xgboost_edf_plot

EDF plot visualization
:::

[](xref:spec/xgboost_hpo#EDF_Plot)
```{iframe} _references/hyperparameter-tuning/XGBoost/xgboost_plot_edf.html
:width: 100%
```

### Optimization History
:::{figure} _references/hyperparameter-tuning/XGBoost/xgboost_plot_optimization_history.png
:width: 80%
:class: with-shadow
:name: xgboost_optimization_history

Optimization history visualization
:::

[](xref:spec/xgboost_hpo#Optimization_History)
```{iframe} _references/hyperparameter-tuning/XGBoost/xgboost_plot_optimization_history.html
:width: 100%
```
