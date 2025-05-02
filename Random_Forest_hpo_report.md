# HPO Report: `RandomForest`

## Best Trial
```json
{
  "_number": 46,
  "state": 1,
  "_values": [
    0.6648064178583886
  ],
  "_datetime_start": "2023-11-17 18:23:30.396578",
  "datetime_complete": "2023-11-17 18:26:19.310797",
  "_user_attrs": {},
  "_system_attrs": {},
  "intermediate_values": {},
  "_distributions": {
    "n_estimators": "CategoricalDistribution(choices=(100, 250, 500, 1000))",
    "max_depth": "IntDistribution(high=4, log=False, low=2, step=1)",
    "min_samples_split": "IntDistribution(high=10, log=False, low=5, step=1)",
    "min_samples_leaf": "IntDistribution(high=5, log=False, low=3, step=1)",
    "train_size": "CategoricalDistribution(choices=(2, 3, 4, 5, 6, 7, 8, 9, 10))"
  },
  "_trial_id": 97
}
```

## Best Parameters
```json
{
  "n_estimators": 500,
  "max_depth": 4,
  "min_samples_split": 7,
  "min_samples_leaf": 3,
  "train_size": 2
}
```

## Embedded Visualizations

### Parameter Importances
:::{figure} _references/hyperparameter-tuning/RandomForest/random_forest_plot_param_importances.png
:width: 80%
:class: with-shadow
:name: random_forest_param_importances

Parameter importance visualization
:::

[](xref:spec/randomforest_hpo#Parameter_Importances)
```{iframe} _references/hyperparameter-tuning/RandomForest/random_forest_plot_param_importances.html
:width: 100%
```

### Slice Plot
:::{figure} _references/hyperparameter-tuning/RandomForest/random_forest_plot_slice.png
:width: 80%
:class: with-shadow
:name: random_forest_slice_plot

Slice plot visualization
:::

[](xref:spec/randomforest_hpo#Slice_Plot)
```{iframe} _references/hyperparameter-tuning/RandomForest/random_forest_plot_slice.html
:width: 100%
```

### Parallel Coordinates Plot
:::{figure} _references/hyperparameter-tuning/RandomForest/random_forest_plot_parallel_coordinate.png
:width: 80%
:class: with-shadow
:name: random_forest_parallel_coordinates

Parallel coordinates visualization
:::

[](xref:spec/randomforest_hpo#Parallel_Coordinate)
```{iframe} _references/hyperparameter-tuning/RandomForest/random_forest_plot_parallel_coordinate.html
:width: 100%
```

### Rank Plot
:::{figure} _references/hyperparameter-tuning/RandomForest/random_forest_plot_rank.png
:width: 80%
:class: with-shadow
:name: random_forest_rank_plot

Rank plot visualization
:::

[](xref:spec/randomforest_hpo#Rank_Plot)
```{iframe} _references/hyperparameter-tuning/RandomForest/random_forest_plot_rank.html
:width: 100%
```

### Contour Plot
:::{figure} _references/hyperparameter-tuning/RandomForest/random_forest_plot_contour.png
:width: 80%
:class: with-shadow
:name: random_forest_contour_plot

Contour plot visualization
:::

[](xref:spec/randomforest_hpo#Contour_Plot)
```{iframe} _references/hyperparameter-tuning/RandomForest/random_forest_plot_contour.html
:width: 100%
```

### EDF Plot
:::{figure} _references/hyperparameter-tuning/RandomForest/random_forest_plot_edf.png
:width: 80%
:class: with-shadow
:name: random_forest_edf_plot

EDF plot visualization
:::

[](xref:spec/randomforest_hpo#EDF_Plot)
```{iframe} _references/hyperparameter-tuning/RandomForest/random_forest_plot_edf.html
:width: 100%
```

### Optimization History
:::{figure} _references/hyperparameter-tuning/RandomForest/random_forest_plot_optimization_history.png
:width: 80%
:class: with-shadow
:name: random_forest_optimization_history

Optimization history visualization
:::

[](xref:spec/randomforest_hpo#Optimization_History)
```{iframe} _references/hyperparameter-tuning/RandomForest/random_forest_plot_optimization_history.html
:width: 100%
```
