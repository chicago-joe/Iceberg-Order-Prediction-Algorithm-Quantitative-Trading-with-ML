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

+++

## Embedded Visualizations

### Parameter Importances
![random_forest_param_importances](/assets/random_forest_plot_param_importances.png)


### Slice Plot
![random_forest_slice_plot](/assets/random_forest_plot_slice.png)


### Parallel Coordinates Plot
![random_forest_parallel_coordinates](/assets/random_forest_plot_parallel_coordinate.png)


### Rank Plot
![random_forest_rank_plot](/assets/random_forest_plot_rank.png)


### Contour Plot
![random_forest_contour_plot](/assets/random_forest_plot_contour.png)


### EDF Plot
![random_forest_edf_plot](/assets/random_forest_plot_edf.png)


### Optimization History
![random_forest_optimization_history](/assets/random_forest_plot_optimization_history.png)
