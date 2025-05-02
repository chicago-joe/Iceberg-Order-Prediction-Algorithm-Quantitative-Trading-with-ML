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
  "min_./data_in_leaf": 100,
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


```{iframe} 

```

<!-- :::{figure} #LightGBM_plot_slice -->
<!-- :label: -->
<!-- ::: -->

<!-- ![](#LightGBM_plot_param_importances) - This will embed the output of a notebook cell -->
