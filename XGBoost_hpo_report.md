---
title: "XGBoost Optuna HPO Results"
author: "Joseph Loss"
date: "2025-04-21"
---

+++

## Best Trial
:label: xgboost-best-trial
<table>
   <tr>
      <th colspan="2" style="background: linear-gradient(90deg, #7b3a3a, #bd6f6f); color: white; text-align: center; padding: 8px;">Best Trial Performance</th>
   </tr>
   <tr>
      <td style="font-weight: bold; width: 40%;">Trial Number</td>
      <td style="text-align: right; font-family: monospace;">21</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">Performance Score</td>
      <td style="text-align: right; font-family: monospace; background: linear-gradient(90deg, #fdf6ec, #f9d29d); font-weight: bold;">0.6747</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">Start Time</td>
      <td style="text-align: right; font-family: monospace;">2023-11-17 23:02:38</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">Completion Time</td>
      <td style="text-align: right; font-family: monospace;">2023-11-17 23:08:38</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">Execution Duration</td>
      <td style="text-align: right; font-family: monospace; font-style: italic;">5:59.99 (minutes:seconds)</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">Trial ID</td>
      <td style="text-align: right; font-family: monospace;">122</td>
   </tr>
</table>


## Best Parameters
:label: xgboost-best-parameters
<table>
   <tr>
      <th colspan="2" style="background: linear-gradient(90deg, #7b3a3a, #bd6f6f); color: white; text-align: center; padding: 8px;">Optimized Hyperparameters</th>
   </tr>
   <tr>
      <td style="font-weight: bold; width: 40%;">eval_metric</td>
      <td style="text-align: right; font-family: monospace; background-color: #f8e8e8; font-weight: bold;">error@0.5</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">learning_rate</td>
      <td style="text-align: right; font-family: monospace; background-color: #f2f2f8;">0.03</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">n_estimators</td>
      <td style="text-align: right; font-family: monospace; background-color: #e8f8e8;">250</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">max_depth</td>
      <td style="text-align: right; font-family: monospace; background-color: #e8f4f8;">4</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">min_child_weight</td>
      <td style="text-align: right; font-family: monospace; background-color: #f2f8e8; font-weight: bold;">8</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">gamma</td>
      <td style="text-align: right; font-family: monospace; background-color: #f2e8f8;">0.2</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">subsample</td>
      <td style="text-align: right; font-family: monospace;">1.0</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">colsample_bytree</td>
      <td style="text-align: right; font-family: monospace;">0.8</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">reg_alpha</td>
      <td style="text-align: right; font-family: monospace; background-color: #f2e8f8;">0.2</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">reg_lambda</td>
      <td style="text-align: right; font-family: monospace; background-color: #f2e8f8; font-weight: bold;">2</td>
   </tr>
   <tr>
      <td style="font-weight: bold; background-color: #f9f0e8;">train_size</td>
      <td style="text-align: right; font-family: monospace; background-color: #f9d29d; font-weight: bold;">2</td>
   </tr>
</table>


+++

## Embedded Visualizations

### Parameter Importances
![xgboost_param_importances](/assets/xgboost_plot_param_importances.png)


### Slice Plot
![xgboost_slice_plot](/assets/xgboost_plot_slice.png)


### Parallel Coordinates Plot
![xgboost_parallel_coordinates](/assets/xgboost_plot_parallel_coordinate.png)


### Rank Plot
![xgboost_rank_plot](/assets/xgboost_plot_rank.png)


### Contour Plot
![xgboost_contour_plot](/assets/xgboost_plot_contour.png)


### EDF Plot
![xgboost_edf_plot](/assets/xgboost_plot_edf.png)


### Optimization History
![xgboost_optimization_history](/assets/xgboost_plot_optimization_history.png)
