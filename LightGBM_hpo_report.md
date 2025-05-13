---
title: "LightGBM Optuna HPO Results"
author: "Joseph Loss"
date: "2025-04-21"
---
+++

## Best Trial
:label: lightgbm-best-trial
<table>
   <tr>
      <th colspan="2" style="background: linear-gradient(90deg, #3a4a7b, #6f8dbd); color: white; text-align: center; padding: 8px;">Best Trial Performance</th>
   </tr>
   <tr>
      <td style="font-weight: bold; width: 40%;">Trial Number</td>
      <td style="text-align: right; font-family: monospace;">49</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">Performance Score</td>
      <td style="text-align: right; font-family: monospace; background: linear-gradient(90deg, #fdf6ec, #f9d29d); font-weight: bold;">0.67460</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">Start Time</td>
      <td style="text-align: right; font-family: monospace;">2023-11-18 01:42:25</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">Completion Time</td>
      <td style="text-align: right; font-family: monospace;">2023-11-18 01:42:59</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">Execution Duration</td>
      <td style="text-align: right; font-family: monospace; font-style: italic;">34.48 seconds</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">Trial ID</td>
      <td style="text-align: right; font-family: monospace;">50</td>
   </tr>
</table>

## Best Parameters
:label: lightgbm-best-parameters
<table>
   <tr>
      <th colspan="2" style="background: linear-gradient(90deg, #3a4a7b, #6f8dbd); color: white; text-align: center; padding: 8px;">Optimized Hyperparameters</th>
   </tr>
   <tr>
      <td style="font-weight: bold; width: 40%;">objective</td>
      <td style="text-align: right; font-family: monospace; background-color: #e8f4f8;">regression</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">learning_rate</td>
      <td style="text-align: right; font-family: monospace;">0.05</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">n_estimators</td>
      <td style="text-align: right; font-family: monospace;">100</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">max_depth</td>
      <td style="text-align: right; font-family: monospace;">4</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">num_leaves</td>
      <td style="text-align: right; font-family: monospace; background-color: #f0f8e8;">31</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">min_sum_hessian_in_leaf</td>
      <td style="text-align: right; font-family: monospace;">10</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">extra_trees</td>
      <td style="text-align: right; font-family: monospace;">true</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">min_data_in_leaf</td>
      <td style="text-align: right; font-family: monospace; background-color: #f2f8e8; font-weight: bold;">100</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">feature_fraction</td>
      <td style="text-align: right; font-family: monospace;">1.0</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">bagging_fraction</td>
      <td style="text-align: right; font-family: monospace;">0.8</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">bagging_freq</td>
      <td style="text-align: right; font-family: monospace;">0</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">lambda_l1</td>
      <td style="text-align: right; font-family: monospace; background-color: #f2e8f8; font-weight: bold;">2</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">lambda_l2</td>
      <td style="text-align: right; font-family: monospace;">0</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">min_gain_to_split</td>
      <td style="text-align: right; font-family: monospace;">0.1</td>
   </tr>
   <tr>
      <td style="font-weight: bold; background-color: #f9f0e8;">train_size</td>
      <td style="text-align: right; font-family: monospace; background-color: #f9d29d; font-weight: bold;">2</td>
   </tr>
</table>


+++

## Embedded Visualizations

### Parameter Importances
![lightgbm_param_importances](/assets/lightgbm_plot_param_importances.*)


### Slice Plot
![lightgbm_slice_plot](/assets/lightgbm_plot_slice.*)


### Parallel Coordinates Plot
![lightgbm_parallel_coordinates](/assets/lightgbm_plot_parallel_coordinate.*)


### Rank Plot
![lightgbm_rank_plot](/assets/lightgbm_plot_rank.*)


### Contour Plot
![lightgbm_contour_plot](/assets/lightgbm_plot_contour.*)


### EDF Plot
![lightgbm_edf_plot](/assets/lightgbm_plot_edf.*)


### Optimization History
![lightgbm_optimization_history](/assets/lightgbm_plot_optimization_history.*)
