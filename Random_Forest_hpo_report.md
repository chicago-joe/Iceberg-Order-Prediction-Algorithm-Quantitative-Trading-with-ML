---
title: "Random Forest Optuna HPO Results"
author: "Joseph Loss"
date: "2025-04-21"
---
+++

## Best Trial
:label: random-forest-best-trial
<table>
   <tr>
      <th colspan="2" style="background: linear-gradient(90deg, #3a7b3a, #6fbd6f); color: white; text-align: center; padding: 8px;">Best Trial Performance</th>
   </tr>
   <tr>
      <td style="font-weight: bold; width: 40%;">Trial Number</td>
      <td style="text-align: right; font-family: monospace;">46</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">Performance Score</td>
      <td style="text-align: right; font-family: monospace; background: linear-gradient(90deg, #f2f8ec, #e0ecd0); font-weight: bold;">0.6648</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">Start Time</td>
      <td style="text-align: right; font-family: monospace;">2023-11-17 18:23:30</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">Completion Time</td>
      <td style="text-align: right; font-family: monospace;">2023-11-17 18:26:19</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">Execution Duration</td>
      <td style="text-align: right; font-family: monospace; font-style: italic;">2:48.91 (minutes:seconds)</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">Trial ID</td>
      <td style="text-align: right; font-family: monospace;">97</td>
   </tr>
</table>


## Best Parameters
:label: random-forest-best-parameters
<table>
   <tr>
      <th colspan="2" style="background: linear-gradient(90deg, #3a7b3a, #6fbd6f); color: white; text-align: center; padding: 8px;">Optimized Hyperparameters</th>
   </tr>
   <tr>
      <td style="font-weight: bold; width: 40%;">n_estimators</td>
      <td style="text-align: right; font-family: monospace; background-color: #e8f8e8; font-weight: bold;">500</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">max_depth</td>
      <td style="text-align: right; font-family: monospace; background-color: #e8f4f8;">4</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">min_samples_split</td>
      <td style="text-align: right; font-family: monospace; background-color: #f2f8e8;">7</td>
   </tr>
   <tr>
      <td style="font-weight: bold;">min_samples_leaf</td>
      <td style="text-align: right; font-family: monospace; background-color: #f2f8e8;">3</td>
   </tr>
   <tr>
      <td style="font-weight: bold; background-color: #f9f0e8;">train_size</td>
      <td style="text-align: right; font-family: monospace; background-color: #f9d29d; font-weight: bold;">2</td>
   </tr>
</table>


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
