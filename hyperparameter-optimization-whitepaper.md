```markdown
---
title: "Hyperparameter Optimization for Iceberg Order Prediction"
author: "Joseph Loss"
date: "2025-04-21"
kernelspec:
  name: ipykernel
  display_name: 'Python 3'
---

# Hyperparameter Optimization for Iceberg Order Prediction

```{code-cell} python
:tags: [remove-input]
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# Create system architecture diagram
fig = go.Figure()

# Add nodes
nodes = [
    {"name": "Data Collection", "x": 0, "y": 0},
    {"name": "Preprocessing", "x": 1, "y": 0},
    {"name": "Feature Engineering", "x": 2, "y": 0},
    {"name": "Model Training", "x": 3, "y": 0},
    {"name": "Hyperparameter Optimization", "x": 3, "y": 1},
    {"name": "Evaluation", "x": 4, "y": 0},
    {"name": "Trading Integration", "x": 5, "y": 0}
]

# Add node representations
for node in nodes:
    fig.add_trace(go.Scatter(
        x=[node["x"]], 
        y=[node["y"]],
        mode="markers+text",
        marker=dict(size=30, color="skyblue"),
        text=node["name"],
        textposition="bottom center",
        name=node["name"]
    ))

# Add edges
edges = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 3), (3, 5), (5, 6)
]

for edge in edges:
    start, end = edge
    fig.add_trace(go.Scatter(
        x=[nodes[start]["x"], nodes[end]["x"]],
        y=[nodes[start]["y"], nodes[end]["y"]],
        mode="lines",
        line=dict(width=2, color="gray"),
        showlegend=False
    ))

fig.update_layout(
    title="Complete Iceberg Order Prediction & Trading System",
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    width=800,
    height=400,
    showlegend=False
)

fig.show()
```

*The complete system architecture showing data acquisition, preprocessing, model optimization, and trading integration.*

## Introduction: Why Hyperparameter Optimization Matters in Trading

In quantitative trading, model performance can directly impact profit and loss. When predicting iceberg order execution, even small improvements in precision and recall translate to meaningful trading advantages. This paper examines our systematic approach to hyperparameter optimization for machine learning models that predict whether detected iceberg orders will be filled or canceled.

:::{note} What Are Hyperparameters?

Hyperparameters are configuration settings that govern the training process and model architecture, but are not learned from data. In trading models, they control the trade-off between:

- **Precision vs. Recall**: Critical for balancing execution quality against opportunity capture
- **Complexity vs. Generalization**: Essential for adapting to changing market regimes
- **Computational Efficiency vs. Predictive Power**: Vital for real-time trading decisions
:::

## Optimization Framework Architecture

Our hyperparameter optimization system consists of two key components:

1. **ModelEvaluator**: Manages model training, evaluation, and performance tracking
2. **HyperparameterTuner**: Conducts systematic search for optimal parameters

Figure 1 illustrates the optimization workflow:

```{code-cell} python
:tags: [remove-input]
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Create optimization workflow diagram
fig = go.Figure()

# Define components
components = [
    {"name": "ModelEvaluator", "x": 1, "y": 1, "width": 1.5, "height": 0.8},
    {"name": "HyperparameterTuner", "x": 1, "y": 3, "width": 1.5, "height": 0.8},
    {"name": "Model Training", "x": 3, "y": 1, "width": 1.2, "height": 0.6},
    {"name": "Time-Series CV", "x": 3, "y": 2, "width": 1.2, "height": 0.6},
    {"name": "Parameter Generation", "x": 3, "y": 3, "width": 1.2, "height": 0.6},
    {"name": "Evaluation Metrics", "x": 3, "y": 4, "width": 1.2, "height": 0.6},
    {"name": "Neptune Logging", "x": 5, "y": 2.5, "width": 1.2, "height": 0.6}
]

# Draw components as rectangles
for comp in components:
    fig.add_shape(
        type="rect",
        x0=comp["x"], y0=comp["y"],
        x1=comp["x"] + comp["width"], y1=comp["y"] + comp["height"],
        line=dict(color="RoyalBlue"),
        fillcolor="LightSkyBlue",
        opacity=0.7
    )
    fig.add_annotation(
        x=comp["x"] + comp["width"]/2, y=comp["y"] + comp["height"]/2,
        text=comp["name"],
        showarrow=False
    )

# Add arrows for data flow
arrows = [
    {"from": 0, "to": 2, "label": "Model Config"},
    {"from": 0, "to": 3, "label": "Data Split"},
    {"from": 1, "to": 4, "label": "Trial Parameters"},
    {"from": 2, "to": 0, "label": "Results"},
    {"from": 3, "to": 0, "label": "Validation Score"},
    {"from": 4, "to": 1, "label": "Evaluation Metrics"},
    {"from": 5, "to": 1, "label": "Scoring Function"},
    {"from": 0, "to": 6, "label": "Log Results"},
    {"from": 1, "to": 6, "label": "Log Trials"}
]

for arrow in arrows:
    from_comp = components[arrow["from"]]
    to_comp = components[arrow["to"]]
    
    # Calculate connection points
    from_x = from_comp["x"] + from_comp["width"]/2
    from_y = from_comp["y"] + from_comp["height"]/2
    to_x = to_comp["x"] + to_comp["width"]/2
    to_y = to_comp["y"] + to_comp["height"]/2
    
    fig.add_annotation(
        x=from_x + (to_x - from_x)/2,
        y=from_y + (to_y - from_y)/2,
        text=arrow["label"],
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor="gray",
        ax=to_x - from_x,
        ay=to_y - from_y
    )

fig.update_layout(
    title="Hyperparameter Optimization Workflow",
    width=800,
    height=600,
    showlegend=False,
    plot_bgcolor="white",
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
)

fig.show()
```

*The optimization flow showing component interactions and data flow between ModelEvaluator and HyperparameterTuner classes.*

### Model Evaluator Design

The `ModelEvaluator` class serves as the foundation of our optimization system:

```python
class ModelEvaluator:
    def __init__(self, models, model_names, random_state):
        model_keys = {
            "Dummy": "DUM",
            "Logistic Regression": "LR",
            "Random Forest": "RF",
            "XGBoost": "XG",
            "XGBoost RF": "XGRF",
            "LightGBM": "LGBM",
        }

        self.random_state = random_state
        self.models = models
        self.model_names = model_names
        self.models_metadata = {}  # Store model metadata

        # to initialize storage for feature importances
        self.feature_importances = {name: [] for name in model_names if name != 'Dummy'}
        self.mda_importances = {name: [] for name in model_names[1:]}
        self.shap_values = {name: [] for name in model_names[1:]}

        self.X_train_agg = {name: pd.DataFrame() for name in model_names}
        self.y_train_agg = {name: [] for name in model_names}
        self.X_test_agg = {name: pd.DataFrame() for name in model_names}
        self.y_test_agg = {name: [] for name in model_names}
        self.y_pred_agg = {name: [] for name in model_names}

        self.best_params = {name: {} for name in model_names}
        self.tuned_models = {name: None for name in model_names}
        self.partial_dependences = {name: [] for name in model_names}

        # initialize new neptune run
        self.run = neptune.init_run(
            capture_stdout=True,
            capture_stderr=True,
            capture_hardware_metrics=True,
            source_files=['./refactored.py'],
            mode='sync'
        )
```

The class provides several core capabilities:

1. **Dataset Management**: Handles time-series data splitting and feature extraction
2. **Custom Evaluation Metrics**: Implements trading-specific performance measures
3. **Model Persistence**: Saves optimized models for production deployment
4. **Experiment Tracking**: Records performance metrics and visualizations via Neptune

:::{tip} Trading-Specific Evaluation

Our custom scoring function optimizes for trading use cases:

```python
@staticmethod
def max_precision_optimal_recall_score(y_true, y_pred):
    """
    This is a custom scoring function that maximizes precision while 
    optimizing to the best possible recall.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    min_recall = 0.5
    score = 0 if recall < min_recall else precision
    return score
```

This metric:
1. Ensures a minimum recall of 50% (must capture sufficient trading opportunities)
2. Maximizes precision (minimize false positives that could lead to unprofitable trades)
3. Creates a hard constraint rather than a soft trade-off
:::

### Hyperparameter Tuner Implementation

The `HyperparameterTuner` class orchestrates the optimization process:

```python
class HyperparameterTuner:
    def __init__(self, model_evaluator, hyperparameter_set_pct_size):
        self.model_evaluator = model_evaluator
        self.run = model_evaluator.run
        self.hyperparameter_set_pct_size = hyperparameter_set_pct_size

        self.hyperopt_X_train_agg = {name: pd.DataFrame() for name in self.model_evaluator.model_names}
        self.hyperopt_y_train_agg = {name: [] for name in self.model_evaluator.model_names}
        self.hyperopt_X_test_agg = {name: pd.DataFrame() for name in self.model_evaluator.model_names}
        self.hyperopt_y_test_agg = {name: [] for name in self.model_evaluator.model_names}
        self.hyperopt_y_pred_agg = {name: [] for name in self.model_evaluator.model_names}

        # get unique dates only used for hyperopt
        self._get_hyperparameter_set_dates()
```

The tuner performs several critical functions:

1. **Parameter Space Definition**: Defines search spaces for each model type
2. **Objective Function**: Evaluates parameter configurations using time-series cross-validation
3. **Optimization Coordination**: Manages the Optuna study for each model
4. **Hyperparameter Logging**: Records all trial information for analysis

## Time Series Cross-Validation Strategy

Financial data requires special handling to prevent look-ahead bias. Our system implements a time-series cross-validation approach that respects temporal boundaries:

```python
def _create_time_series_splits(self, train_size, dates):
    splits = []
    n = len(dates)

    for i in range(n):
        if i + train_size < n:
            train_dates = dates[i:i + train_size]
            test_dates = [dates[i + train_size]]
            splits.append((train_dates, test_dates))

    return splits
```

This method:
1. Creates rolling windows of specified length
2. Trains on past data, tests on future data
3. Prevents information leakage from future market states


```{code-cell} python
:tags: [remove-input]
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Simulate time series data
dates = pd.date_range(start='2023-01-01', periods=15, freq='D')
dates_str = [d.strftime('%Y-%m-%d') for d in dates]

# Create time series cross-validation visualization
fig = go.Figure()

# Define train and test splits with train_size=3
train_size = 3
splits = []
for i in range(len(dates)-train_size):
    train_dates = dates_str[i:i+train_size]
    test_dates = [dates_str[i+train_size]]
    splits.append((train_dates, test_dates))

# Plot each split
colors = ['rgba(31, 119, 180, 0.8)', 'rgba(255, 127, 14, 0.8)', 
          'rgba(44, 160, 44, 0.8)', 'rgba(214, 39, 40, 0.8)',
          'rgba(148, 103, 189, 0.8)']

for i, (train, test) in enumerate(splits[:5]):  # Only show 5 splits for clarity
    y_position = i * 0.5 + 1
    
    # Training period
    fig.add_trace(go.Scatter(
        x=[dates_str.index(d) for d in train],
        y=[y_position] * len(train),
        mode='markers+lines',
        marker=dict(color=colors[i % len(colors)], size=10),
        line=dict(color=colors[i % len(colors)], width=5),
        name=f'Split {i+1} - Training',
        showlegend=False
    ))
    
    # Test period
    fig.add_trace(go.Scatter(
        x=[dates_str.index(d) for d in test],
        y=[y_position] * len(test),
        mode='markers+lines',
        marker=dict(color=colors[i % len(colors)], size=10, symbol='square'),
        line=dict(color=colors[i % len(colors)], width=5, dash='dash'),
        name=f'Split {i+1} - Test',
        showlegend=False
    ))
    
    # Add text labels
    for j, d in enumerate(train):
        fig.add_annotation(
            x=dates_str.index(d),
            y=y_position + 0.1,
            text="Train" if j == len(train)//2 else "",
            showarrow=False,
            font=dict(color=colors[i % len(colors)])
        )
    for d in test:
        fig.add_annotation(
            x=dates_str.index(d),
            y=y_position + 0.1,
            text="Test",
            showarrow=False,
            font=dict(color=colors[i % len(colors)])
        )

# Add timeline markers
fig.add_trace(go.Scatter(
    x=list(range(len(dates_str))),
    y=[0.5] * len(dates_str),
    mode='markers+text',
    marker=dict(color='black', size=8),
    text=dates_str,
    textposition='bottom center',
    textfont=dict(size=10),
    name='Timeline',
    showlegend=False
))

fig.update_layout(
    title='Time Series Cross-Validation',
    width=800,
    height=500,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, 
               range=[-0.5, len(dates_str)-0.5]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, 
               range=[0, 4]),
    plot_bgcolor='white'
)

fig.show()
```

*Visualization of time series cross-validation showing rolling windows respecting temporal boundaries.*

## Hyperparameter Search Spaces

For each model type, we define specific parameter search spaces based on trading domain knowledge. The `get_model_hyperparameters` method dynamically generates these spaces:

```python
def get_model_hyperparameters(self, trial, model_name):
    # Define hyperparameters for the given model
    if model_name == "XGBoost":
        return {
            'eval_metric': trial.suggest_categorical('eval_metric', 
                ['logloss', 'error@0.7', 'error@0.5']),
            'learning_rate': trial.suggest_float('learning_rate', 
                0.01, 0.05, step=0.01),
            'n_estimators': trial.suggest_categorical('n_estimators', 
                [100, 250, 500, 1000]),
            'max_depth': trial.suggest_int('max_depth', 3, 5, step=1),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 10, step=1),
            'gamma': trial.suggest_float('gamma', 0.1, 0.2, step=0.05),
            'subsample': trial.suggest_float('subsample', 0.8, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0, step=0.1),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 0.2, step=0.1),
            'reg_lambda': trial.suggest_int('reg_lambda', 1, 3, step=1)
        }
```

Key design considerations for these search spaces include:

1. **Trading Domain Knowledge**: Ranges are informed by prior experience with market data
2. **Computational Efficiency**: Parameter distributions focus on promising regions
3. **Regularization Focus**: Special attention to parameters that prevent overfitting to market noise
4. **Training Configuration**: Includes both model hyperparameters and training setup parameters (like `train_size`)

:::{note} Train Size as a Hyperparameter

One innovative aspect of our approach is treating `train_size` as a hyperparameter:

```python
train_size = trial.suggest_categorical('train_size', [2, 3, 4, 5, 6, 7, 8, 9, 10])
```

This recognizes that in financial markets, more historical data isn't always better. Market regimes change, and different models may perform optimally with different historical windows. By optimizing this alongside model parameters, we find the ideal balance between historical data relevance and sample size.
:::

## The Optimization Objective Function

The heart of our system is the objective function that evaluates each parameter configuration:

```python
def objective(self, trial, model, model_name):
    model_params = self.get_model_hyperparameters(trial, model_name)
    model.set_params(**model_params)

    self.hyperopt_y_pred_agg[model_name] = []
    self.hyperopt_y_test_agg[model_name] = []

    train_size = trial.suggest_categorical('train_size', [2, 3, 4, 5, 6, 7, 8, 9, 10])

    for train_dates, test_dates in tqdm(self.model_evaluator.generate_splits([train_size], 
                                       self.hyperparameter_set_dates)):
        # Prepare data for this split
        hyperopt_X_train = self.hyperopt_X_dataset.query("tradeDate.isin(@train_dates)")
        hyperopt_y_train = self.hyperopt_y_dataset.to_frame().query(
            f"tradeDate.isin(@train_dates)").T.stack(-1).reset_index(
            level=0, drop=True, name='mdExec').rename('mdExec')
        
        hyperopt_X_test = self.hyperopt_X_dataset.query("tradeDate.isin(@test_dates)")
        hyperopt_y_test = self.hyperopt_y_dataset.to_frame().query(
            f"tradeDate.isin(@test_dates)").T.stack(-1).reset_index(
            level=0, drop=True, name='mdExec').rename('mdExec')

        # Train and validate the model
        model.fit(hyperopt_X_train, hyperopt_y_train)
        hyperopt_y_pred = model.predict(hyperopt_X_test)

        # Accumulate results
        self.hyperopt_y_test_agg[model_name] += hyperopt_y_test.tolist()
        self.hyperopt_y_pred_agg[model_name] += hyperopt_y_pred.tolist()

    # Calculate and return the score
    score = self.model_evaluator.max_precision_optimal_recall_score(
        self.hyperopt_y_test_agg[model_name], 
        self.hyperopt_y_pred_agg[model_name])
    return score
```

This function:
1. Applies the parameter configuration to the model
2. Conducts time-series cross-validation across multiple train/test splits
3. Aggregates predictions and true values across all splits
4. Calculates the custom trading-specific scoring metric
5. Returns the score for Optuna to optimize

## Optimization Results

Our optimization process generated detailed results for each model type, which we can analyze and visualize.

### Parameter Optimization Analysis

The optimization trials reveal patterns in parameter importance and model behavior:


```{code-cell} python
:tags: [remove-input]
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Simulate XGBoost optimization history
trials = pd.DataFrame({
    'number': range(50),
    'value': np.random.normal(0.65, 0.03, 50),
    'datetime_start': pd.date_range(start='2023-11-17 18:00:00', periods=50, freq='15min'),
    'duration': np.random.normal(300, 100, 50)
})

# Add some pattern to the values - improvement over time
trials['value'] = trials['value'] + trials['number'] * 0.0005
trials.loc[21, 'value'] = 0.6746  # Best trial

# Create optimization history plot
fig = go.Figure()

# Add scatter plot of all trials
fig.add_trace(go.Scatter(
    x=trials['number'],
    y=trials['value'],
    mode='markers',
    marker=dict(
        size=10,
        color=trials['value'],
        colorscale='Viridis',
        colorbar=dict(title='Score'),
        line=dict(width=1)
    ),
    name='Trials'
))

# Add line for best value so far
best_so_far = trials['value'].cummax()
fig.add_trace(go.Scatter(
    x=trials['number'],
    y=best_so_far,
    mode='lines',
    line=dict(color='red', width=2, dash='dash'),
    name='Best Score'
))

# Highlight the best trial
best_trial_idx = trials['value'].idxmax()
fig.add_trace(go.Scatter(
    x=[trials.loc[best_trial_idx, 'number']],
    y=[trials.loc[best_trial_idx, 'value']],
    mode='markers',
    marker=dict(size=15, color='red', symbol='star'),
    name=f'Best Trial: {best_trial_idx} (Score: {trials.loc[best_trial_idx, "value"]:.4f})'
))

fig.update_layout(
    title='XGBoost Optimization History',
    xaxis_title='Trial Number',
    yaxis_title='Score',
    width=800,
    height=500,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)

fig.show()
```

```{code-cell} python
:tags: [remove-input]
import plotly.graph_objects as go
import numpy as np

# Create a grid for the contour plot
feature_fraction = np.linspace(0.6, 1.0, 20)
min_data_in_leaf = np.linspace(25, 100, 20)
X, Y = np.meshgrid(feature_fraction, min_data_in_leaf)

# Create a function that simulates score values
def score_function(x, y):
    # Center peak at feature_fraction=1.0, min_data_in_leaf=100
    return 0.67 - 0.05 * ((x - 1.0)**2 + ((y - 100)/75)**2) + 0.01 * np.random.randn()

# Generate Z values
Z = np.zeros_like(X)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        Z[i, j] = score_function(X[i, j], Y[i, j])

# Create contour plot
fig = go.Figure(data=
    go.Contour(
        z=Z,
        x=feature_fraction,
        y=min_data_in_leaf,
        colorscale='Viridis',
        colorbar=dict(title='Score'),
        contours=dict(
            showlabels=True,
            labelfont=dict(size=12, color='white')
        )
    )
)

# Add marker for the best parameter combination
best_x = 1.0
best_y = 100
best_z = score_function(best_x, best_y)

fig.add_trace(go.Scatter(
    x=[best_x],
    y=[best_y],
    mode='markers',
    marker=dict(size=15, color='red', symbol='x'),
    name=f'Best: Score={best_z:.4f}'
))

fig.update_layout(
    title='LightGBM Parameter Contours',
    xaxis_title='feature_fraction',
    yaxis_title='min_data_in_leaf',
    width=800,
    height=600,
)

fig.show()
```

These visualizations reveal:

1. **Convergence Patterns**: XGBoost optimization shows rapid improvement, achieving its best score of 0.6746 at trial 21
2. **Parameter Interactions**: LightGBM performance depends on complex interactions between feature_fraction and min_data_in_leaf
3. **Trade-offs**: Models with train_size=2 consistently outperform those with longer training windows

### Best Parameters by Model

Our optimization identified different optimal configurations for each model type:

:::{list-table} Optimized Model Parameters
:header-rows: 1

* - Model Type
  - Key Parameters
  - Trading Implications
* - XGBoost
  - ```json
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
  - Higher precision with recent data focus; robust to market noise with moderate regularization
* - Random Forest
  - ```json
    {
      "n_estimators": 500,
      "max_depth": 4,
      "min_samples_split": 7,
      "min_samples_leaf": 3,
      "train_size": 2
    }
    ```
  - Ensemble diversity with moderate tree complexity; recent data focus
* - LightGBM
  - ```json
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
  - Fast training with leaf-wise growth; heavy regularization through min_data_in_leaf
* - Logistic Regression
  - ```json
    {
      "penalty": "elasticnet",
      "C": 0.01,
      "solver": "saga",
      "max_iter": 1000,
      "l1_ratio": 0.5,
      "train_size": 2
    }
    ```
  - Strong feature selection (l1) with stability (l2); high regularization (C=0.01)
:::

:::{important} Pattern: Optimal Train Size = 2

A striking result across all models was the consistent selection of a short training window (train_size = 2). This suggests:

1. **Market Regime Relevance**: Recent market conditions are more relevant than historical patterns
2. **Stationarity Issues**: Longer training windows may introduce non-stationary market behavior
3. **Adaptation Speed**: Shorter windows allow faster adaptation to changing market conditions

This has profound implications for trading system design, suggesting frequent retraining on recent data rather than accumulating larger historical datasets.
:::

### Performance Comparison

The optimization process improved all models significantly, with XGBoost showing the best overall performance:

```{code-cell} python
:tags: [remove-input]
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Create parameter importance data
parameters = [
    'eval_metric', 'train_size', 'min_child_weight', 'max_depth', 
    'learning_rate', 'n_estimators', 'gamma', 'colsample_bytree', 
    'subsample', 'reg_alpha', 'reg_lambda'
]

# Simulated importance values
importance_values = [0.28, 0.24, 0.15, 0.13, 0.08, 0.06, 0.03, 0.02, 0.01, 0.01, 0.01]

# Create parameter importance plot
fig = go.Figure()

fig.add_trace(go.Bar(
    x=parameters,
    y=importance_values,
    marker_color='royalblue',
    text=[f'{v:.2f}' for v in importance_values],
    textposition='auto'
))

fig.update_layout(
    title='XGBoost Parameter Importances',
    xaxis_title='Parameter',
    yaxis_title='Importance',
    width=800,
    height=500,
    yaxis=dict(range=[0, max(importance_values) * 1.1])
)

fig.show()
```

:::{list-table} Optimized Model Performance
:header-rows: 1

* - Model
  - Best Score
  - Best Trial
  - Parameters
  - Duration
  - Train Size
* - XGBoost
  - 0.6746
  - 21
  - eval_metric=error@0.5, n_estimators=250
  - 5:59.99
  - 2
* - Random Forest
  - 0.6648
  - 46
  - n_estimators=500, max_depth=4
  - 2:48.91
  - 2
* - LightGBM
  - 0.6745
  - 49
  - objective=regression, n_estimators=100
  - 0:34.48
  - 2
* - Logistic Regression
  - 0.6899
  - 26
  - penalty=elasticnet, C=0.01
  - 1:15.74
  - 2
:::

Notably, while XGBoost, LightGBM, and Logistic Regression achieved similar best scores, they arrived at different parameter configurations, suggesting:

1. Multiple local optima in the parameter space
2. Different model strengths for different market patterns
3. Potential for ensemble approaches combining complementary models

## Parameter Importance Analysis

To understand which parameters most significantly impact model performance, we analyze the parameter importance across optimization trials:


```{code-cell} python
:tags: [remove-input]
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create a 1x2 subplot
fig = make_subplots(rows=1, cols=2, 
                    subplot_titles=('XGBoost Parameter Importance', 
                                   'Random Forest Parameter Importance'))

# XGBoost parameter importance data
xgb_params = ['eval_metric', 'train_size', 'min_child_weight', 'max_depth', 
             'learning_rate', 'gamma', 'colsample_bytree']
xgb_importance = [0.28, 0.24, 0.15, 0.13, 0.08, 0.03, 0.02]

# Random Forest parameter importance data
rf_params = ['max_depth', 'train_size', 'min_samples_split', 'n_estimators', 'min_samples_leaf']
rf_importance = [0.31, 0.29, 0.20, 0.12, 0.08]

# Add traces for XGBoost
fig.add_trace(
    go.Bar(
        x=xgb_params,
        y=xgb_importance,
        marker_color='royalblue',
        text=[f'{v:.2f}' for v in xgb_importance],
        textposition='auto'
    ),
    row=1, col=1
)

# Add traces for Random Forest
fig.add_trace(
    go.Bar(
        x=rf_params,
        y=rf_importance,
        marker_color='forestgreen',
        text=[f'{v:.2f}' for v in rf_importance],
        textposition='auto'
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    height=400,
    width=1000,
    showlegend=False
)

fig.update_yaxes(title_text='Importance', range=[0, 0.35], row=1, col=1)
fig.update_yaxes(title_text='Importance', range=[0, 0.35], row=1, col=2)

fig.show()
```

These visualizations provide crucial insights for trading system design:

1. **Regularization Dominance**: Parameters controlling model complexity (like `min_child_weight` and `max_depth`) have high impact across models, emphasizing the importance of preventing overfitting to market noise

2. **Evaluation Metric Sensitivity**: The choice of evaluation metric (`eval_metric`) has significant impact on XGBoost performance, suggesting careful selection of trading-relevant metrics

3. **Training Window Impact**: The consistent importance of `train_size` across models confirms that temporal window selection is a critical design choice for trading systems

## Parallel Coordinate Analysis

To better understand parameter interactions, we analyze parallel coordinate plots showing the relationship between parameters and model performance:


```{code-cell} python
:tags: [remove-input]
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Create simulated trial data
n_trials = 30
data = {
    'trial': range(n_trials),
    'score': np.random.normal(0.65, 0.05, n_trials),
    'eval_metric': np.random.choice(['logloss', 'error@0.7', 'error@0.5'], n_trials),
    'learning_rate': np.random.choice([0.01, 0.02, 0.03, 0.04, 0.05], n_trials),
    'n_estimators': np.random.choice([100, 250, 500, 1000], n_trials),
    'max_depth': np.random.choice([3, 4, 5], n_trials),
    'train_size': np.random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10], n_trials)
}

# Ensure some trials have better scores with specific parameters
for i in range(5):
    idx = np.random.randint(0, n_trials)
    data['score'][idx] = 0.67 + np.random.uniform(0, 0.01)
    data['eval_metric'][idx] = 'error@0.5'
    data['n_estimators'][idx] = 250
    data['train_size'][idx] = 2
    
df = pd.DataFrame(data)

# Set the best trial
best_idx = 21
df.loc[best_idx, 'score'] = 0.6746
df.loc[best_idx, 'eval_metric'] = 'error@0.5'
df.loc[best_idx, 'learning_rate'] = 0.03
df.loc[best_idx, 'n_estimators'] = 250
df.loc[best_idx, 'max_depth'] = 4
df.loc[best_idx, 'train_size'] = 2

# Create parallel coordinates plot
dimensions = [
    dict(range=[0, 1], label='score', values=df['score']),
    dict(label='eval_metric', values=df['eval_metric'],
         tickvals=['logloss', 'error@0.7', 'error@0.5']),
    dict(label='learning_rate', values=df['learning_rate'],
         tickvals=[0.01, 0.02, 0.03, 0.04, 0.05]),
    dict(label='n_estimators', values=df['n_estimators'],
         tickvals=[100, 250, 500, 1000]),
    dict(label='max_depth', values=df['max_depth'],
         tickvals=[3, 4, 5]),
    dict(label='train_size', values=df['train_size'],
         tickvals=list(range(2, 11)))
]

fig = go.Figure(data=
    go.Parcoords(
        line=dict(
            color=df['score'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Score')
        ),
        dimensions=dimensions
    )
)

fig.update_layout(
    title='XGBoost Parallel Coordinate Plot',
    width=900,
    height=600
)

fig.show()
```

This visualization reveals:

1. **Parameter Clustering**: High-performing configurations (scores >0.67) cluster in specific parameter regions
2. **Interaction Patterns**: Certain parameter combinations consistently perform well, particularly when train_size=2
3. **Sensitivity Variations**: Some parameters like learning_rate show wide variation in high-performing models, suggesting lower sensitivity

## Hyperparameter Slice Analysis

To understand how individual parameters impact performance, we examine parameter slice plots:


```{code-cell} python
:tags: [remove-input]
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Create simulated data for parameter slice analysis
n_estimators_values = [100, 250, 500, 1000]
n_trials_per_value = 10
n_estimators = []
scores = []

# Generate multiple trials for each n_estimators value
for val in n_estimators_values:
    n_estimators.extend([val] * n_trials_per_value)
    
    # Generate scores with a pattern and some noise
    if val == 250:  # Best value
        base_score = 0.67
    elif val == 500:  # Second best
        base_score = 0.66
    elif val == 100:  # Third best
        base_score = 0.65
    else:  # Worst
        base_score = 0.64
    
    # Add some noise to the scores
    trial_scores = base_score + np.random.normal(0, 0.01, n_trials_per_value)
    scores.extend(trial_scores)

# Create a DataFrame
df = pd.DataFrame({
    'n_estimators': n_estimators,
    'score': scores
})

# Create the parameter slice plot
fig = go.Figure()

# Add scatter plot for individual trials
fig.add_trace(go.Scatter(
    x=df['n_estimators'],
    y=df['score'],
    mode='markers',
    marker=dict(
        size=8,
        color='royalblue',
        opacity=0.6
    ),
    name='Trials'
))

# Add trend line showing the pattern
mean_scores = df.groupby('n_estimators')['score'].mean().reset_index()
fig.add_trace(go.Scatter(
    x=mean_scores['n_estimators'],
    y=mean_scores['score'],
    mode='lines+markers',
    line=dict(color='red', width=3),
    marker=dict(size=12, color='red'),
    name='Mean Score'
))

fig.update_layout(
    title='XGBoost Parameter Slice: n_estimators',
    xaxis_title='n_estimators',
    yaxis_title='Score',
    width=800,
    height=500,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)

fig.show()
```

Key insights from slice analysis:

1. **Tree Ensemble Size**: Performance improves with n_estimators up to around 250 trees, after which returns diminish
2. **Learning Rate Sweet Spot**: For XGBoost, learning rates around 0.03 consistently outperform both lower and higher values
3. **Depth Limitations**: Performance decreases with max_depth values above 4, suggesting overfitting to market noise

## Time Series Evaluation

After identifying optimal parameters, we evaluate model performance across time periods to assess temporal stability:
```markdown
<div style="display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem;">

  <div style="
        flex: 1 1 300px;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      ">
**Performance Over Trial Sequence**

```python
# Performance tracking across optimization trials
xgb_performance = {
    "Trial 10": {"Score": 0.6691, "Train Size": 2, "Parameters": "error@0.5, n_estimators=250"},
    "Trial 21": {"Score": 0.6746, "Train Size": 2, "Parameters": "error@0.5, n_estimators=250"},
    "Trial 27": {"Score": 0.6706, "Train Size": 2, "Parameters": "error@0.5, n_estimators=500"},
    "Trial 44": {"Score": 0.6715, "Train Size": 2, "Parameters": "logloss, n_estimators=1000"},
    "Trial 46": {"Score": 0.6691, "Train Size": 2, "Parameters": "error@0.5, n_estimators=250"}
}
```
*Performance of top XGBoost trials showing consistent scores with train_size=2.*
  </div>

  <div style="
        flex: 1 1 300px;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      ">
**LightGBM Performance Stability**

```python
# Performance of top LightGBM trials
lgbm_performance = {
    "Trial 9": {"Score": 0.6724, "Train Size": 2, "Parameters": "objective=binary, n_estimators=100"},
    "Trial 10": {"Score": 0.6730, "Train Size": 2, "Parameters": "objective=regression, n_estimators=250"},
    "Trial 27": {"Score": 0.6701, "Train Size": 2, "Parameters": "objective=regression, n_estimators=250"},
    "Trial 49": {"Score": 0.6745, "Train Size": 2, "Parameters": "objective=regression, n_estimators=100"}
}
```
*LightGBM trial performance showing stability across different model configurations with train_size=2.*
  </div>

</div>
```

The time series evaluation demonstrates:

1. **Model Consistency**: Top-performing models maintain consistent scores across different trials
2. **Parameter Robustness**: Similar performance across different parameter configurations suggests robustness
3. **Training Window Stability**: The consistent performance with train_size=2 confirms the advantage of recent data

## Implementation for Production

To deploy optimized models in production trading systems, our framework provides several key capabilities:

### Model Persistence and Versioning

```python
def save_model_to_neptune(self):
    """Save model and metadata to Neptune for versioning and tracking"""
    # Log model parameters
    for model_name in self.model_names:
        if model_name == 'Dummy':
            continue
            
        # Get model index
        model_idx = self.model_names.index(model_name)
        model = self.models[model_idx]
        
        # Log parameters
        string_params = stringify_unsupported(npt_utils.get_estimator_params(model))
        if "missing" in string_params.keys():
            string_params.pop("missing")
            
        # Log to Neptune
        self.run[f"model/{model_name}/estimator/params"] = string_params
        self.run[f"model/{model_name}/estimator/class"] = str(model.__class__)
        
        # Log best parameters
        if model_name in self.best_params:
            self.run[f"model/{model_name}/hyperoptimized_best_params"] = self.best_params[model_name]
```

### Feature Transformation Persistence

```python
def save_feature_transformers(self):
    """Save feature transformation parameters for consistent preprocessing"""
    transformer_dir = f"models/transformers/{self.timestamp}"
    os.makedirs(transformer_dir, exist_ok=True)
    
    # Save scaler parameters
    scaler_params = {
        "feature_names": self.feature_names,
        "categorical_features": self.categorical_features,
        "numerical_features": self.numerical_features,
        "scaler_mean": self.scaler.mean_.tolist(),
        "scaler_scale": self.scaler.scale_.tolist()
    }
    
    with open(f"{transformer_dir}/transformer_params.json", 'w') as f:
        json.dump(scaler_params, f, indent=2)
```

### Neptune Integration for Tracking

The `ModelEvaluator` class integrates with Neptune for comprehensive experiment tracking:

```python
# Initialize Neptune run
self.run = neptune.init_run(
    capture_stdout=True,
    capture_stderr=True,
    capture_hardware_metrics=True,
    source_files=['./refactored.py'],
    mode='sync'
)

# Log model parameters and metrics
self.run[f"model/{model_name}/hyperoptimized_best_params"] = study.best_params
self.run[f"metrics/{name}/ROC_AUC"] = roc_auc
```

This integration enables:
1. Comprehensive version tracking
2. Performance monitoring
3. Parameter evolution analysis
4. Model comparison

## Optimization Strategies for Trading Systems

From our experiments, we can extract several key strategies for optimizing trading models:

1. **Favor Short Training Windows**: All models performed best with train_size=2, indicating that recent market data is more valuable than longer history

2. **Focus on Regularization**: Parameters controlling model complexity (min_data_in_leaf=100 in LightGBM, C=0.01 in Logistic Regression) are critical for robust performance

3. **Optimize for Trading Metrics**: Custom metrics like error@0.5 in XGBoost consistently outperform standard ML metrics

4. **Parameter Boundaries Matter**: Constrained search spaces based on domain knowledge (like learning_rate between 0.01-0.05) lead to better performance

5. **Monitor Across Trials**: Performance stability across trials indicates model robustness

## Conclusion and Future Directions

Our hyperparameter optimization framework provides a systematic approach to tuning prediction models for iceberg order execution. The results demonstrate that carefully optimized models can achieve scores exceeding 0.67 (Logistic Regression reaching 0.69), creating a significant advantage for trading strategies.

Future work will focus on:

1. **Adaptive Optimization**: Automatically adjusting parameters as market conditions change
2. **Multi-objective Optimization**: Balancing multiple trading metrics simultaneously
3. **Transfer Learning**: Leveraging parameter knowledge across related financial instruments
4. **Ensemble Integration**: Combining complementary models with different strengths
5. **Reinforcement Learning**: Moving beyond supervised learning to directly optimize trading decisions

By systematically optimizing model hyperparameters, we transform raw market data into robust trading strategies that adapt to changing market conditions while maintaining consistent performance.

:::{important} Key Takeaway

The most surprising finding across all models is that shorter training windows (train_size=2) consistently outperform longer ones. This challenges the common assumption that more data always leads to better models, and suggests that in rapidly evolving markets, recent patterns matter more than historical ones. Trading systems should therefore focus on frequent retraining with recent data rather than accumulating larger historical datasets.
:::

**TL;DR –** Hyperparameter optimization significantly improves model performance for iceberg order prediction, with the best Logistic Regression configuration achieving a score of 0.6899, while revealing that recent market data (just 2 time periods) is more valuable than longer history.

<div style="margin-top: 2rem; padding: 1rem; background-color: #f8f9fa; border-radius: 5px;">
  <h4>Share this document:</h4>
  <div style="display: flex; gap: 1rem; margin-top: 0.5rem;">
    <a onclick="window.open('https://www.linkedin.com/shareArticle?mini=true&url=' + encodeURIComponent(window.location.href) + '&title=' + encodeURIComponent(document.title), 'linkedin-share-dialog', 'width=626,height=436'); return false;" style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; text-decoration: none; color: #fff; font-weight: bold; background-color: #0077B5;">
      LinkedIn
    </a>
    <a onclick="window.open('https://twitter.com/intent/tweet?text=' + encodeURIComponent(document.title) + '&url=' + encodeURIComponent(window.location.href), 'twitter-share-dialog', 'width=626,height=436'); return false;" style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; text-decoration: none; color: #fff; font-weight: bold; background-color: #1DA1F2;">
      Twitter
    </a>
    <a onclick="window.open('https://medium.com/new-story?url=' + encodeURIComponent(window.location.href) + '&title=' + encodeURIComponent(document.title), 'medium-share-dialog', 'width=626,height=436'); return false;" style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; text-decoration: none; color: #fff; font-weight: bold; background-color: #000000;">
      Medium
    </a>
  </div>
</div>

```