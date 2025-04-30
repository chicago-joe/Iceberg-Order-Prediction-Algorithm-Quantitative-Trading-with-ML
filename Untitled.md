---
jupytext:
  formats: ipynb,md:myst,py:percent
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.0
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

```{code-cell} ipython3
:tags: [hide-input]

import plotly.io as pio
pio.renderers.default = 'png'           # options: 'notebook', 'notebook_connected', 'svg', 'png', 'iframe', etc.
# pio.renderers.default = 'notebook'      # interactive HTML widget

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

fig.show()            # now produces a PNG in the output cell
```
