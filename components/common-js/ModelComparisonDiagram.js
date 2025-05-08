const React = require('react');

const ModelComparisonDiagram = () => {
  const styles = {
    container: {
      fontFamily: 'Arial, sans-serif',
      padding: '20px',
      maxWidth: '100%'
    },
    table: {
      width: '100%',
      borderCollapse: 'collapse',
      marginTop: '20px',
      fontSize: '14px'
    },
    th: {
      backgroundColor: '#4a90e2',
      color: 'white',
      padding: '10px',
      textAlign: 'left',
      border: '1px solid #ddd'
    },
    td: {
      padding: '10px',
      border: '1px solid #ddd',
      verticalAlign: 'top'
    },
    highlight: {
      backgroundColor: '#e8f4f8'
    },
    header: {
      marginBottom: '20px'
    },
    metricsBox: {
      backgroundColor: '#f9f9f9',
      border: '1px solid #ddd',
      borderRadius: '5px',
      padding: '15px',
      marginTop: '20px'
    },
    metricsTitle: {
      fontWeight: 'bold',
      marginBottom: '10px'
    },
    metricItem: {
      display: 'flex',
      justifyContent: 'space-between',
      margin: '5px 0'
    }
  };

  return (
    React.createElement('div', { style: styles.container },
      React.createElement('div', { style: styles.header },
        React.createElement('h2', null, 'Model Comparison for Iceberg Order Prediction'),
        React.createElement('p', null, 'Comparison of machine learning models evaluated on the iceberg order simulation dataset')
      ),
      
      React.createElement('table', { style: styles.table },
        React.createElement('thead', null,
          React.createElement('tr', null,
            React.createElement('th', { style: styles.th }, 'Model'),
            React.createElement('th', { style: styles.th }, 'Description'),
            React.createElement('th', { style: styles.th }, 'Parameters'),
            React.createElement('th', { style: styles.th }, 'Strengths'),
            React.createElement('th', { style: styles.th }, 'Limitations')
          )
        ),
        React.createElement('tbody', null,
          React.createElement('tr', null,
            React.createElement('td', { style: styles.td }, React.createElement('strong', null, 'RandomForest')),
            React.createElement('td', { style: styles.td }, 'Ensemble of decision trees using bootstrap samples'),
            React.createElement('td', { style: styles.td }, 
              'n_estimators=500', React.createElement('br'),
              'criterion=\'log_loss\'', React.createElement('br'),
              'max_depth=4', React.createElement('br'),
              'min_samples_split=7', React.createElement('br'),
              'min_samples_leaf=3', React.createElement('br'),
              'train_size=2'
            ),
            React.createElement('td', { style: styles.td }, 
              '- Handles non-linearity well', React.createElement('br'),
              '- Robust to outliers', React.createElement('br'),
              '- Native feature importance'
            ),
            React.createElement('td', { style: styles.td }, 
              '- May overfit on noisy data', React.createElement('br'),
              '- Less interpretable than single trees', React.createElement('br'),
              '- Memory intensive for large datasets'
            )
          ),
          React.createElement('tr', { style: styles.highlight },
            React.createElement('td', { style: styles.td }, React.createElement('strong', null, 'XGBoost')),
            React.createElement('td', { style: styles.td }, 'Gradient boosted trees with regularization'),
            React.createElement('td', { style: styles.td }, 
              'eval_metric=\'error@0.5\'', React.createElement('br'),
              'learning_rate=0.03', React.createElement('br'),
              'n_estimators=250', React.createElement('br'),
              'max_depth=4', React.createElement('br'),
              'gamma=0.2', React.createElement('br'),
              'subsample=1.0', React.createElement('br'),
              'colsample_bytree=0.8', React.createElement('br'),
              'reg_alpha=0.2', React.createElement('br'),
              'reg_lambda=2', React.createElement('br'),
              'train_size=2'
            ),
            React.createElement('td', { style: styles.td }, 
              '- High prediction accuracy', React.createElement('br'),
              '- Handles imbalanced data well', React.createElement('br'),
              '- Efficient implementation'
            ),
            React.createElement('td', { style: styles.td }, 
              '- More prone to overfitting than RF', React.createElement('br'),
              '- Requires more hyperparameter tuning', React.createElement('br'),
              '- Less interpretable'
            )
          ),
          React.createElement('tr', null,
            React.createElement('td', { style: styles.td }, React.createElement('strong', null, 'LightGBM')),
            React.createElement('td', { style: styles.td }, 'Gradient boosting framework that uses tree-based algorithms'),
            React.createElement('td', { style: styles.td }, 
              'objective=\'regression\'', React.createElement('br'),
              'learning_rate=0.05', React.createElement('br'),
              'n_estimators=100', React.createElement('br'),
              'max_depth=4', React.createElement('br'),
              'num_leaves=31', React.createElement('br'),
              'min_sum_hessian_in_leaf=10', React.createElement('br'),
              'extra_trees=True', React.createElement('br'),
              'min_data_in_leaf=100', React.createElement('br'),
              'feature_fraction=1.0', React.createElement('br'),
              'bagging_fraction=0.8', React.createElement('br'),
              'lambda_l1=2', React.createElement('br'),
              'lambda_l2=0', React.createElement('br'),
              'min_gain_to_split=0.1', React.createElement('br'),
              'train_size=2'
            ),
            React.createElement('td', { style: styles.td }, 
              '- Faster training speed', React.createElement('br'),
              '- Lower memory usage', React.createElement('br'),
              '- Better accuracy with categorical features'
            ),
            React.createElement('td', { style: styles.td }, 
              '- Can overfit on small datasets', React.createElement('br'),
              '- Less common in production environments', React.createElement('br'),
              '- Newer with less community support'
            )
          ),
          React.createElement('tr', null,
            React.createElement('td', { style: styles.td }, React.createElement('strong', null, 'LogisticRegression')),
            React.createElement('td', { style: styles.td }, 'Linear model for binary classification'),
            React.createElement('td', { style: styles.td }, 
              'penalty=\'elasticnet\'', React.createElement('br'),
              'C=0.01', React.createElement('br'),
              'solver=\'saga\'', React.createElement('br'),
              'max_iter=1000', React.createElement('br'),
              'l1_ratio=0.5', React.createElement('br'),
              'train_size=2'
            ),
            React.createElement('td', { style: styles.td }, 
              '- Simple and interpretable', React.createElement('br'),
              '- Fast training and inference', React.createElement('br'),
              '- Less prone to overfitting'
            ),
            React.createElement('td', { style: styles.td }, 
              '- Cannot capture non-linear relationships', React.createElement('br'),
              '- Lower performance ceiling', React.createElement('br'),
              '- Feature engineering more important'
            )
          )
        )
      ),

      React.createElement('div', { style: styles.metricsBox },
        React.createElement('div', { style: styles.metricsTitle }, 'Evaluation Metrics Used:'),
        React.createElement('div', { style: styles.metricItem },
          React.createElement('span', null, React.createElement('strong', null, 'Accuracy:')),
          React.createElement('span', null, 'Overall correctness of predictions')
        ),
        React.createElement('div', { style: styles.metricItem },
          React.createElement('span', null, React.createElement('strong', null, 'Precision:')),
          React.createElement('span', null, 'Proportion of positive identifications that were correct')
        ),
        React.createElement('div', { style: styles.metricItem },
          React.createElement('span', null, React.createElement('strong', null, 'Recall:')),
          React.createElement('span', null, 'Proportion of actual positives that were identified correctly')
        ),
        React.createElement('div', { style: styles.metricItem },
          React.createElement('span', null, React.createElement('strong', null, 'F1 Score:')),
          React.createElement('span', null, 'Harmonic mean of precision and recall')
        ),
        React.createElement('div', { style: styles.metricItem },
          React.createElement('span', null, React.createElement('strong', null, 'F-beta Score:')),
          React.createElement('span', null, 'Weighted F-score with emphasis on precision (β=0.5) or recall (β=2.0)')
        ),
        React.createElement('div', { style: styles.metricItem },
          React.createElement('span', null, React.createElement('strong', null, 'Custom Metric:')),
          React.createElement('span', null, 'max_precision_optimal_recall_score - Maximizes precision with minimum recall threshold')
        )
      )
    )
  );
};

module.exports = ModelComparisonDiagram;
