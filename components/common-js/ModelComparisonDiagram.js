const React = require('react');

const ModelComparisonDiagram = () => {
  const styles = {
    container: {
      fontFamily: 'Arial, sans-serif',
      padding: '20px',
      maxWidth: '100%',
      width: '100%',
      overflowX: 'auto'
    },
    table: {
      width: '100%',
      borderCollapse: 'collapse',
      marginTop: '20px',
      fontSize: '16px'
    },
    th: {
      backgroundColor: '#4a90e2',
      color: 'white',
      padding: '12px',
      textAlign: 'left',
      border: '1px solid #ddd',
      fontSize: '17px',
      fontWeight: 'bold'
    },
    td: {
      padding: '12px',
      border: '1px solid #ddd',
      verticalAlign: 'top',
      fontSize: '16px',
      lineHeight: '1.5'
    },
    highlight: {
      backgroundColor: '#e8f4f8'
    },
    header: {
      marginBottom: '20px'
    },
    h2: {
      fontSize: '24px',
      marginBottom: '15px',
      color: '#333'
    },
    p: {
      fontSize: '17px',
      lineHeight: '1.5'
    },
    metricsBox: {
      backgroundColor: '#f9f9f9',
      border: '1px solid #ddd',
      borderRadius: '5px',
      padding: '20px',
      marginTop: '25px',
      width: '100%'
    },
    metricsTitle: {
      fontWeight: 'bold',
      marginBottom: '15px',
      fontSize: '18px'
    },
    metricItem: {
      display: 'flex',
      justifyContent: 'space-between',
      margin: '8px 0',
      fontSize: '16px',
      lineHeight: '1.4'
    },
    barChart: {
      marginTop: '30px',
      width: '100%'
    },
    barContainer: {
      display: 'flex',
      flexDirection: 'column',
      gap: '10px',
      marginBottom: '25px'
    },
    modelBar: {
      display: 'flex',
      alignItems: 'center',
      height: '40px'
    },
    modelLabel: {
      width: '150px',
      fontWeight: 'bold',
      fontSize: '16px'
    },
    bar: {
      height: '30px',
      borderRadius: '3px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: 'white',
      fontWeight: 'bold'
    }
  };

  // Model performance data
  const models = [
    { name: 'LogisticRegression', score: 69, color: '#9c59b6' },
    { name: 'XGBoost', score: 67, color: '#3498db' },
    { name: 'LightGBM', score: 67, color: '#e67e22' },
    { name: 'RandomForest', score: 66, color: '#2ecc71' }
  ];

  return (
    React.createElement('div', { style: styles.container },
      React.createElement('div', { style: styles.header },
        React.createElement('h2', { style: styles.h2 }, 'Model Comparison'),
        React.createElement('p', { style: styles.p }, 'Comparison of machine learning models evaluated on the iceberg order dataset')
      ),
      
      React.createElement('div', { style: styles.barChart },
        React.createElement('div', { style: styles.barContainer },
          models.map((model, index) => 
            React.createElement('div', { key: index, style: styles.modelBar },
              React.createElement('div', { style: styles.modelLabel }, model.name),
              React.createElement('div', 
                { 
                  style: {
                    ...styles.bar,
                    width: `${model.score * 5}px`,
                    backgroundColor: model.color
                  }
                }, 
                `${model.score}%`
              )
            )
          )
        )
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
            React.createElement('td', { style: styles.td }, React.createElement('strong', null, 'LogisticRegression')),
            React.createElement('td', { style: styles.td }, 'Linear model for binary classification'),
            React.createElement('td', { style: styles.td }, 
              React.createElement('ul', { style: {margin: 0, paddingLeft: '20px'} },
                React.createElement('li', null, 'penalty=\'elasticnet\''),
                React.createElement('li', null, 'C=0.01'),
                React.createElement('li', null, 'solver=\'saga\''),
                React.createElement('li', null, 'max_iter=1000'),
                React.createElement('li', null, 'l1_ratio=0.5'),
                React.createElement('li', null, 'train_size=2')
              )
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
          ),
          React.createElement('tr', { style: styles.highlight },
            React.createElement('td', { style: styles.td }, React.createElement('strong', null, 'XGBoost')),
            React.createElement('td', { style: styles.td }, 'Gradient boosted trees with regularization'),
            React.createElement('td', { style: styles.td }, 
              React.createElement('ul', { style: {margin: 0, paddingLeft: '20px'} },
                React.createElement('li', null, 'eval_metric=\'error@0.5\''),
                React.createElement('li', null, 'learning_rate=0.03'),
                React.createElement('li', null, 'n_estimators=250'),
                React.createElement('li', null, 'max_depth=4'),
                React.createElement('li', null, 'gamma=0.2'),
                React.createElement('li', null, 'subsample=1.0'),
                React.createElement('li', null, 'colsample_bytree=0.8'),
                React.createElement('li', null, 'reg_alpha=0.2'),
                React.createElement('li', null, 'reg_lambda=2'),
                React.createElement('li', null, 'train_size=2')
              )
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
              React.createElement('ul', { style: {margin: 0, paddingLeft: '20px'} },
                React.createElement('li', null, 'objective=\'regression\''),
                React.createElement('li', null, 'learning_rate=0.05'),
                React.createElement('li', null, 'n_estimators=100'),
                React.createElement('li', null, 'max_depth=4'),
                React.createElement('li', null, 'num_leaves=31'),
                React.createElement('li', null, 'min_sum_hessian_in_leaf=10'),
                React.createElement('li', null, 'extra_trees=True'),
                React.createElement('li', null, 'min_data_in_leaf=100'),
                React.createElement('li', null, 'feature_fraction=1.0'),
                React.createElement('li', null, 'bagging_fraction=0.8'),
                React.createElement('li', null, 'lambda_l1=2'),
                React.createElement('li', null, 'lambda_l2=0'),
                React.createElement('li', null, 'min_gain_to_split=0.1'),
                React.createElement('li', null, 'train_size=2')
              )
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
            React.createElement('td', { style: styles.td }, React.createElement('strong', null, 'RandomForest')),
            React.createElement('td', { style: styles.td }, 'Ensemble of decision trees using bootstrap samples'),
            React.createElement('td', { style: styles.td }, 
              React.createElement('ul', { style: {margin: 0, paddingLeft: '20px'} },
                React.createElement('li', null, 'n_estimators=500'),
                React.createElement('li', null, 'criterion=\'log_loss\''),
                React.createElement('li', null, 'max_depth=4'),
                React.createElement('li', null, 'min_samples_split=7'),
                React.createElement('li', null, 'min_samples_leaf=3'),
                React.createElement('li', null, 'train_size=2')
              )
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