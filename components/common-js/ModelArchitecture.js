const React = require('react');

const ModelArchitecture = () => {
  // Styles for visual elements
  const styles = {
    container: {
      fontFamily: 'Arial, sans-serif',
      padding: '25px',
      maxWidth: '100%',
      width: '100%',
      backgroundColor: '#f9f9f9',
      borderRadius: '8px',
      border: '1px solid #e0e0e0',
      overflowX: 'auto'
    },
    heading: {
      color: '#333',
      marginBottom: '25px',
      fontSize: '24px',
      fontWeight: 'bold'
    },
    section: {
      backgroundColor: 'white',
      padding: '20px',
      borderRadius: '8px',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
      marginBottom: '25px',
      border: '1px solid #e0e0e0',
      width: '100%'
    },
    subheading: {
      borderBottom: '2px solid #4a90e2',
      paddingBottom: '10px',
      marginBottom: '20px',
      color: '#4a90e2',
      fontWeight: 'bold',
      fontSize: '20px'
    },
    p: {
      fontSize: '16px',
      lineHeight: '1.5',
      marginBottom: '15px'
    },
    architectureContainer: {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '20px 0',
      width: '100%'
    },
    treesContainer: {
      display: 'flex',
      justifyContent: 'center',
      gap: '25px',
      marginTop: '25px',
      flexWrap: 'wrap',
      width: '100%'
    },
    tree: {
      width: '200px',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      border: '1px solid #ddd',
      borderRadius: '8px',
      padding: '15px',
      backgroundColor: '#f5f9ff'
    },
    treeLabel: {
      fontWeight: 'bold',
      marginBottom: '12px',
      fontSize: '16px'
    },
    node: {
      width: '160px',
      height: '45px',
      backgroundColor: '#4a90e2',
      color: 'white',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      borderRadius: '5px',
      fontSize: '14px',
      position: 'relative',
      margin: '6px 0',
      padding: '5px',
      textAlign: 'center'
    },
    nodeConnection: {
      width: '2px',
      height: '18px',
      backgroundColor: '#4a90e2',
      margin: '0 auto'
    },
    leafNode: {
      width: '70px',
      height: '35px',
      backgroundColor: '#56b45d',
      color: 'white',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      borderRadius: '5px',
      fontSize: '14px'
    },
    leafContainer: {
      display: 'flex',
      justifyContent: 'space-around',
      width: '100%'
    },
    diagram: {
      width: '100%',
      marginTop: '25px'
    },
    table: {
      width: '100%',
      borderCollapse: 'collapse',
      marginTop: '20px',
      fontSize: '16px'
    },
    tableHeader: {
      backgroundColor: '#f5f5f5',
      padding: '12px',
      textAlign: 'left',
      borderBottom: '1px solid #ddd',
      fontWeight: 'bold',
      fontSize: '16px'
    },
    tableCell: {
      padding: '12px',
      borderBottom: '1px solid #ddd',
      fontSize: '15px',
      lineHeight: '1.5'
    },
    code: {
      fontFamily: 'monospace',
      backgroundColor: '#f5f5f5',
      padding: '3px 5px',
      borderRadius: '3px',
      fontSize: '15px'
    },
    arrow: {
      fontSize: '28px',
      color: '#4a90e2',
      textAlign: 'center',
      width: '100%'
    },
    flowStep: {
      display: 'flex',
      alignItems: 'center',
      margin: '15px 0',
      width: '100%'
    },
    flowBox: {
      flex: 1,
      padding: '15px',
      border: '1px solid #ddd',
      borderRadius: '5px',
      backgroundColor: 'white',
      fontSize: '16px',
      lineHeight: '1.5'
    },
    featureVec: {
      fontFamily: 'monospace',
      padding: '12px',
      backgroundColor: '#f0f8ff',
      borderRadius: '5px',
      marginTop: '12px',
      fontSize: '15px',
      overflowX: 'auto',
      whiteSpace: 'pre-wrap',
      wordBreak: 'break-word',
      lineHeight: '1.5'
    },
    highlight: {
      backgroundColor: '#fffacd',
      padding: '3px 5px',
      borderRadius: '3px'
    },
    note: {
      fontSize: '15px',
      color: '#666',
      fontStyle: 'italic',
      marginTop: '20px',
      padding: '15px',
      backgroundColor: '#f5f5f5',
      borderRadius: '5px',
      lineHeight: '1.5',
      width: '100%'
    },
    strong: {
      fontWeight: 'bold',
      fontSize: '16px'
    },
    li: {
      fontSize: '15px',
      lineHeight: '1.6',
      marginBottom: '5px'
    }
  };

  return React.createElement('div', { style: styles.container },
    React.createElement('h2', { style: styles.heading }, 'XGBoost Model Architecture'),
    
    React.createElement('div', { style: styles.section },
      React.createElement('h3', { style: styles.subheading }, 'Model Configuration'),
      React.createElement('p', { style: styles.p }, 'The XGBoost classifier is configured with the following parameters:'),
      
      React.createElement('table', { style: styles.table },
        React.createElement('thead', null,
          React.createElement('tr', null,
            React.createElement('th', { style: styles.tableHeader }, 'Parameter'),
            React.createElement('th', { style: styles.tableHeader }, 'Value'),
            React.createElement('th', { style: styles.tableHeader }, 'Trading Significance')
          )
        ),
        React.createElement('tbody', null,
          React.createElement('tr', null,
            React.createElement('td', { style: styles.tableCell }, 
              React.createElement('span', { style: styles.code }, 'max_depth')
            ),
            React.createElement('td', { style: styles.tableCell }, '4'),
            React.createElement('td', { style: styles.tableCell }, 'Moderate tree depth balances detail capture and generalization')
          ),
          React.createElement('tr', null,
            React.createElement('td', { style: styles.tableCell }, 
              React.createElement('span', { style: styles.code }, 'learning_rate')
            ),
            React.createElement('td', { style: styles.tableCell }, '0.03'),
            React.createElement('td', { style: styles.tableCell }, 'Low learning rate provides more stable predictions as market conditions evolve')
          ),
          React.createElement('tr', null,
            React.createElement('td', { style: styles.tableCell }, 
              React.createElement('span', { style: styles.code }, 'n_estimators')
            ),
            React.createElement('td', { style: styles.tableCell }, '250'),
            React.createElement('td', { style: styles.tableCell }, 'Sufficient number of trees to capture market relationships without overfitting')
          ),
          React.createElement('tr', null,
            React.createElement('td', { style: styles.tableCell }, 
              React.createElement('span', { style: styles.code }, 'eval_metric')
            ),
            React.createElement('td', { style: styles.tableCell }, '\'error@0.5\''),
            React.createElement('td', { style: styles.tableCell }, 'Optimized for error rate at 0.5 threshold, balanced for trading decisions')
          ),
          React.createElement('tr', null,
            React.createElement('td', { style: styles.tableCell }, 
              React.createElement('span', { style: styles.code }, 'min_child_weight')
            ),
            React.createElement('td', { style: styles.tableCell }, '8'),
            React.createElement('td', { style: styles.tableCell }, 'Controls model complexity and reduces overfitting to market noise')
          ),
          React.createElement('tr', null,
            React.createElement('td', { style: styles.tableCell }, 
              React.createElement('span', { style: styles.code }, 'gamma')
            ),
            React.createElement('td', { style: styles.tableCell }, '0.2'),
            React.createElement('td', { style: styles.tableCell }, 'Minimum loss reduction required for further tree partitioning, prevents capturing random market fluctuations')
          ),
          React.createElement('tr', null,
            React.createElement('td', { style: styles.tableCell }, 
              React.createElement('span', { style: styles.code }, 'subsample')
            ),
            React.createElement('td', { style: styles.tableCell }, '1.0'),
            React.createElement('td', { style: styles.tableCell }, 'Uses all training data for each tree, maximizing information capture in this case')
          ),
          React.createElement('tr', null,
            React.createElement('td', { style: styles.tableCell }, 
              React.createElement('span', { style: styles.code }, 'colsample_bytree')
            ),
            React.createElement('td', { style: styles.tableCell }, '0.8'),
            React.createElement('td', { style: styles.tableCell }, 'Each tree considers 80% of features, reducing overfitting to specific market signals')
          ),
          React.createElement('tr', null,
            React.createElement('td', { style: styles.tableCell }, 
              React.createElement('span', { style: styles.code }, 'reg_alpha')
            ),
            React.createElement('td', { style: styles.tableCell }, '0.2'),
            React.createElement('td', { style: styles.tableCell }, 'L1 regularization controls model sparsity, focusing on most significant market factors')
          ),
          React.createElement('tr', null,
            React.createElement('td', { style: styles.tableCell }, 
              React.createElement('span', { style: styles.code }, 'reg_lambda')
            ),
            React.createElement('td', { style: styles.tableCell }, '2'),
            React.createElement('td', { style: styles.tableCell }, 'L2 regularization prevents individual features from dominating prediction, improving stability')
          )
        )
      )
    ),
    
    React.createElement('div', { style: styles.section },
      React.createElement('h3', { style: styles.subheading }, 'Tree Structure Visualization'),
      
      React.createElement('p', { style: styles.p }, 'XGBoost uses an ensemble of gradient boosted decision trees. Each tree contributes to the final prediction, with later trees focusing on correcting errors made by earlier ones.'),
      
      React.createElement('div', { style: styles.architectureContainer },
        React.createElement('div', { style: styles.treesContainer },
          // Tree 1
          React.createElement('div', { style: styles.tree },
            React.createElement('div', { style: styles.treeLabel }, 'Tree 1'),
            React.createElement('div', { style: styles.node }, 
              'ticksFromSupportLevel', 
              React.createElement('br'), 
              '< 6.5 ?'
            ),
            React.createElement('div', { style: styles.nodeConnection }),
            React.createElement('div', { style: styles.node }, 
              'sameSideImbalance', 
              React.createElement('br'), 
              '< 0.63 ?'
            ),
            React.createElement('div', { style: styles.nodeConnection }),
            React.createElement('div', { style: styles.leafContainer },
              React.createElement('div', { style: styles.leafNode }, '0.23'),
              React.createElement('div', { style: styles.leafNode }, '0.78')
            )
          ),
          
          // Tree 2
          React.createElement('div', { style: styles.tree },
            React.createElement('div', { style: styles.treeLabel }, 'Tree 2'),
            React.createElement('div', { style: styles.node }, 
              'fillToDisplayRatio', 
              React.createElement('br'), 
              '< 12.5 ?'
            ),
            React.createElement('div', { style: styles.nodeConnection }),
            React.createElement('div', { style: styles.node }, 
              'ticksFromResistance', 
              React.createElement('br'), 
              '< 8.2 ?'
            ),
            React.createElement('div', { style: styles.nodeConnection }),
            React.createElement('div', { style: styles.leafContainer },
              React.createElement('div', { style: styles.leafNode }, '-0.12'),
              React.createElement('div', { style: styles.leafNode }, '0.41')
            )
          ),
          
          // Tree 3
          React.createElement('div', { style: styles.tree },
            React.createElement('div', { style: styles.treeLabel }, 'Tree 3'),
            React.createElement('div', { style: styles.node }, 
              '90sec_tradeImbalance', 
              React.createElement('br'), 
              '< 0.52 ?'
            ),
            React.createElement('div', { style: styles.nodeConnection }),
            React.createElement('div', { style: styles.node }, 
              'leanOverHedgeRatio', 
              React.createElement('br'), 
              '< 0.83 ?'
            ),
            React.createElement('div', { style: styles.nodeConnection }),
            React.createElement('div', { style: styles.leafContainer },
              React.createElement('div', { style: styles.leafNode }, '-0.09'),
              React.createElement('div', { style: styles.leafNode }, '0.32')
            )
          ),
          
          // Tree N
          React.createElement('div', { style: styles.tree },
            React.createElement('div', { style: styles.treeLabel }, 'Tree N (of 250)'),
            React.createElement('div', { style: styles.node }, 
              'firstNoticeDays', 
              React.createElement('br'), 
              '< 21.5 ?'
            ),
            React.createElement('div', { style: styles.nodeConnection }),
            React.createElement('div', { style: styles.node }, 
              'nReloads', 
              React.createElement('br'), 
              '< 10.5 ?'
            ),
            React.createElement('div', { style: styles.nodeConnection }),
            React.createElement('div', { style: styles.leafContainer },
              React.createElement('div', { style: styles.leafNode }, '0.05'),
              React.createElement('div', { style: styles.leafNode }, '-0.03')
            )
          )
        ),
        
        React.createElement('div', { style: styles.note },
          'Tree diagrams are simplified representations. Actual trees may have more complex branching structures.',
          ' Each leaf node contains a prediction value that contributes to the final ensemble prediction.'
        )
      )
    ),
    
    React.createElement('div', { style: styles.section },
      React.createElement('h3', { style: styles.subheading }, 'Prediction Flow'),
      
      React.createElement('p', { style: styles.p }, 'When making a prediction for a new iceberg order, the model processes it through these steps:'),
      
      React.createElement('div', { style: styles.flowStep },
        React.createElement('div', { style: styles.flowBox },
          React.createElement('strong', { style: styles.strong }, '1. Feature Vector Construction'),
          React.createElement('p', { style: styles.p }, 'Raw order data is transformed into engineered features'),
          React.createElement('div', { style: styles.featureVec },
            '[ticksFromSupportLevel=5.0, ticksFromResistanceLevel=12.0, fillToDisplayRatio=14.0, ',
            'sameSideImbalance=0.73, tradeImbalance90s=0.62, leanOverHedgeRatio=0.75, ...]'
          )
        )
      ),
      
      React.createElement('div', { style: styles.arrow, align: 'center' }, '↓'),
      
      React.createElement('div', { style: styles.flowStep },
        React.createElement('div', { style: styles.flowBox },
          React.createElement('strong', { style: styles.strong }, '2. Feature Scaling'),
          React.createElement('p', { style: styles.p }, 'Features are normalized using the stored scaling parameters'),
          React.createElement('div', { style: styles.featureVec },
            '[ticksFromSupportLevel=0.37, ticksFromResistanceLevel=0.76, fillToDisplayRatio=1.28, ',
            'sameSideImbalance=0.91, tradeImbalance90s=0.44, leanOverHedgeRatio=0.58, ...]'
          )
        )
      ),
      
      React.createElement('div', { style: styles.arrow, align: 'center' }, '↓'),
      
      React.createElement('div', { style: styles.flowStep },
        React.createElement('div', { style: styles.flowBox },
          React.createElement('strong', { style: styles.strong }, '3. Tree Ensemble Processing'),
          React.createElement('p', { style: styles.p }, 'The feature vector passes through all 250 decision trees'),
          React.createElement('div', null,
            React.createElement('ul', { style: {margin: '5px 0', paddingLeft: '20px'} },
              React.createElement('li', { style: styles.li }, 'Tree 1 Output: 0.78'),
              React.createElement('li', { style: styles.li }, 'Tree 2 Output: 0.41'),
              React.createElement('li', { style: styles.li }, 'Tree 3 Output: 0.32'),
              React.createElement('li', { style: styles.li }, '...'),
              React.createElement('li', { style: styles.li }, 'Tree 250 Output: -0.03')
            )
          )
        )
      ),
      
      React.createElement('div', { style: styles.arrow, align: 'center' }, '↓'),
      
      React.createElement('div', { style: styles.flowStep },
        React.createElement('div', { style: styles.flowBox },
          React.createElement('strong', { style: styles.strong }, '4. Prediction Combination'),
          React.createElement('p', { style: styles.p }, 'Tree outputs are combined and transformed to a probability'),
          React.createElement('div', null,
            React.createElement('p', { style: styles.p }, 'Sum of tree outputs (weighted by learning rate): 1.25'),
            React.createElement('p', { style: styles.p }, 'Logistic transformation: ', 
              React.createElement('span', { style: styles.code }, 'sigmoid(1.25) = 0.778')
            ),
            React.createElement('p', { style: styles.p }, 'Final prediction: ', 
              React.createElement('span', { style: styles.highlight }, '77.8% probability of execution')
            )
          )
        )
      ),
      
      React.createElement('div', { style: styles.note },
        React.createElement('strong', { style: styles.strong }, 'Trading Insight:'), 
        ' The model produces a probability of execution rather than just a binary yes/no. ',
        'This probability can be used to prioritize trading opportunities, size positions dynamically, or adjust ',
        'trading strategies based on confidence levels.'
      )
    ),
    
    React.createElement('div', { style: styles.section },
      React.createElement('h3', { style: styles.subheading }, 'Model Advantages for Trading Applications'),
      
      React.createElement('table', { style: styles.table },
        React.createElement('thead', null,
          React.createElement('tr', null,
            React.createElement('th', { style: styles.tableHeader }, 'Advantage'),
            React.createElement('th', { style: styles.tableHeader }, 'Description'),
            React.createElement('th', { style: styles.tableHeader }, 'Trading Application')
          )
        ),
        React.createElement('tbody', null,
          React.createElement('tr', null,
            React.createElement('td', { style: styles.tableCell }, 'Non-Linear Relationships'),
            React.createElement('td', { style: styles.tableCell }, 'Captures complex, non-linear interactions between market variables'),
            React.createElement('td', { style: styles.tableCell }, 'Better models tipping points and threshold effects in market behavior')
          ),
          React.createElement('tr', null,
            React.createElement('td', { style: styles.tableCell }, 'Robust to Feature Scaling'),
            React.createElement('td', { style: styles.tableCell }, 'Tree-based models are less sensitive to feature scaling than neural networks'),
            React.createElement('td', { style: styles.tableCell }, 'More stable in production when market metrics have unusual ranges')
          ),
          React.createElement('tr', null,
            React.createElement('td', { style: styles.tableCell }, 'Handles Missing Values'),
            React.createElement('td', { style: styles.tableCell }, 'XGBoost can handle missing values in features'),
            React.createElement('td', { style: styles.tableCell }, 'Resilient against data quality issues in live market feeds')
          ),
          React.createElement('tr', null,
            React.createElement('td', { style: styles.tableCell }, 'Interpretable Structure'),
            React.createElement('td', { style: styles.tableCell }, 'Individual trees can be examined for trading logic'),
            React.createElement('td', { style: styles.tableCell }, 'Easier to explain to regulatory bodies and trading strategy committees')
          ),
          React.createElement('tr', null,
            React.createElement('td', { style: styles.tableCell }, 'Fast Inference'),
            React.createElement('td', { style: styles.tableCell }, 'Tree traversal is computationally efficient'),
            React.createElement('td', { style: styles.tableCell }, 'Low latency prediction suitable for high-frequency trading systems')
          ),
          React.createElement('tr', null,
            React.createElement('td', { style: styles.tableCell }, 'Built-in Regularization'),
            React.createElement('td', { style: styles.tableCell }, 'Prevents overfitting to market noise'),
            React.createElement('td', { style: styles.tableCell }, 'More stable performance across changing market regimes')
          )
        )
      )
    )
  );
};

module.exports = ModelArchitecture;
