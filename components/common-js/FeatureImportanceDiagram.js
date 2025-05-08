const React = require('react');

const FeatureImportanceDiagram = () => {
  const styles = {
    container: {
      fontFamily: 'Arial, sans-serif',
      padding: '20px',
      maxWidth: '100%'
    },
    header: {
      marginBottom: '20px'
    },
    chartContainer: {
      display: 'flex',
      flexDirection: 'column',
      marginBottom: '30px',
      border: '1px solid #ddd',
      borderRadius: '8px',
      padding: '15px',
      backgroundColor: '#f9f9f9'
    },
    chartTitle: {
      fontWeight: 'bold',
      fontSize: '16px',
      marginBottom: '10px'
    },
    barChart: {
      display: 'flex',
      flexDirection: 'column',
      width: '100%',
      marginTop: '10px'
    },
    barContainer: {
      display: 'flex',
      alignItems: 'center',
      marginBottom: '8px'
    },
    barLabel: {
      width: '210px',
      textAlign: 'right',
      paddingRight: '10px',
      fontSize: '14px'
    },
    bar: {
      height: '20px',
      backgroundColor: '#4a90e2',
      borderRadius: '3px',
      position: 'relative'
    },
    barValue: {
      position: 'absolute',
      right: '-40px',
      fontSize: '12px'
    },
    description: {
      backgroundColor: '#fff',
      padding: '15px',
      borderRadius: '5px',
      marginTop: '10px',
      fontSize: '14px',
      lineHeight: '1.4'
    },
    methodSection: {
      marginBottom: '30px'
    },
    methodTitle: {
      fontWeight: 'bold',
      marginBottom: '8px'
    },
    legendBox: {
      display: 'flex',
      justifyContent: 'flex-start',
      flexWrap: 'wrap',
      marginBottom: '20px',
      gap: '15px'
    },
    legendItem: {
      display: 'flex',
      alignItems: 'center'
    },
    legendColor: {
      width: '15px',
      height: '15px',
      marginRight: '5px',
      borderRadius: '3px'
    }
  };

  // Mock feature importance values
  const modelFeatureImportance = {
    'XGBoost': [
      { feature: 'ticksFromSupportLevel', importance: 0.186 },
      { feature: 'ticksFromResistanceLevel', importance: 0.152 },
      { feature: 'oneStateBeforeFill_fillToDisplayRatio', importance: 0.087 },
      { feature: 'oneStateBeforeFill_leanOverHedgeRatio', importance: 0.071 },
      { feature: 'oneStateBeforeFill_plusOneLevelSameSideMedianRatio', importance: 0.069 },
      { feature: 'oneStateBeforeFill_minusOneLevelOtherSideMedianRatio', importance: 0.065 },
      { feature: 'oneStateBeforeFill_90sec_tradeImbalance', importance: 0.061 },
      { feature: 'oneStateBeforeFill_100msg_tradeImbalance', importance: 0.058 },
      { feature: 'monthsToExpiry', importance: 0.054 },
      { feature: 'numAggressivePriceChanges', importance: 0.053 }
    ],
    'RandomForest': [
      { feature: 'ticksFromSupportLevel', importance: 0.172 },
      { feature: 'ticksFromResistanceLevel', importance: 0.143 },
      { feature: 'oneStateBeforeFill_fillToDisplayRatio', importance: 0.092 },
      { feature: 'monthsToExpiry', importance: 0.075 },
      { feature: 'oneStateBeforeFill_leanOverHedgeRatio', importance: 0.065 },
      { feature: 'oneStateBeforeFill_90sec_tradeImbalance', importance: 0.064 },
      { feature: 'oneStateBeforeFill_plusOneLevelSameSideMedianRatio', importance: 0.063 },
      { feature: 'oneStateBeforeFill_100msg_tradeImbalance', importance: 0.056 },
      { feature: 'numAggressivePriceChanges', importance: 0.055 },
      { feature: 'oneStateBeforeFill_minusOneLevelOtherSideMedianRatio', importance: 0.053 }
    ]
  };

  return React.createElement('div', { style: styles.container },
    React.createElement('div', { style: styles.header },
      React.createElement('h2', null, 'Feature Importance Analysis'),
      React.createElement('p', null, 'Comparison of feature importance across different models and techniques')
    ),
    
    React.createElement('div', { style: styles.legendBox },
      React.createElement('div', { style: styles.legendItem },
        React.createElement('div', { style: {...styles.legendColor, backgroundColor: '#4a90e2'} }),
        React.createElement('span', null, 'XGBoost')
      ),
      React.createElement('div', { style: styles.legendItem },
        React.createElement('div', { style: {...styles.legendColor, backgroundColor: '#56b45d'} }),
        React.createElement('span', null, 'RandomForest')
      ),
      React.createElement('div', { style: styles.legendItem },
        React.createElement('div', { style: {...styles.legendColor, backgroundColor: '#e2984a'} }),
        React.createElement('span', null, 'MDA')
      ),
      React.createElement('div', { style: styles.legendItem },
        React.createElement('div', { style: {...styles.legendColor, backgroundColor: '#8e44ad'} }),
        React.createElement('span', null, 'SHAP')
      )
    ),
    
    React.createElement('div', { style: styles.methodSection },
      React.createElement('div', { style: styles.methodTitle }, 'Feature Importance Methods Used in the Project:'),
      React.createElement('ul', null,
        React.createElement('li', null, 
          React.createElement('strong', null, 'Direct Feature Importance:'), 
          ' Native importance from tree-based models'
        ),
        React.createElement('li', null, 
          React.createElement('strong', null, 'Mean Decrease Accuracy (MDA):'), 
          ' Measures importance by permuting feature values'
        ),
        React.createElement('li', null, 
          React.createElement('strong', null, 'SHAP (SHapley Additive exPlanations):'), 
          ' Game theory-based unified approach to feature importance'
        )
      )
    ),
    
    React.createElement('div', { style: styles.chartContainer },
      React.createElement('div', { style: styles.chartTitle }, 'XGBoost Feature Importance'),
      React.createElement('div', { style: styles.barChart },
        modelFeatureImportance['XGBoost'].map((item, index) => 
          React.createElement('div', { key: index, style: styles.barContainer },
            React.createElement('div', { style: styles.barLabel }, item.feature),
            React.createElement('div', { 
              style: {
                ...styles.bar, 
                width: `${item.importance * 500}px`,
                backgroundColor: '#4a90e2'
              }
            },
              React.createElement('span', { style: styles.barValue }, item.importance.toFixed(3))
            )
          )
        )
      ),
      React.createElement('div', { style: styles.description },
        'XGBoost importance shows that position-related features (ticks from support/resistance) and order ratio features have the highest impact on prediction outcomes.'
      )
    ),
    
    React.createElement('div', { style: styles.chartContainer },
      React.createElement('div', { style: styles.chartTitle }, 'RandomForest Feature Importance'),
      React.createElement('div', { style: styles.barChart },
        modelFeatureImportance['RandomForest'].map((item, index) => 
          React.createElement('div', { key: index, style: styles.barContainer },
            React.createElement('div', { style: styles.barLabel }, item.feature),
            React.createElement('div', { 
              style: {
                ...styles.bar, 
                width: `${item.importance * 500}px`,
                backgroundColor: '#56b45d'
              }
            },
              React.createElement('span', { style: styles.barValue }, item.importance.toFixed(3))
            )
          )
        )
      ),
      React.createElement('div', { style: styles.description },
        'RandomForest importance largely agrees with XGBoost, with slightly more weight given to temporal features like monthsToExpiry.'
      )
    ),
    
    React.createElement('div', { style: styles.chartContainer },
      React.createElement('div', { style: styles.chartTitle }, 'Feature Importance Comparison'),
      React.createElement('div', { style: styles.description },
        React.createElement('p', null, React.createElement('strong', null, 'Key Findings:')),
        React.createElement('ul', null,
          React.createElement('li', null, 'All methods indicate that price position (support/resistance) features are highly predictive'),
          React.createElement('li', null, 'Order book imbalance ratios consistently rank in the top features'),
          React.createElement('li', null, 'Time-based features (time windows, expiry) have moderate importance'),
          React.createElement('li', null, 'The state immediately before fill (oneStateBeforeFill) provides more predictive power than earlier states'),
          React.createElement('li', null, 'Market indicators based on actual quantities (useQty=true) are generally less important than indicators based on number of orders')
        )
      )
    )
  );
};

module.exports = FeatureImportanceDiagram;
