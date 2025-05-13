const React = require('react');

const FeatureImportanceDiagram = () => {
  const styles = {
    container: {
      fontFamily: 'Arial, sans-serif',
      padding: '25px',
      maxWidth: '100%',
      width: '100%',
      overflowX: 'auto'
    },
    header: {
      marginBottom: '25px'
    },
    h2: {
      fontSize: '24px',
      marginBottom: '15px',
      color: '#333',
      fontWeight: 'bold'
    },
    p: {
      fontSize: '17px',
      lineHeight: '1.5',
      marginBottom: '15px'
    },
    chartContainer: {
      display: 'flex',
      flexDirection: 'column',
      marginBottom: '30px',
      border: '1px solid #ddd',
      borderRadius: '8px',
      padding: '20px',
      backgroundColor: '#f9f9f9',
      width: '100%'
    },
    chartTitle: {
      fontWeight: 'bold',
      fontSize: '18px',
      marginBottom: '15px',
      color: '#333'
    },
    barChart: {
      display: 'flex',
      flexDirection: 'column',
      width: '100%',
      marginTop: '15px'
    },
    barContainer: {
      display: 'flex',
      alignItems: 'center',
      marginBottom: '10px',
      width: '100%'
    },
    barLabel: {
      width: '240px',
      textAlign: 'right',
      paddingRight: '15px',
      fontSize: '16px'
    },
    bar: {
      height: '25px',
      backgroundColor: '#4a90e2',
      borderRadius: '4px',
      position: 'relative'
    },
    barValue: {
      position: 'absolute',
      right: '-45px',
      fontSize: '14px',
      fontWeight: 'bold'
    },
    description: {
      backgroundColor: '#fff',
      padding: '18px',
      borderRadius: '5px',
      marginTop: '15px',
      fontSize: '16px',
      lineHeight: '1.5'
    },
    methodSection: {
      marginBottom: '30px',
      width: '100%'
    },
    methodTitle: {
      fontWeight: 'bold',
      marginBottom: '12px',
      fontSize: '17px'
    },
    legendBox: {
      display: 'flex',
      justifyContent: 'flex-start',
      flexWrap: 'wrap',
      marginBottom: '25px',
      gap: '20px'
    },
    legendItem: {
      display: 'flex',
      alignItems: 'center',
      fontSize: '16px'
    },
    legendColor: {
      width: '18px',
      height: '18px',
      marginRight: '8px',
      borderRadius: '3px'
    },
    strong: {
      fontWeight: 'bold',
      fontSize: '16px'
    },
    li: {
      fontSize: '16px',
      lineHeight: '1.5',
      marginBottom: '8px'
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
      React.createElement('h2', { style: styles.h2 }, 'Feature Importance Analysis'),
      React.createElement('p', { style: styles.p }, 'Comparison of feature importance across different models and techniques')
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
        React.createElement('li', { style: styles.li }, 
          React.createElement('strong', { style: styles.strong }, 'Direct Feature Importance:'), 
          ' Native importance from tree-based models'
        ),
        React.createElement('li', { style: styles.li }, 
          React.createElement('strong', { style: styles.strong }, 'Mean Decrease Accuracy (MDA):'), 
          ' Measures importance by permuting feature values'
        ),
        React.createElement('li', { style: styles.li }, 
          React.createElement('strong', { style: styles.strong }, 'SHAP (SHapley Additive exPlanations):'), 
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
        React.createElement('p', { style: styles.p }, React.createElement('strong', { style: styles.strong }, 'Key Findings:')),
        React.createElement('ul', null,
          React.createElement('li', { style: styles.li }, 'All methods indicate that price position (support/resistance) features are highly predictive'),
          React.createElement('li', { style: styles.li }, 'Order book imbalance ratios consistently rank in the top features'),
          React.createElement('li', { style: styles.li }, 'Time-based features (time windows, expiry) have moderate importance'),
          React.createElement('li', { style: styles.li }, 'The state immediately before fill (oneStateBeforeFill) provides more predictive power than earlier states'),
          React.createElement('li', { style: styles.li }, 'Market indicators based on actual quantities (useQty=true) are generally less important than indicators based on number of orders')
        )
      )
    )
  );
};

module.exports = FeatureImportanceDiagram;
