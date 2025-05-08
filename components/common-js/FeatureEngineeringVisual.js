const React = require('react');

const FeatureEngineeringVisual = () => {
  // Styles for visual elements
  const styles = {
    container: {
      fontFamily: 'Arial, sans-serif',
      padding: '20px',
      maxWidth: '100%',
      backgroundColor: '#f9f9f9',
      borderRadius: '8px',
      border: '1px solid #e0e0e0'
    },
    heading: {
      color: '#333',
      marginBottom: '20px'
    },
    featureGroup: {
      backgroundColor: 'white',
      padding: '15px',
      borderRadius: '8px',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
      marginBottom: '20px',
      border: '1px solid #e0e0e0'
    },
    groupTitle: {
      borderBottom: '2px solid #4a90e2',
      paddingBottom: '8px',
      marginBottom: '15px',
      color: '#4a90e2',
      fontWeight: 'bold'
    },
    featureTable: {
      width: '100%',
      borderCollapse: 'collapse',
      marginTop: '10px',
      fontSize: '14px'
    },
    tableHeader: {
      backgroundColor: '#f5f5f5',
      padding: '10px',
      textAlign: 'left',
      borderBottom: '1px solid #ddd',
      fontWeight: 'bold'
    },
    tableCell: {
      padding: '10px',
      borderBottom: '1px solid #ddd',
      verticalAlign: 'top'
    },
    code: {
      fontFamily: 'monospace',
      backgroundColor: '#f5f5f5',
      padding: '2px 4px',
      borderRadius: '3px',
      fontSize: '13px'
    },
    exampleBox: {
      backgroundColor: '#f0f8ff',
      padding: '10px',
      borderRadius: '5px',
      marginTop: '10px',
      border: '1px solid #d0e3f0',
      fontSize: '14px'
    },
    highlight: {
      backgroundColor: '#fffacd',
      padding: '2px 4px',
      borderRadius: '3px'
    },
    transformation: {
      display: 'flex',
      alignItems: 'center',
      margin: '15px 0',
      gap: '15px'
    },
    transformStep: {
      flex: 1,
      backgroundColor: 'white',
      padding: '10px',
      borderRadius: '5px',
      border: '1px solid #ddd',
      minHeight: '80px'
    },
    arrow: {
      fontSize: '24px',
      color: '#4a90e2'
    },
    title: {
      fontWeight: 'bold',
      marginBottom: '5px',
      fontSize: '14px'
    },
    note: {
      fontSize: '14px',
      color: '#666',
      fontStyle: 'italic',
      marginTop: '8px'
    },
    importanceIndicator: {
      display: 'flex',
      marginTop: '5px',
      alignItems: 'center'
    },
    importanceLabel: {
      marginRight: '10px',
      fontSize: '13px',
      width: '100px'
    },
    importanceBar: {
      height: '8px',
      borderRadius: '4px'
    }
  };

  // Feature examples and descriptions
  const featureGroups = [
    {
      title: "Order Book Position Features",
      features: [
        {
          name: "ticksFromSupportLevel",
          description: "Distance from local support level, determined by order direction",
          code: "np.where(df.isBid==True, df['ticksFromLow'], df['ticksFromHigh'])",
          example: "For a Buy order: 5 ticks above the lowest recent price\nFor a Sell order: 7 ticks below the highest recent price",
          importance: 0.92,
          tradingSignificance: "Orders closer to support levels (for buys) have higher fill probability due to price bounces"
        },
        {
          name: "ticksFromResistanceLevel",
          description: "Distance from local resistance level, determined by order direction",
          code: "np.where(df.isBid!=True, df['ticksFromLow'], df['ticksFromHigh'])",
          example: "For a Buy order: 12 ticks below the highest recent price\nFor a Sell order: 3 ticks above the lowest recent price",
          importance: 0.88,
          tradingSignificance: "Orders closer to resistance levels (for sells) have higher fill probability due to price reversals"
        }
      ]
    },
    {
      title: "Market Imbalance Features",
      features: [
        {
          name: "oneStateBeforeFill_90sec_tradeImbalance",
          description: "Trade imbalance (buys vs sells) over 90-second window before fill state",
          code: "flattened_df.columns=flattened_df.columns.map(\"_\".join).to_series().apply(lambda x:f\"{col}_\"+x).tolist()",
          example: "Value: 0.557 (more buying pressure than selling in 90-second window)",
          importance: 0.75,
          tradingSignificance: "Higher same-sided trade imbalance correlates with increased order fill probability"
        },
        {
          name: "oneStateBeforeFill_sameSideImbalance",
          description: "Order book imbalance relative to the order's side (buy/sell)",
          code: "df[col.replace(\"bid\",\"sameSide\")] = np.where(df.isBid==True, df[col], 1-df[col])",
          example: "Buy order with book imbalance tilted toward buys: 0.72\nSell order with book imbalance tilted toward sells: 0.65",
          importance: 0.81,
          tradingSignificance: "Stronger book imbalance on same side means harder to get filled, unless iceberg is using it as cover"
        }
      ]
    },
    {
      title: "Order Dynamics Features",
      features: [
        {
          name: "oneStateBeforeFill_fillToDisplayRatio",
          description: "Ratio of filled quantity to displayed quantity in the order book",
          code: "flattened_dfNumOrders = pd.concat([df.T.stack().reset_index(level=1, drop=True) for df in expandedNumOrders], axis=1)",
          example: "Value: 15.0 (15 times more quantity executed than visible on the book)",
          importance: 0.85,
          tradingSignificance: "Higher ratios indicate aggressive icebergs that are more likely to complete execution"
        },
        {
          name: "oneStateBeforeFill_leanOverHedgeRatio",
          description: "Ratio of directional pressure from hedging activity",
          code: "filtered_dfQty.columns=filtered_dfQty.columns.to_series().apply(lambda x:x+\"Qty\").tolist()",
          example: "Value: 0.75 (moderate hedging pressure in direction of order)",
          importance: 0.77,
          tradingSignificance: "Stronger hedging flows increase likelihood of order execution completion"
        }
      ]
    },
    {
      title: "Temporal Features",
      features: [
        {
          name: "firstNoticeDays",
          description: "Days until the first notice date for the futures contract",
          code: "df.apply(lambda row: firstNoticeDays(row['expirationMonthYear'], row['tradeDate']), axis=1)",
          example: "Value: 28 (28 days until contract first notice date)",
          importance: 0.70,
          tradingSignificance: "Orders closer to expiration often have higher execution urgency and completion rates"
        },
        {
          name: "numAggressivePriceChanges",
          description: "Count of aggressive price movements in the direction of the order",
          code: "[Original feature from dataset]",
          example: "Value: 3 (three aggressive price moves in order direction)",
          importance: 0.68,
          tradingSignificance: "More aggressive price action indicates stronger directional conviction and higher fill probability"
        }
      ]
    }
  ];

  // Side imbalance transformation example
  const sideImbalanceTransformation = [
    {
      title: "Raw Book Data",
      content: "Buy/Sell Imbalance: 0.75 (bid)\nAsk Volume: 25\nBid Volume: 75"
    },
    {
      title: "Consider Order Side",
      content: "Order Type: SELL\nImbalance in raw form is not directly useful for the model"
    },
    {
      title: "Side-Relative Transformation",
      content: "sameSideImbalance = 1 - 0.75 = 0.25\n(low value indicates unfavorable book condition for this sell order)"
    }
  ];

  // Support level transformation example
  const supportLevelTransformation = [
    {
      title: "Raw Price Data",
      content: "Current Price: 100.25\nRecent High: 102.50\nRecent Low: 98.75\nTicksFromHigh: 9\nTicksFromLow: 6"
    },
    {
      title: "Consider Order Side",
      content: "Order Type: BUY\nFor buy orders, support is relevant\nFor sell orders, resistance is relevant"
    },
    {
      title: "Side-Adjusted Feature",
      content: "ticksFromSupportLevel = 6\n(buy order is 6 ticks from support level)\nticksFromResistanceLevel = 9\n(buy order is 9 ticks from resistance)"
    }
  ];

  return React.createElement('div', { style: styles.container },
    React.createElement('h2', { style: styles.heading }, 'Feature Engineering Deep Dive'),
    
    featureGroups.map((group, groupIndex) => 
      React.createElement('div', { key: groupIndex, style: styles.featureGroup },
        React.createElement('h3', { style: styles.groupTitle }, group.title),
        
        React.createElement('table', { style: styles.featureTable },
          React.createElement('thead', null,
            React.createElement('tr', null,
              React.createElement('th', { style: styles.tableHeader }, 'Feature'),
              React.createElement('th', { style: styles.tableHeader }, 'Description'),
              React.createElement('th', { style: styles.tableHeader }, 'Trading Significance'),
              React.createElement('th', { style: styles.tableHeader }, 'Importance')
            )
          ),
          React.createElement('tbody', null,
            group.features.map((feature, featureIndex) => 
              React.createElement('tr', { key: featureIndex },
                React.createElement('td', { style: styles.tableCell },
                  React.createElement('strong', null, feature.name),
                  React.createElement('div', { style: {marginTop: '8px'} },
                    React.createElement('span', { style: styles.code }, feature.code)
                  )
                ),
                React.createElement('td', { style: styles.tableCell },
                  feature.description,
                  React.createElement('div', { style: styles.exampleBox },
                    React.createElement('strong', null, 'Example:'),
                    React.createElement('br'),
                    feature.example
                  )
                ),
                React.createElement('td', { style: styles.tableCell },
                  feature.tradingSignificance
                ),
                React.createElement('td', { style: styles.tableCell },
                  React.createElement('div', { style: styles.importanceIndicator },
                    React.createElement('div', { style: styles.importanceLabel }, 'Predictive Power:'),
                    React.createElement('div', 
                      { 
                        style: {
                          ...styles.importanceBar, 
                          width: `${feature.importance * 100}px`,
                          backgroundColor: `rgba(74, 144, 226, ${feature.importance})`
                        }
                      }
                    )
                  )
                )
              )
            )
          )
        )
      )
    ),
    
    React.createElement('div', { style: styles.featureGroup },
      React.createElement('h3', { style: styles.groupTitle }, 'Feature Transformation Examples'),
      
      React.createElement('div', null,
        React.createElement('h4', null, 'Example 1: Side-Relative Order Book Imbalance'),
        React.createElement('div', { style: styles.transformation },
          sideImbalanceTransformation.map((step, index) => 
            React.createElement(React.Fragment, { key: index },
              React.createElement('div', { style: styles.transformStep },
                React.createElement('div', { style: styles.title }, step.title),
                React.createElement('pre', { style: {margin: 0, whiteSpace: 'pre-wrap'} }, step.content)
              ),
              index < sideImbalanceTransformation.length - 1 && 
                React.createElement('div', { style: styles.arrow }, '→')
            )
          )
        ),
        
        React.createElement('div', { style: styles.note },
          React.createElement('strong', null, 'Trading Insight:'), 
          ' Converting raw order book imbalance to a side-relative metric creates a consistent feature that indicates favorable/unfavorable market conditions regardless of whether the iceberg is a buy or sell.'
        )
      ),
      
      React.createElement('div', { style: {marginTop: '30px'} },
        React.createElement('h4', null, 'Example 2: Support/Resistance Level Positioning'),
        React.createElement('div', { style: styles.transformation },
          supportLevelTransformation.map((step, index) => 
            React.createElement(React.Fragment, { key: index },
              React.createElement('div', { style: styles.transformStep },
                React.createElement('div', { style: styles.title }, step.title),
                React.createElement('pre', { style: {margin: 0, whiteSpace: 'pre-wrap'} }, step.content)
              ),
              index < supportLevelTransformation.length - 1 && 
                React.createElement('div', { style: styles.arrow }, '→')
            )
          )
        ),
        
        React.createElement('div', { style: styles.note },
          React.createElement('strong', null, 'Trading Insight:'), 
          ' Buy orders are more likely to be filled when close to support levels, while sell orders are more likely to be filled when close to resistance levels. This transformation captures that market dynamic in a consistent way.'
        )
      )
    )
  );
};

module.exports = FeatureEngineeringVisual;
