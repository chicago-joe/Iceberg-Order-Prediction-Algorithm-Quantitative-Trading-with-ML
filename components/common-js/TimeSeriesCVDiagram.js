const React = require('react');

const TimeSeriesCVDiagram = () => {
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
    h3: {
      fontSize: '20px',
      marginBottom: '15px',
      color: '#333',
      fontWeight: 'bold'
    },
    p: {
      fontSize: '17px',
      lineHeight: '1.5',
      marginBottom: '15px'
    },
    code: {
      fontFamily: 'monospace',
      backgroundColor: '#f5f5f5',
      padding: '3px 5px',
      borderRadius: '3px',
      fontSize: '15px'
    },
    pre: {
      fontFamily: 'monospace',
      backgroundColor: '#f5f5f5',
      padding: '15px',
      borderRadius: '5px',
      fontSize: '16px',
      lineHeight: '1.5',
      overflowX: 'auto',
      whiteSpace: 'pre-wrap',
      marginTop: '15px'
    },
    visualContainer: {
      border: '1px solid #ddd',
      borderRadius: '8px',
      padding: '25px',
      marginBottom: '30px',
      backgroundColor: '#fafafa',
      width: '100%'
    },
    visualization: {
      display: 'flex',
      flexDirection: 'column',
      gap: '18px',
      marginTop: '25px',
      width: '100%'
    },
    timeBlock: {
      display: 'flex',
      alignItems: 'center',
      height: '45px'
    },
    dateLabel: {
      width: '100px',
      textAlign: 'right',
      paddingRight: '12px',
      fontSize: '15px',
      fontWeight: 'bold'
    },
    timeBlockContainer: {
      display: 'flex',
      width: '100%',
      position: 'relative'
    },
    trainBlock: {
      height: '100%',
      backgroundColor: '#4a90e2',
      borderRadius: '4px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#000',
      fontSize: '16px',
      fontWeight: 'bold'
    },
    testBlock: {
      height: '100%',
      backgroundColor: '#e2984a',
      borderRadius: '4px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#000',
      fontSize: '16px',
      fontWeight: 'bold'
    },
    unusedBlock: {
      height: '100%',
      backgroundColor: '#ddd',
      opacity: 0.5,
      borderRadius: '4px'
    },
    legend: {
      display: 'flex',
      gap: '25px',
      marginBottom: '25px'
    },
    legendItem: {
      display: 'flex',
      alignItems: 'center',
      fontSize: '16px'
    },
    legendColor: {
      width: '22px',
      height: '22px',
      marginRight: '10px',
      borderRadius: '4px'
    },
    notes: {
      fontSize: '16px',
      lineHeight: '1.6',
      marginTop: '25px',
      padding: '20px',
      backgroundColor: '#f5f5f5',
      borderRadius: '5px',
      border: '1px solid #eee'
    },
    li: {
      marginBottom: '8px',
      lineHeight: '1.6'
    }
  };

  // Sample dates for visualization
  const dates = [
    '2023-07-03', '2023-07-10', '2023-07-17', '2023-07-24',
    '2023-07-31', '2023-08-07', '2023-08-14', '2023-08-21'
  ];

  // Time series CV fold configurations using train_size=2 days
  // This matches the optimal train_size found in hyperparameter optimization
  const folds = [
    {
      trainStart: 0,
      trainEnd: 1,
      testStart: 2,
      testEnd: 2
    },
    {
      trainStart: 1,
      trainEnd: 2,
      testStart: 3,
      testEnd: 3
    },
    {
      trainStart: 2,
      trainEnd: 3,
      testStart: 4,
      testEnd: 4
    },
    {
      trainStart: 3,
      trainEnd: 4,
      testStart: 5,
      testEnd: 5
    },
    {
      trainStart: 4,
      trainEnd: 5,
      testStart: 6,
      testEnd: 6
    },
    {
      trainStart: 5,
      trainEnd: 6,
      testStart: 7,
      testEnd: 7
    }
  ];

  return React.createElement('div', { style: styles.container },
    React.createElement('div', { style: styles.header },
      React.createElement('h2', { style: styles.h2 }, 'Time Series Cross-Validation Approach'),
      React.createElement('p', { style: styles.p }, 'The project uses a rolling window approach to handle the time-dependent nature of financial data')
    ),
    
    React.createElement('div', { style: styles.legend },
      React.createElement('div', { style: styles.legendItem },
        React.createElement('div', { style: {...styles.legendColor, backgroundColor: '#4a90e2'} }),
        React.createElement('span', null, 'Training Period')
      ),
      React.createElement('div', { style: styles.legendItem },
        React.createElement('div', { style: {...styles.legendColor, backgroundColor: '#e2984a'} }),
        React.createElement('span', null, 'Testing Period')
      ),
      React.createElement('div', { style: styles.legendItem },
        React.createElement('div', { style: {...styles.legendColor, backgroundColor: '#ddd'} }),
        React.createElement('span', null, 'Unused Data')
      )
    ),
    
    React.createElement('div', { style: styles.visualContainer },
      React.createElement('h3', { style: styles.h3 }, 'Rolling Window Cross-Validation'),
      
      React.createElement('div', { style: styles.visualization },
        folds.map((fold, foldIndex) => 
          React.createElement('div', { key: foldIndex, style: styles.timeBlock },
            React.createElement('div', { style: styles.dateLabel }, `Fold ${foldIndex + 1}`),
            React.createElement('div', { style: styles.timeBlockContainer },
              dates.map((date, dateIndex) => {
                let content = null;
                
                if (dateIndex >= fold.trainStart && dateIndex <= fold.trainEnd) {
                  content = React.createElement('div', 
                    {
                      style: {
                        ...styles.trainBlock,
                        width: `${100 / dates.length}%`
                      }
                    },
                    'Train'
                  );
                } else if (dateIndex >= fold.testStart && dateIndex <= fold.testEnd) {
                  content = React.createElement('div', 
                    {
                      style: {
                        ...styles.testBlock,
                        width: `${100 / dates.length}%`
                      }
                    },
                    'Test'
                  );
                } else {
                  content = React.createElement('div', 
                    {
                      style: {
                        ...styles.unusedBlock,
                        width: `${100 / dates.length}%`
                      }
                    }
                  );
                }
                
                return React.createElement('div', 
                  { key: dateIndex, style: {width: `${100 / dates.length}%`} },
                  content
                );
              })
            )
          )
        ),
        
        React.createElement('div', { style: styles.timeBlock },
          React.createElement('div', { style: styles.dateLabel }),
          React.createElement('div', { style: styles.timeBlockContainer },
            dates.map((date, index) => 
              React.createElement('div', 
                { 
                  key: index, 
                  style: {
                    width: `${100 / dates.length}%`,
                    textAlign: 'center',
                    fontSize: '14px',
                    fontWeight: 'bold'
                  }
                },
                date
              )
            )
          )
        )
      ),
      
      React.createElement('div', { style: styles.notes },
        React.createElement('p', { style: styles.p }, React.createElement('strong', null, 'Implementation Details:')),
        React.createElement('ul', null,
          React.createElement('li', { style: styles.li }, 'Each fold uses a fixed window size of 2 days for training (optimal as found in hyperparameter optimization)'),
          React.createElement('li', { style: styles.li }, 'Test window is 1 day immediately following the training period'),
          React.createElement('li', { style: styles.li }, 'Windows "roll forward" in time for each fold'),
          React.createElement('li', { style: styles.li }, 'This approach respects the temporal nature of financial data - prevents future information leakage'),
          React.createElement('li', { style: styles.li }, 'Hyperparameter tuning uses early data, final evaluation uses later data (validation set)'),
          React.createElement('li', { style: styles.li }, 'Models are evaluated on ability to generalize to future, unseen data')
        )
      )
    )
  );
};

module.exports = TimeSeriesCVDiagram;