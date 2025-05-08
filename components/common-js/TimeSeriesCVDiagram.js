const React = require('react');

const TimeSeriesCVDiagram = () => {
  const styles = {
    container: {
      fontFamily: 'Arial, sans-serif',
      padding: '20px',
      maxWidth: '100%'
    },
    header: {
      marginBottom: '20px'
    },
    visualContainer: {
      border: '1px solid #ddd',
      borderRadius: '8px',
      padding: '20px',
      marginBottom: '30px',
      backgroundColor: '#fafafa'
    },
    visualization: {
      display: 'flex',
      flexDirection: 'column',
      gap: '15px',
      marginTop: '20px'
    },
    timeBlock: {
      display: 'flex',
      alignItems: 'center',
      height: '40px'
    },
    dateLabel: {
      width: '100px',
      textAlign: 'right',
      paddingRight: '10px',
      fontSize: '12px',
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
      color: '#fff',
      fontSize: '12px',
      fontWeight: 'bold'
    },
    testBlock: {
      height: '100%',
      backgroundColor: '#e2984a',
      borderRadius: '4px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#fff',
      fontSize: '12px',
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
      gap: '20px',
      marginBottom: '20px'
    },
    legendItem: {
      display: 'flex',
      alignItems: 'center',
      fontSize: '14px'
    },
    legendColor: {
      width: '20px',
      height: '20px',
      marginRight: '8px',
      borderRadius: '4px'
    },
    notes: {
      fontSize: '14px',
      lineHeight: '1.5',
      marginTop: '20px',
      padding: '15px',
      backgroundColor: '#f5f5f5',
      borderRadius: '5px',
      border: '1px solid #eee'
    }
  };

  // Sample dates for visualization
  const dates = [
    '2023-07-03', '2023-07-10', '2023-07-17', '2023-07-24',
    '2023-07-31', '2023-08-07', '2023-08-14', '2023-08-21'
  ];

  // Time series CV fold configurations
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
      React.createElement('h2', null, 'Time Series Cross-Validation Approach'),
      React.createElement('p', null, 'The project uses a rolling window approach to handle the time-dependent nature of financial data')
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
      React.createElement('h3', null, 'Rolling Window Cross-Validation'),
      
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
                    fontSize: '11px'
                  }
                },
                date
              )
            )
          )
        )
      ),
      
      React.createElement('div', { style: styles.notes },
        React.createElement('p', null, React.createElement('strong', null, 'Implementation Details:')),
        React.createElement('ul', null,
          React.createElement('li', null, 'Each fold uses a fixed window size for training (typically 2 days in the project)'),
          React.createElement('li', null, 'Test window is 1 day immediately following the training period'),
          React.createElement('li', null, 'Windows "roll forward" in time for each fold'),
          React.createElement('li', null, 'This approach respects the temporal nature of financial data - prevents future information leakage'),
          React.createElement('li', null, 'Hyperparameter tuning uses early data, final evaluation uses later data (validation set)'),
          React.createElement('li', null, 'Models are evaluated on ability to generalize to future, unseen data')
        ),
        React.createElement('p', null, 'The code implements this approach using the function ', 
          React.createElement('code', null, '_create_time_series_splits'), 
          ' which generates train/test splits with proper date separation:'
        ),
        React.createElement('pre', null,
          'def _create_time_series_splits(self, train_size, dates):\n' +
          '    splits = []\n' +
          '    n = len(dates)\n\n' +
          '    for i in range(n):\n' +
          '        if i + train_size < n:\n' +
          '            train_dates = dates[i:i + train_size]\n' +
          '            test_dates = [dates[i + train_size]]\n' +
          '            splits.append((train_dates, test_dates))\n\n' +
          '    return splits'
        )
      )
    )
  );
};

module.exports = TimeSeriesCVDiagram;
