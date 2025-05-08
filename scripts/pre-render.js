// pre-render.js
const React = require('react');
const ReactDOMServer = require('react-dom/server');
const fs = require('fs');
const path = require('path');

// Import all your components (using the paths you provided)
const ModelComparisonDiagram = require('../components/common-js/ModelComparisonDiagram');
const FeatureEngineeringVisual = require('../components/common-js/FeatureEngineeringVisual');
// const FeatureImportanceDiagram = require('.../components/common-js/FeatureImportanceDiagram');
const ModelArchitecture = require('../components/common-js/ModelArchitecture');
const ModelResults = require('../components/common-js/ModelResults');
const TimeSeriesCVDiagram = require('../components/common-js/TimeSeriesCVDiagram');

// Define the output directory
const outputDir = path.join('../components/', 'html-components');

// Create the output directory if it doesn't exist
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

// Define component heights (in pixels) for proper iframe sizing
const componentHeights = {
  'ModelComparisonDiagram': 800,
  'FeatureEngineeringVisual': 1400,
  'FeatureImportanceDiagram': 1000,
  'ModelArchitecture': 1300,
  'ModelResults': 1600,
  'TimeSeriesCVDiagram': 800
};

// Function to render component to HTML and save
function renderComponentToHTML(Component, componentName) {
  console.log(`Rendering ${componentName}...`);
  
  // Get component height (default to 800px if not specified)
  const height = componentHeights[componentName] || 800;
  
  // Generate the HTML string
  const html = ReactDOMServer.renderToStaticMarkup(
    React.createElement(Component)
  );
  
  // Add wrapper with CSS that matches book-theme styling
  const fullHTML = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${componentName}</title>
  <style>
    /* Base styles that match book-theme */
    :root {
      --color-primary: #0066cc;
      --color-secondary: #6c757d;
      --color-background: #ffffff;
      --color-background-secondary: #f5f5f7;
      --color-background-tertiary: #e8f4f8;
      --color-text: #333333;
      --color-text-secondary: #6c757d;
      --color-border: #e0e0e0;
      --color-heading: #333333;
      --font-ui: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
      --font-heading: var(--font-ui);
      --font-text: var(--font-ui);
      --font-mono: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }
    
    html, body {
      margin: 0;
      padding: 0;
      font-family: var(--font-text);
      font-size: 16px;
      line-height: 1.5;
      color: var(--color-text);
      background-color: var(--color-background);
      height: 100%;
      width: 100%;
      overflow: auto;
    }
    
    /* Component container */
    .component-container {
      width: 100%;
      padding: 0.5rem;
      box-sizing: border-box;
    }
    
    /* Table styles */
    table {
      border-collapse: collapse;
      width: 100%;
      margin: 1rem 0;
      font-size: 0.9rem;
    }
    
    th, td {
      padding: 0.75rem;
      border: 1px solid var(--color-border);
      text-align: left;
    }
    
    th {
      background-color: var(--color-background-secondary);
      color: var(--color-text);
      font-weight: 600;
    }
    
    /* Headers and text */
    h1, h2, h3, h4, h5, h6 {
      color: var(--color-heading);
      font-family: var(--font-heading);
      margin-top: 1rem;
      margin-bottom: 0.5rem;
      font-weight: 600;
      line-height: 1.2;
    }
    
    p {
      margin-top: 0;
      margin-bottom: 1rem;
    }
    
    /* Code blocks */
    pre, code {
      font-family: var(--font-mono);
      background-color: var(--color-background-secondary);
      border-radius: 3px;
      font-size: 0.9em;
    }
    
    pre {
      padding: 0.75rem;
      overflow-x: auto;
    }
    
    code {
      padding: 0.2rem 0.4rem;
    }
    
    /* Component-specific sizing */
    .component-full {
      width: 100%;
    }
    
    /* Make sure everything is responsive */
    img, svg {
      max-width: 100%;
      height: auto;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
      html, body {
        font-size: 14px;
      }
      
      table {
        font-size: 0.8rem;
      }
      
      .responsive-scroll {
        overflow-x: auto;
      }
    }
  </style>
  <script>
    // Script to notify parent document of actual height for iframe resizing
    window.addEventListener('load', function() {
      // Get the actual height of the content
      const height = document.body.scrollHeight;
      
      // Send message to parent with height info
      if (window.parent) {
        window.parent.postMessage({ 
          type: 'resize-iframe', 
          height: height,
          componentName: '${componentName}'
        }, '*');
      }
    });
  </script>
</head>
<body>
  <div class="component-container">
    <div class="component-full responsive-scroll">
      ${html}
    </div>
  </div>
</body>
</html>`;

  // Create the full HTML file
  const filePath = path.join(outputDir, `${componentName}.html`);
  fs.writeFileSync(filePath, fullHTML);
  console.log(`✓ Saved ${componentName} to ${filePath}`);
}

// Render each component
renderComponentToHTML(ModelComparisonDiagram, 'ModelComparisonDiagram');
renderComponentToHTML(FeatureEngineeringVisual, 'FeatureEngineeringVisual');
// renderComponentToHTML(FeatureImportanceDiagram, 'FeatureImportanceDiagram');
renderComponentToHTML(ModelArchitecture, 'ModelArchitecture');
renderComponentToHTML(ModelResults, 'ModelResults');
renderComponentToHTML(TimeSeriesCVDiagram, 'TimeSeriesCVDiagram');

console.log('All components rendered successfully!');
