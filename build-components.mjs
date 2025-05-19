// build-components.mjs
import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import path from 'path';
import fs from 'fs';
import { dirname } from 'path';

// Set up dirname equivalent for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Set up require for ES modules
const require = createRequire(import.meta.url);

// Import your components
const FeatureEngineeringVisual = require('./components/common-js/FeatureEngineeringVisual');
const ModelArchitecture = require('./components/common-js/ModelArchitecture');
const FeatureImportanceDiagram = require('./components/common-js/FeatureImportanceDiagram');
const ModelComparisonDiagram = require('./components/common-js/ModelComparisonDiagram');
const ModelResults = require('./components/common-js/ModelResults');
const TimeSeriesCVDiagram = require('./components/common-js/TimeSeriesCVDiagram');

// For React server-side rendering
const React = require('react');
const ReactDOMServer = require('react-dom/server');

// Directory where rendered HTML will be saved
const outputDir = path.join(__dirname, './_components');
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

// Function to render a component
function renderComponent(Component, filename) {
  const html = ReactDOMServer.renderToStaticMarkup(React.createElement(Component));
  fs.writeFileSync(path.join(outputDir, `${filename}.html`), html);
  console.log(`Rendered ${filename}.html`);
}

// Render all components
renderComponent(FeatureEngineeringVisual, 'FeatureEngineeringVisual');
renderComponent(ModelArchitecture, 'ModelArchitecture');
renderComponent(FeatureImportanceDiagram, 'FeatureImportanceDiagram');
renderComponent(ModelComparisonDiagram, 'ModelComparisonDiagram');
renderComponent(ModelResults, 'ModelResults');
renderComponent(TimeSeriesCVDiagram, 'TimeSeriesCVDiagram');

console.log('All components rendered successfully!');
