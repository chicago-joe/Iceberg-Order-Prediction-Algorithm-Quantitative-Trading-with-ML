const fs = require('fs');
const path = require('path');
const express = require('express');
const puppeteer = require('puppeteer');
const React = require('react');
const ReactDOMServer = require('react-dom/server');

// Components
const components = {
  ModelComparisonDiagram: require('../components/common-js/ModelComparisonDiagram'),
  FeatureEngineeringVisual: require('../components/common-js/FeatureEngineeringVisual'),
  // FeatureImportanceDiagram: require('../components/common-js/FeatureImportanceDiagram'),
  ModelArchitecture: require('../components/common-js/ModelArchitecture'),
  ModelResults: require('../components/common-js/ModelResults'),
  TimeSeriesCVDiagram: require('../components/common-js/TimeSeriesCVDiagram'),
};

const componentHeights = {
  ModelComparisonDiagram: 800,
  FeatureEngineeringVisual: 1400,
  FeatureImportanceDiagram: 1000,
  ModelArchitecture: 1300,
  ModelResults: 1600,
  TimeSeriesCVDiagram: 800,
};

const htmlDir = path.resolve(__dirname, '../static-html');
const outputDir = path.resolve(__dirname, '../output');

// Ensure output directories exist
fs.mkdirSync(htmlDir, { recursive: true });
fs.mkdirSync(outputDir, { recursive: true });

// HTML Template
const generateHTML = (componentName, html) => `
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>${componentName}</title></head>
<body>
<div id="root">${html}</div>
</body>
</html>`;

// Step 1: Render HTML files
for (const [name, Component] of Object.entries(components)) {
  const html = ReactDOMServer.renderToStaticMarkup(React.createElement(Component));
  const fullHtml = generateHTML(name, html);
  fs.writeFileSync(path.join(htmlDir, `${name}.html`), fullHtml);
  console.log(`✓ Rendered ${name}.html`);
}

// Step 2: Serve HTML locally
const app = express();
app.use(express.static(htmlDir));
const server = app.listen(3000, async () => {
  console.log('HTTP server running on port 3000');

  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  for (const name of Object.keys(components)) {
    const height = componentHeights[name] || 800;
    await page.setViewport({ width: 1200, height, deviceScaleFactor: 2 });

    const url = `http://localhost:3000/${name}.html`;
    await page.goto(url, { waitUntil: 'networkidle0' });

    const screenshotPath = path.join(outputDir, `${name}.png`);
    await page.screenshot({ path: screenshotPath });
    console.log(`✓ Screenshot saved: ${screenshotPath}`);
  }

  await browser.close();
  server.close();
  console.log('All done!');
});
