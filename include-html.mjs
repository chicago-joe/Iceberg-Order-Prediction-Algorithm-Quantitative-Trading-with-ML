// include-html.mjs
import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { readFileSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const includeHtmlDirective = {
  name: 'include-html',
  doc: 'Include HTML from a file',
  arg: {
    type: String,
    doc: 'The name of the HTML file to include (without .html extension)',
  },
  run(data) {
    const filename = data.arg;
    if (!filename) {
      return [{ type: 'paragraph', children: [{ type: 'text', value: 'Error: Filename required' }] }];
    }
    
    try {
      let htmlContent = readFileSync(join(__dirname, '_components', `${filename}.html`), 'utf8');
      
      // Add wrapper div for component styling and auto-sizing
      htmlContent = `<div class="component-wrapper" data-component="${filename}">
        ${htmlContent}
      </div>`;
      
      return [{ type: 'html', value: htmlContent }];
    } catch (error) {
      return [{ 
        type: 'paragraph', 
        children: [{ type: 'text', value: `Error loading HTML file: ${error.message}` }] 
      }];
    }
  },
};

const plugin = { 
  name: 'HTML File Includer', 
  directives: [includeHtmlDirective] 
};

export default plugin;
