// fix-components.js
const fs = require('fs');
const path = require('path');

// Directory where the components are stored
const componentsDir = path.join(__dirname, '_components');

// Process each HTML file in the components directory
fs.readdirSync(componentsDir)
  .filter(file => file.endsWith('.html'))
  .forEach(file => {
    const filePath = path.join(componentsDir, file);
    console.log(`Processing ${filePath}...`);
    
    let content = fs.readFileSync(filePath, 'utf8');
    
    // 1. Fix asterisks in code and examples
    // Replace *feature_name* patterns with <strong>feature_name</strong>
    content = content.replace(/\*([^*\n]+)\*/g, '<strong>$1</strong>');
    
    // 2. Fix backslashes in code that cause Typst issues
    content = content.replace(/\\(?![a-zA-Z])/g, '\\\\');
    
    // 3. Fix Example:* \ patterns
    content = content.replace(/\*Example:\*\s*\\/g, '<strong>Example:</strong>');
    
    // 4. Ensure all code blocks have a language specified
    content = content.replace(/<pre><code>/g, '<pre><code class="language-python">');
    
    // Write the fixed content back
    fs.writeFileSync(filePath, content);
    console.log(`Fixed ${filePath}`);
  });

console.log('All components fixed!');
