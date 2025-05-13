// html-autosize.js
document.addEventListener('DOMContentLoaded', function() {
  // Function to handle JS component resizing
  function setupComponentAutosize() {
    // Target all HTML content blocks rendered from JS components
    const componentDivs = document.querySelectorAll('.myst-content div[class*="html"]');
    
    componentDivs.forEach(container => {
      // Make sure tables in components take full width but allow overflow scrolling
      const tables = container.querySelectorAll('table');
      tables.forEach(table => {
        table.style.width = '100%';
        table.style.maxWidth = '100%';
        
        // Create a wrapper if it doesn't exist
        if (table.parentElement.tagName !== 'DIV' || !table.parentElement.classList.contains('table-wrapper')) {
          const wrapper = document.createElement('div');
          wrapper.className = 'table-wrapper';
          wrapper.style.width = '100%';
          wrapper.style.overflow = 'auto';
          table.parentNode.insertBefore(wrapper, table);
          wrapper.appendChild(table);
        }
      });
      
      // Make sure pre and code elements don't break layout
      const codeBlocks = container.querySelectorAll('pre, code');
      codeBlocks.forEach(block => {
        block.style.maxWidth = '100%';
        block.style.whiteSpace = 'pre-wrap';
        block.style.overflowWrap = 'break-word';
        block.style.wordBreak = 'break-word';
      });

      // Handle all component divs to ensure they expand and don't overflow
      container.style.width = '100%';
      container.style.maxWidth = '100%';
      container.style.overflowX = 'hidden';
      container.style.overflowY = 'visible';
      
      // Reset any fixed heights and widths from component styles
      const allDivs = container.querySelectorAll('div');
      allDivs.forEach(div => {
        // Only modify if it has fixed dimensions
        if (div.style.width && div.style.width.endsWith('px')) {
          div.style.maxWidth = '100%';
          div.style.width = '100%';
        }
        
        // Remove or adjust fixed heights
        if (div.style.overflow === 'auto' || div.style.overflow === 'scroll') {
          // Keep vertical scroll if needed
          div.style.overflowX = 'hidden';
        }
      });
    });
  }

  // Initial setup
  setupComponentAutosize();
  
  // Monitor for dynamic content changes (for components loaded after page load)
  const observer = new MutationObserver(function(mutations) {
    setupComponentAutosize();
  });
  
  // Start observing the document with configured parameters
  observer.observe(document.body, { 
    childList: true,
    subtree: true 
  });
});