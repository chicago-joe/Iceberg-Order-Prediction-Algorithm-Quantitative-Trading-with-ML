// iframe-resizer.js
document.addEventListener('DOMContentLoaded', function() {
  // Find all component iframes
  const iframes = document.querySelectorAll('.component-iframe');
  
  // Initialize iframe heights
  iframes.forEach(iframe => {
    // Set initial height based on data-height attribute
    const initialHeight = iframe.getAttribute('data-height') || '800';
    iframe.style.height = `${initialHeight}px`;
  });
  
  // Listen for messages from iframes
  window.addEventListener('message', function(e) {
    const message = e.data;
    
    // Check if it's a resize message
    if (message && message.type === 'resize-iframe') {
      // Find the iframe for this component
      const iframe = document.querySelector(`.component-iframe[data-component="${message.componentName}"]`);
      
      if (iframe) {
        // Add a little extra height to avoid scrollbars (20px padding)
        iframe.style.height = `${message.height + 20}px`;
      }
    }
  });
});
