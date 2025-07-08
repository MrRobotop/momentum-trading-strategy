/**
 * Web Vitals Performance Monitoring
 * =================================
 * 
 * Measures and reports Core Web Vitals for the dashboard.
 * Helps monitor real-world performance metrics.
 */

const reportWebVitals = onPerfEntry => {
  if (onPerfEntry && onPerfEntry instanceof Function) {
    import('web-vitals').then(({ getCLS, getFID, getFCP, getLCP, getTTFB }) => {
      // Core Web Vitals
      getCLS(onPerfEntry);  // Cumulative Layout Shift
      getFID(onPerfEntry);  // First Input Delay
      getFCP(onPerfEntry);  // First Contentful Paint
      getLCP(onPerfEntry);  // Largest Contentful Paint
      getTTFB(onPerfEntry); // Time to First Byte
    }).catch(error => {
      console.log('Web Vitals not available:', error);
    });
  }
};

export default reportWebVitals;