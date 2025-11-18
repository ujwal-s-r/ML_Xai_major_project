/**
 * AI Analysis Page - Frontend Logic
 * Fetches and displays AI-generated mental health assessment report
 */

let sessionId = null;
let reportMarkdown = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    console.log('AI Analysis page loaded');
    
    // Get session ID
    sessionId = localStorage.getItem('session_id');
    if (!sessionId) {
        showError('No session ID found. Please complete the assessment first.');
        return;
    }
    
    // Display session ID
    const sessionElement = document.getElementById('sessionId');
    if (sessionElement) {
        sessionElement.textContent = `Session: ${sessionId}`;
    }
    
    // Generate AI analysis
    await generateAnalysis();
});

async function generateAnalysis() {
    const loadingScreen = document.getElementById('loadingScreen');
    const reportContainer = document.getElementById('reportContainer');
    const errorScreen = document.getElementById('errorScreen');
    
    // Show loading
    loadingScreen.style.display = 'block';
    reportContainer.classList.remove('active');
    errorScreen.classList.remove('active');
    
    try {
        console.log(`Requesting AI analysis for session: ${sessionId}`);
        
        // Create form data
        const formData = new FormData();
        formData.append('session_id', sessionId);
        
        // Call API
        const response = await fetch('/api/ai-analysis/generate', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('AI analysis received');
        
        reportMarkdown = data.report;
        
        // Render markdown
        renderReport(reportMarkdown);
        
        // Show report
        loadingScreen.style.display = 'none';
        reportContainer.classList.add('active');
        
    } catch (error) {
        console.error('Failed to generate AI analysis:', error);
        showError(error.message);
    }
}

function renderReport(markdown) {
    const reportContent = document.getElementById('reportContent');
    
    if (!reportContent) {
        console.error('Report content element not found');
        return;
    }
    
    try {
        // Configure marked for better rendering
        marked.setOptions({
            breaks: true,
            gfm: true,
            headerIds: true,
            mangle: false
        });
        
        // Convert markdown to HTML
        const html = marked.parse(markdown);
        
        // Inject HTML
        reportContent.innerHTML = html;
        
        console.log('Report rendered successfully');
        
    } catch (error) {
        console.error('Error rendering markdown:', error);
        reportContent.innerHTML = `
            <h2>Report Rendering Error</h2>
            <p>The report was generated but could not be displayed properly.</p>
            <pre>${escapeHtml(markdown)}</pre>
        `;
    }
}

function showError(message) {
    const loadingScreen = document.getElementById('loadingScreen');
    const reportContainer = document.getElementById('reportContainer');
    const errorScreen = document.getElementById('errorScreen');
    const errorMessage = document.getElementById('errorMessage');
    
    loadingScreen.style.display = 'none';
    reportContainer.classList.remove('active');
    errorScreen.classList.add('active');
    
    if (errorMessage) {
        errorMessage.textContent = message;
    }
}

function retryAnalysis() {
    generateAnalysis();
}

function downloadReport() {
    if (!reportMarkdown) {
        alert('No report available to download');
        return;
    }
    
    try {
        // Create blob
        const blob = new Blob([reportMarkdown], { type: 'text/markdown' });
        
        // Create download link
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `mental-health-assessment-${sessionId}.md`;
        
        // Trigger download
        document.body.appendChild(a);
        a.click();
        
        // Cleanup
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        console.log('Report downloaded');
        
    } catch (error) {
        console.error('Download failed:', error);
        alert('Failed to download report. Please try again.');
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
