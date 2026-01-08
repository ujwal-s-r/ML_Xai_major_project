/**
 * Dashboard - Visualization of Video Analysis Results
 * Displays continuous timeline data with charts and XAI visualizations
 */

let sessionId = null;
let analysisData = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Dashboard loaded');
    
    // Get session ID
    sessionId = localStorage.getItem('session_id');
    if (!sessionId) {
        showError('No session ID found');
        return;
    }
    
    const sessionElement = document.getElementById('sessionId');
    if (sessionElement) {
        sessionElement.textContent = sessionId;
    }
    
    // Load analysis data
    await loadAnalysisData();
});

async function loadAnalysisData() {
    try {
        console.log('[Dashboard] Fetching data for session:', sessionId);
        const response = await fetch(`/api/video/results/${sessionId}`);
        
        console.log('[Dashboard] Response status:', response.status);
        if (!response.ok) {
            const errorText = await response.text();
            console.error('[Dashboard] Response error:', errorText);
            throw new Error(`Failed to load analysis data: ${response.status}`);
        }
        
        const rawData = await response.json();
        console.log('[Dashboard] Raw data loaded:', rawData);
        console.log('[Dashboard] Has timeline?', !!rawData.timeline);
        console.log('[Dashboard] Has summary?', !!rawData.summary);
        console.log('[Dashboard] Timeline length:', rawData.timeline?.length);
        console.log('[Dashboard] Summary object:', JSON.stringify(rawData.summary));
        console.log('[Dashboard] Summary keys:', rawData.summary ? Object.keys(rawData.summary) : 'NO SUMMARY');
        
        // Check if we have timeline and summary directly (from /api/video/results endpoint)
        if (rawData.timeline && rawData.summary && Object.keys(rawData.summary).length > 0) {
            console.log('[Dashboard] Using direct structure');
            analysisData = {
                timeline: rawData.timeline || [],
                summary: rawData.summary || {}
            };
        }
        // Otherwise check for nested video structure (from /api/results endpoint)
        else if (rawData.video) {
            console.log('[Dashboard] Using nested video structure');
            analysisData = {
                timeline: rawData.video.timeline || [],
                summary: rawData.video.server_summary || {}
            };
        }
        else {
            console.error('[Dashboard] Data structure not recognized');
            console.error('[Dashboard] rawData keys:', Object.keys(rawData));
            console.error('[Dashboard] summary is empty?', rawData.summary && Object.keys(rawData.summary).length === 0);
            throw new Error('No video analysis data found in response');
        }
        
        console.log('[Dashboard] Analysis data extracted successfully');
        console.log('[Dashboard] Timeline entries:', analysisData.timeline.length);
        console.log('[Dashboard] Summary keys:', Object.keys(analysisData.summary));
        
        // Hide loading, show dashboard
        const loadingState = document.getElementById('loadingState');
        const dashboardContent = document.getElementById('dashboardContent');
        
        if (loadingState) {
            loadingState.style.display = 'none';
        }
        if (dashboardContent) {
            dashboardContent.style.display = 'block';
        }
        
        // Populate dashboard
        populateSummary();
        createCharts();
        
        // Show XAI if available
        if (analysisData.summary.emotion && analysisData.summary.emotion.xai_available) {
            showXAIVisualizations();
        }
        
    } catch (error) {
        console.error('Error loading data:', error);
        showError('Could not load analysis data. Please try the analysis again.');
    }
}

function showError(message) {
    const loadingState = document.getElementById('loadingState');
    const errorState = document.getElementById('errorState');
    
    if (loadingState) {
        loadingState.style.display = 'none';
    }
    if (errorState) {
        errorState.style.display = 'block';
        const errorText = errorState.querySelector('p');
        if (errorText) {
            errorText.textContent = message;
        }
    }
}

function populateSummary() {
    console.log('populateSummary called with:', analysisData);
    
    if (!analysisData || !analysisData.summary) {
        console.error('No analysisData or summary available');
        return;
    }
    
    const { summary } = analysisData;
    console.log('Summary object:', summary);
    
    // Blinks
    if (summary.blink) {
        const totalBlinksEl = document.getElementById('totalBlinks');
        const blinkRateEl = document.getElementById('blinkRate');
        if (totalBlinksEl) totalBlinksEl.textContent = summary.blink.total_blinks || 0;
        if (blinkRateEl) blinkRateEl.textContent = `${(summary.blink.blink_rate_per_minute || 0).toFixed(1)} blinks/min`;
    }
    
    // Emotion
    if (summary.emotion) {
        const dominantEmotionEl = document.getElementById('dominantEmotion');
        const detectionRateEl = document.getElementById('detectionRate');
        if (dominantEmotionEl) {
            dominantEmotionEl.textContent = 
                summary.emotion.dominant_emotion.charAt(0).toUpperCase() + summary.emotion.dominant_emotion.slice(1);
        }
        if (detectionRateEl) {
            detectionRateEl.textContent = 
                `${(summary.emotion.detection_rate || 0).toFixed(1)}% detection rate`;
        }
    }
    
    // Gaze
    if (summary.gaze) {
        const dominantGazeEl = document.getElementById('dominantGaze');
        const attentionScoreEl = document.getElementById('attentionScore');
        if (dominantGazeEl) {
            dominantGazeEl.textContent = 
                summary.gaze.dominant_direction.charAt(0).toUpperCase() + summary.gaze.dominant_direction.slice(1);
        }
        if (attentionScoreEl) {
            attentionScoreEl.textContent = 
                `${((summary.gaze.attention_score || 0) * 100).toFixed(0)}% attention`;
        }
    }
    
    // Pupil
    if (summary.pupil) {
        const avgPupilSizeEl = document.getElementById('avgPupilSize');
        const dilationEventsEl = document.getElementById('dilationEvents');
        if (avgPupilSizeEl) avgPupilSizeEl.textContent = (summary.pupil.avg_pupil_size || 0).toFixed(2);
        if (dilationEventsEl) {
            dilationEventsEl.textContent = 
                `${summary.pupil.pupil_dilation_events || 0} dilation events`;
        }
    }
    
    console.log('populateSummary completed successfully');
}

function createCharts() {
    console.log('createCharts called');
    
    if (!analysisData || !analysisData.timeline || analysisData.timeline.length === 0) {
        console.error('No timeline data available for charts');
        return;
    }
    
    const { timeline } = analysisData;
    console.log('Timeline length:', timeline.length);
    
    // Extract timestamps
    const timestamps = timeline.map(entry => entry.t.toFixed(1));
    
    // 1. Blink Activity Chart
    createBlinkChart(timestamps, timeline);
    
    // 2. Emotion Timeline Chart
    createEmotionChart(timestamps, timeline);
    
    // 3. Gaze Direction Chart
    createGazeChart(timestamps, timeline);
    
    // 4. Pupil Dilation Chart
    createPupilChart(timestamps, timeline);
    
    // 5. Emotion Distribution Pie Chart
    createEmotionDistributionChart();
    
    console.log('All charts created successfully');
}

function createBlinkChart(timestamps, timeline) {
    const ctx = document.getElementById('blinkChart');
    if (!ctx) return;
    
    // Extract EAR values
    const earData = timeline.map(entry => entry.blink?.avg_ear || null);
    const blinkEvents = timeline.map(entry => entry.blink?.is_blinking ? 1 : 0);
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: [
                {
                    label: 'Eye Aspect Ratio (EAR)',
                    data: earData,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Blink Events',
                    data: blinkEvents,
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.2)',
                    type: 'bar',
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            scales: {
                x: {
                    title: { display: true, text: 'Time (seconds)' }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: { display: true, text: 'EAR Value' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: { display: true, text: 'Blink' },
                    grid: { drawOnChartArea: false },
                    max: 1
                }
            }
        }
    });
}

function createEmotionChart(timestamps, timeline) {
    const ctx = document.getElementById('emotionChart');
    if (!ctx) return;
    
    // Map emotions to numeric codes for visualization
    const emotionMap = {
        'sad': 0, 'disgust': 1, 'angry': 2, 'neutral': 3, 
        'fear': 4, 'surprise': 5, 'happy': 6
    };
    
    const emotionData = timeline.map(entry => {
        const emotion = entry.emotion?.dominant_emotion;
        return emotion ? emotionMap[emotion] : null;
    });
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: [{
                label: 'Emotion',
                data: emotionData,
                borderColor: '#764ba2',
                backgroundColor: 'rgba(118, 75, 162, 0.1)',
                stepped: true,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'Time (seconds)' }
                },
                y: {
                    title: { display: true, text: 'Emotion' },
                    ticks: {
                        callback: function(value) {
                            const emotions = ['Sad', 'Disgust', 'Angry', 'Neutral', 'Fear', 'Surprise', 'Happy'];
                            return emotions[value] || '';
                        }
                    },
                    min: 0,
                    max: 6
                }
            }
        }
    });
}

function createGazeChart(timestamps, timeline) {
    const ctx = document.getElementById('gazeChart');
    if (!ctx) return;
    
    // Map gaze directions to numeric values
    const gazeMap = { 'left': -1, 'center': 0, 'right': 1 };
    const gazeData = timeline.map(entry => {
        const direction = entry.gaze?.gaze_direction;
        return direction ? gazeMap[direction] : 0;
    });
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: [{
                label: 'Gaze Direction',
                data: gazeData,
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                stepped: true,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'Time (seconds)' }
                },
                y: {
                    title: { display: true, text: 'Direction' },
                    ticks: {
                        callback: function(value) {
                            if (value === -1) return 'Left';
                            if (value === 0) return 'Center';
                            if (value === 1) return 'Right';
                            return '';
                        }
                    },
                    min: -1,
                    max: 1
                }
            }
        }
    });
}

function createPupilChart(timestamps, timeline) {
    const ctx = document.getElementById('pupilChart');
    if (!ctx) return;
    
    const pupilData = timeline.map(entry => entry.pupil?.avg_pupil_size || null);
    const dilationRatio = timeline.map(entry => entry.pupil?.pupil_dilation_ratio || null);
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: [
                {
                    label: 'Avg Pupil Size (px)',
                    data: pupilData,
                    borderColor: '#2ecc71',
                    backgroundColor: 'rgba(46, 204, 113, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Dilation Ratio',
                    data: dilationRatio,
                    borderColor: '#f39c12',
                    backgroundColor: 'rgba(243, 156, 18, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            scales: {
                x: {
                    title: { display: true, text: 'Time (seconds)' }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: { display: true, text: 'Size (pixels)' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: { display: true, text: 'Ratio' },
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });
}

function createEmotionDistributionChart() {
    const ctx = document.getElementById('emotionDistChart');
    if (!ctx) return;
    
    const distribution = analysisData.summary.emotion?.distribution || {};
    const labels = Object.keys(distribution).map(e => 
        e.charAt(0).toUpperCase() + e.slice(1)
    );
    const data = Object.values(distribution);
    
    const colors = [
        '#3498db', // sad
        '#9b59b6', // disgust
        '#e74c3c', // angry
        '#95a5a6', // neutral
        '#f39c12', // fear
        '#f1c40f', // surprise
        '#2ecc71'  // happy
    ];
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors,
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right'
                }
            }
        }
    });
}

// XAI visualization mode
let currentXAIMode = 'attention';

function showXAIVisualizations() {
    const xaiSection = document.getElementById('xaiSection');
    const xaiGrid = document.getElementById('xaiGrid');
    
    const xaiFrames = analysisData.summary.emotion?.xai_frames || [];
    
    if (xaiFrames.length === 0) {
        console.log('[XAI] No XAI frames available');
        return;
    }
    
    console.log('[XAI] Rendering', xaiFrames.length, 'XAI frames');
    xaiSection.style.display = 'block';
    xaiGrid.innerHTML = ''; // Clear existing content
    
    // Create XAI frame cards
    xaiFrames.forEach((frameIdx, index) => {
        const entry = analysisData.timeline[frameIdx];
        if (!entry || !entry.emotion?.has_xai) {
            console.log('[XAI] Skipping frame', frameIdx, '- no XAI data');
            return;
        }
        
        const frameDiv = document.createElement('div');
        frameDiv.className = 'xai-frame';
        frameDiv.id = `xai-frame-${frameIdx}`;
        
        // Get position label
        const positionLabels = ['Early (25%)', 'Mid (50%)', 'Late (75%)', 'Peak Emotion'];
        const positionLabel = positionLabels[index] || `Frame ${index + 1}`;
        
        frameDiv.innerHTML = `
            <div class="xai-frame-header">
                ${positionLabel} - Frame #${frameIdx}
            </div>
            <div class="xai-frame-content">
                <div class="xai-canvas-container">
                    <canvas id="xai-canvas-${frameIdx}" width="150" height="150"></canvas>
                </div>
                <div class="xai-info">
                    <div class="xai-info-row">
                        <span class="xai-info-label">Time</span>
                        <span class="xai-info-value">${entry.t.toFixed(2)}s</span>
                    </div>
                    <div class="xai-info-row">
                        <span class="xai-info-label">Emotion</span>
                        <span class="xai-info-value">${capitalizeFirst(entry.emotion.dominant_emotion)}</span>
                    </div>
                    <div class="xai-info-row">
                        <span class="xai-info-label">Confidence</span>
                        <span class="xai-info-value">${(entry.emotion.emotions[entry.emotion.dominant_emotion] || 0).toFixed(1)}%</span>
                    </div>
                    <div class="xai-info-row">
                        <span class="xai-info-label">Grid Size</span>
                        <span class="xai-info-value">${entry.attention_grid_size || 14}×${entry.attention_grid_size || 14}</span>
                    </div>
                </div>
            </div>
        `;
        
        xaiGrid.appendChild(frameDiv);
        
        // Render the heatmap on canvas after adding to DOM
        setTimeout(() => {
            renderXAICanvas(frameIdx, entry, currentXAIMode);
        }, 50);
    });
}

function capitalizeFirst(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function setXAIMode(mode) {
    currentXAIMode = mode;
    
    // Update button states
    document.getElementById('xaiModeAttention').classList.toggle('active', mode === 'attention');
    document.getElementById('xaiModeGradcam').classList.toggle('active', mode === 'gradcam');
    
    // Re-render all XAI canvases
    const xaiFrames = analysisData.summary.emotion?.xai_frames || [];
    xaiFrames.forEach(frameIdx => {
        const entry = analysisData.timeline[frameIdx];
        if (entry && entry.emotion?.has_xai) {
            renderXAICanvas(frameIdx, entry, mode);
        }
    });
}

function renderXAICanvas(frameIdx, entry, mode) {
    const canvas = document.getElementById(`xai-canvas-${frameIdx}`);
    if (!canvas) {
        console.log('[XAI] Canvas not found for frame', frameIdx);
        return;
    }
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, width, height);
    
    // Get heatmap data based on mode
    let heatmapData = null;
    if (mode === 'attention' && entry.attention_map) {
        heatmapData = entry.attention_map;
    } else if (mode === 'gradcam' && entry.gradcam_heatmap) {
        heatmapData = entry.gradcam_heatmap;
    }
    
    // Check for XAI data in emotion.xai object (video_pipeline format)
    if (!heatmapData && entry.emotion?.xai) {
        if (mode === 'attention' && entry.emotion.xai.attention_map) {
            heatmapData = entry.emotion.xai.attention_map;
        } else if (mode === 'gradcam' && entry.emotion.xai.gradcam_heatmap) {
            heatmapData = entry.emotion.xai.gradcam_heatmap;
        }
    }
    
    if (!heatmapData) {
        // Draw placeholder
        ctx.fillStyle = '#ddd';
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = '#999';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`No ${mode} data`, width / 2, height / 2);
        return;
    }
    
    // Get face image if available
    const faceBase64 = entry.face_image_base64;
    
    if (faceBase64) {
        // Load face image first, then overlay heatmap
        const img = new Image();
        img.onload = () => {
            // Draw face image
            ctx.drawImage(img, 0, 0, width, height);
            
            // Overlay heatmap with transparency
            overlayHeatmap(ctx, heatmapData, width, height, 0.5);
        };
        img.onerror = () => {
            console.log('[XAI] Failed to load face image for frame', frameIdx);
            // Just draw heatmap without face
            overlayHeatmap(ctx, heatmapData, width, height, 1.0);
        };
        img.src = `data:image/jpeg;base64,${faceBase64}`;
    } else {
        // No face image, just draw heatmap
        overlayHeatmap(ctx, heatmapData, width, height, 1.0);
    }
}

function overlayHeatmap(ctx, heatmapData, width, height, alpha) {
    // heatmapData is a 2D array (e.g., 14x14 or 150x150)
    const gridSize = heatmapData.length;
    const cellWidth = width / gridSize;
    const cellHeight = height / gridSize;
    
    // Create temporary canvas for heatmap
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Draw heatmap cells
    for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
            const value = heatmapData[i][j];
            const color = valueToJetColor(value);
            tempCtx.fillStyle = color;
            tempCtx.fillRect(j * cellWidth, i * cellHeight, cellWidth + 1, cellHeight + 1);
        }
    }
    
    // Apply slight blur for smoother visualization
    if (gridSize < 30) {
        tempCtx.filter = 'blur(2px)';
        tempCtx.drawImage(tempCanvas, 0, 0);
        tempCtx.filter = 'none';
    }
    
    // Draw heatmap with alpha blending
    ctx.globalAlpha = alpha;
    ctx.drawImage(tempCanvas, 0, 0);
    ctx.globalAlpha = 1.0;
}

function valueToJetColor(value) {
    // Jet colormap: blue -> cyan -> green -> yellow -> red
    // Value should be between 0 and 1
    const v = Math.max(0, Math.min(1, value));
    
    let r, g, b;
    
    if (v < 0.25) {
        // Blue to Cyan
        const t = v / 0.25;
        r = 0;
        g = Math.round(255 * t);
        b = 255;
    } else if (v < 0.5) {
        // Cyan to Green
        const t = (v - 0.25) / 0.25;
        r = 0;
        g = 255;
        b = Math.round(255 * (1 - t));
    } else if (v < 0.75) {
        // Green to Yellow
        const t = (v - 0.5) / 0.25;
        r = Math.round(255 * t);
        g = 255;
        b = 0;
    } else {
        // Yellow to Red
        const t = (v - 0.75) / 0.25;
        r = 255;
        g = Math.round(255 * (1 - t));
        b = 0;
    }
    
    return `rgb(${r}, ${g}, ${b})`;
}

function showXAIDetail(frameIdx, entry) {
    // Could implement a modal for detailed view in the future
    console.log('[XAI] Detail view for frame', frameIdx, entry);
}
