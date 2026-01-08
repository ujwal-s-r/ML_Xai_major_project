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

function showXAIVisualizations() {
    const xaiSection = document.getElementById('xaiSection');
    const xaiGrid = document.getElementById('xaiGrid');
    
    const xaiFrames = analysisData.summary.emotion.xai_frames || [];
    
    if (xaiFrames.length === 0) {
        return;
    }
    
    xaiSection.style.display = 'block';
    
    // Create XAI frame thumbnails
    xaiFrames.forEach(frameIdx => {
        const entry = analysisData.timeline[frameIdx];
        if (!entry || !entry.emotion?.has_xai) return;
        
        const frameDiv = document.createElement('div');
        frameDiv.className = 'xai-frame';
        frameDiv.innerHTML = `
            <div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; background: #f8f9fa;">
                <div style="text-align: center; padding: 1rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔍</div>
                    <div style="font-size: 0.9rem; color: #666;">Frame ${frameIdx}</div>
                    <div style="font-size: 0.8rem; color: #999; margin-top: 0.5rem;">
                        ${entry.emotion.dominant_emotion}
                    </div>
                </div>
            </div>
            <div class="frame-label">
                t=${entry.t.toFixed(1)}s | ${entry.emotion.dominant_emotion}
            </div>
        `;
        
        frameDiv.onclick = () => showXAIDetail(frameIdx, entry);
        xaiGrid.appendChild(frameDiv);
    });
}

function showXAIDetail(frameIdx, entry) {
    alert(`XAI Detail for Frame ${frameIdx}\n\nEmotion: ${entry.emotion.dominant_emotion}\nTime: ${entry.t.toFixed(1)}s\n\nAttention Map: ${entry.emotion.xai?.attention_map ? 'Available' : 'N/A'}\nGrad-CAM: ${entry.emotion.xai?.gradcam_heatmap ? 'Available' : 'N/A'}\n\n(Detailed XAI viewer coming soon!)`);
}
