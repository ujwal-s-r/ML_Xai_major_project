/**
 * Dashboard - Data Visualization
 * Fetches and displays PHQ-8, Game, and Video analysis data
 */

let sessionId = null;
let data = null;
let charts = {};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Dashboard loaded');
    
    // Get session ID from localStorage
    sessionId = localStorage.getItem('session_id');
    if (!sessionId) {
        showError('No session ID found. Please complete the assessment first.');
        return;
    }
    
    document.getElementById('sessionId').textContent = sessionId;
    
    // Fetch data
    await loadData();
});

async function loadData() {
    try {
        const response = await fetch(`/api/results/${sessionId}`);
        
        if (!response.ok) {
            throw new Error(`Failed to load data: ${response.statusText}`);
        }
        
        data = await response.json();
        console.log('Loaded data:', data);
        
        // Hide loading, show dashboard
        document.getElementById('loadingSection').style.display = 'none';
        document.getElementById('dashboardContent').style.display = 'grid';
        
        // Render all sections
        renderPHQ8(data.phq8);
        renderGame(data.game);
        renderVideo(data.video);
        
    } catch (error) {
        console.error('Error loading data:', error);
        showError(error.message);
    }
}

function showError(message) {
    document.getElementById('loadingSection').style.display = 'none';
    document.getElementById('errorSection').style.display = 'block';
    document.getElementById('errorMessage').textContent = message;
}

// ============================================
// PHQ-8 Rendering
// ============================================

function renderPHQ8(phq8Data) {
    if (!phq8Data) {
        document.getElementById('phq8Section').innerHTML = '<div class="no-data">No PHQ-8 data available</div>';
        return;
    }
    
    const score = phq8Data.score || 0;
    const severity = phq8Data.severity || 'Unknown';
    const answers = phq8Data.answers || [];
    
    // Update cards
    document.getElementById('phq8Score').textContent = score;
    document.getElementById('phq8Severity').textContent = severity;
    
    // Color code severity card
    const severityCard = document.getElementById('phq8SeverityCard');
    if (severity === 'None' || severity === 'Mild') {
        severityCard.classList.add('success');
    } else if (severity === 'Moderate') {
        severityCard.classList.add('info');
    } else {
        severityCard.classList.add('warning');
    }
    
    // Create bar chart for individual answers
    const ctx = document.getElementById('phq8Chart').getContext('2d');
    charts.phq8 = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [
                'Q1: Little interest',
                'Q2: Feeling down',
                'Q3: Sleep problems',
                'Q4: Feeling tired',
                'Q5: Appetite changes',
                'Q6: Feeling bad',
                'Q7: Concentration',
                'Q8: Moving/speaking'
            ],
            datasets: [{
                label: 'Response Score',
                data: answers,
                backgroundColor: answers.map(score => {
                    if (score === 0) return '#43e97b';
                    if (score === 1) return '#4facfe';
                    if (score === 2) return '#f093fb';
                    return '#f5576c';
                }),
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const labels = ['Not at all', 'Several days', 'More than half', 'Nearly every day'];
                            return labels[context.parsed.y];
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 3,
                    ticks: {
                        stepSize: 1,
                        callback: function(value) {
                            const labels = ['Not at all', 'Several days', 'More than half', 'Nearly every day'];
                            return labels[value];
                        }
                    }
                }
            }
        }
    });
}

// ============================================
// Game Rendering
// ============================================

function renderGame(gameData) {
    if (!gameData) {
        document.getElementById('gameSection').innerHTML = '<div class="no-data">No game data available</div>';
        return;
    }
    
    const summary = gameData.server_summary || gameData.summary || {};
    
    // Update metrics
    document.getElementById('gameAccuracy').textContent = summary.accuracy ? `${summary.accuracy}%` : '-';
    document.getElementById('gameReactionTime').textContent = summary.avgRt ? `${summary.avgRt}ms` : '-';
    document.getElementById('gameImpulsive').textContent = summary.impulsive || 0;
    document.getElementById('gameErrors').textContent = summary.errors || 0;
    
    // Render distraction cards
    const distractionsContainer = document.getElementById('distractionCards');
    distractionsContainer.innerHTML = '';
    
    if (summary.perDistraction && summary.perDistraction.length > 0) {
        summary.perDistraction.forEach((distraction, index) => {
            const card = document.createElement('div');
            card.className = 'distraction-card';
            
            const pre = distraction.pre || {};
            const post = distraction.post || {};
            
            const preAcc = pre.acc !== null && pre.acc !== undefined ? Math.round(pre.acc * 100) : 'N/A';
            const postAcc = post.acc !== null && post.acc !== undefined ? Math.round(post.acc * 100) : 'N/A';
            const preRt = pre.avgRt || 'N/A';
            const postRt = post.avgRt || 'N/A';
            
            const accChange = (pre.acc !== null && post.acc !== null) ? 
                (post.acc - pre.acc) * 100 : 0;
            const accChangeText = accChange > 0 ? 
                `+${accChange.toFixed(1)}%` : 
                `${accChange.toFixed(1)}%`;
            const accChangeColor = accChange >= 0 ? '#43e97b' : '#f5576c';
            
            card.innerHTML = `
                <div class="distraction-header">
                    <span class="distraction-type">
                        ${distraction.kind === 'visual' ? '👁️ Visual' : 
                          distraction.kind === 'auditory' ? '🔊 Auditory' : 
                          '❓ Other'} Distraction #${index + 1}
                    </span>
                    <span style="color: ${accChangeColor}; font-weight: 600;">
                        ${accChangeText}
                    </span>
                </div>
                <div class="distraction-stats">
                    <div class="stat-item">
                        <span>Pre-accuracy:</span>
                        <strong>${preAcc}${typeof preAcc === 'number' ? '%' : ''}</strong>
                    </div>
                    <div class="stat-item">
                        <span>Post-accuracy:</span>
                        <strong>${postAcc}${typeof postAcc === 'number' ? '%' : ''}</strong>
                    </div>
                    <div class="stat-item">
                        <span>Pre-RT:</span>
                        <strong>${preRt}${typeof preRt === 'number' ? 'ms' : ''}</strong>
                    </div>
                    <div class="stat-item">
                        <span>Post-RT:</span>
                        <strong>${postRt}${typeof postRt === 'number' ? 'ms' : ''}</strong>
                    </div>
                </div>
            `;
            
            distractionsContainer.appendChild(card);
        });
    } else {
        distractionsContainer.innerHTML = '<div class="no-data">No distraction data available</div>';
    }
}

// ============================================
// Video Rendering
// ============================================

function renderVideo(videoData) {
    if (!videoData) {
        document.getElementById('videoSection').innerHTML = '<div class="no-data">No video data available</div>';
        return;
    }
    
    const summary = videoData.server_summary || videoData.client_summary || {};
    const timeline = videoData.timeline || [];
    
    // Update summary metrics
    if (summary.emotion) {
        document.getElementById('videoDominantEmotion').textContent = 
            summary.emotion.dominant_emotion || '-';
        document.getElementById('videoEmotionChanges').textContent = 
            `${summary.emotion.emotion_changes || 0} changes`;
    }
    
    if (summary.blink) {
        document.getElementById('videoBlinkRate').textContent = 
            summary.blink.blink_rate_per_minute || '-';
    }
    
    if (summary.gaze) {
        const attentionScore = summary.gaze.attention_score || 0;
        document.getElementById('videoAttention').textContent = 
            `${Math.round(attentionScore * 100)}%`;
    }
    
    if (summary.pupil) {
        document.getElementById('videoDilations').textContent = 
            summary.pupil.pupil_dilation_events || 0;
    }
    
    // Render time-series charts
    renderEmotionTimeline(timeline);
    renderBlinkTimeline(timeline);
    renderPupilTimeline(timeline);
    renderGazeTimeline(timeline);
    renderEmotionDistribution(summary.emotion);
}

function renderEmotionTimeline(timeline) {
    const emotionData = timeline
        .filter(entry => entry.emotion)
        .map(entry => ({
            x: entry.timestamp,
            emotion: entry.emotion.label,
            confidence: entry.emotion.confidence
        }));
    
    if (emotionData.length === 0) {
        return;
    }
    
    // Map emotions to numeric values for visualization
    const emotionMap = {
        'sad': 1,
        'fear': 2,
        'angry': 3,
        'disgust': 4,
        'surprise': 5,
        'happy': 6,
        'neutral': 7
    };
    
    const ctx = document.getElementById('emotionTimelineChart').getContext('2d');
    charts.emotionTimeline = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Emotion',
                data: emotionData.map(d => ({
                    x: d.x,
                    y: emotionMap[d.emotion] || 0
                })),
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                stepped: true,
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const point = emotionData[context.dataIndex];
                            return `${point.emotion} (${(point.confidence * 100).toFixed(1)}%)`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Time (seconds)'
                    }
                },
                y: {
                    min: 0,
                    max: 8,
                    ticks: {
                        callback: function(value) {
                            const reverseMap = {
                                1: 'Sad',
                                2: 'Fear',
                                3: 'Angry',
                                4: 'Disgust',
                                5: 'Surprise',
                                6: 'Happy',
                                7: 'Neutral'
                            };
                            return reverseMap[value] || '';
                        }
                    }
                }
            }
        }
    });
}

function renderBlinkTimeline(timeline) {
    const blinkData = timeline
        .filter(entry => entry.blink)
        .map(entry => ({
            x: entry.timestamp,
            y: entry.blink.cumulative_blinks || 0
        }));
    
    if (blinkData.length === 0) {
        return;
    }
    
    const ctx = document.getElementById('blinkTimelineChart').getContext('2d');
    charts.blinkTimeline = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Cumulative Blinks',
                data: blinkData,
                borderColor: '#43e97b',
                backgroundColor: 'rgba(67, 233, 123, 0.1)',
                fill: true,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Time (seconds)'
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Total Blinks'
                    }
                }
            }
        }
    });
}

function renderPupilTimeline(timeline) {
    const pupilData = timeline
        .filter(entry => entry.pupil && entry.pupil.avg)
        .map(entry => ({
            x: entry.timestamp,
            y: entry.pupil.avg
        }));
    
    if (pupilData.length === 0) {
        return;
    }
    
    const ctx = document.getElementById('pupilTimelineChart').getContext('2d');
    charts.pupilTimeline = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Pupil Size',
                data: pupilData,
                borderColor: '#f093fb',
                backgroundColor: 'rgba(240, 147, 251, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Time (seconds)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Pupil Size (normalized)'
                    }
                }
            }
        }
    });
}

function renderGazeTimeline(timeline) {
    const gazeData = {
        left: [],
        center: [],
        right: []
    };
    
    timeline
        .filter(entry => entry.gaze)
        .forEach(entry => {
            const direction = entry.gaze.direction;
            gazeData.left.push({
                x: entry.timestamp,
                y: direction === 'left' ? 1 : 0
            });
            gazeData.center.push({
                x: entry.timestamp,
                y: direction === 'center' ? 1 : 0
            });
            gazeData.right.push({
                x: entry.timestamp,
                y: direction === 'right' ? 1 : 0
            });
        });
    
    if (gazeData.left.length === 0) {
        return;
    }
    
    const ctx = document.getElementById('gazeTimelineChart').getContext('2d');
    charts.gazeTimeline = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Left',
                    data: gazeData.left,
                    borderColor: '#f5576c',
                    backgroundColor: 'rgba(245, 87, 108, 0.3)',
                    fill: true,
                    stepped: true
                },
                {
                    label: 'Center',
                    data: gazeData.center,
                    borderColor: '#43e97b',
                    backgroundColor: 'rgba(67, 233, 123, 0.3)',
                    fill: true,
                    stepped: true
                },
                {
                    label: 'Right',
                    data: gazeData.right,
                    borderColor: '#4facfe',
                    backgroundColor: 'rgba(79, 172, 254, 0.3)',
                    fill: true,
                    stepped: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Time (seconds)'
                    }
                },
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Gaze Direction'
                    }
                }
            }
        }
    });
}

function renderEmotionDistribution(emotionSummary) {
    if (!emotionSummary || !emotionSummary.distribution) {
        return;
    }
    
    const distribution = emotionSummary.distribution;
    const emotions = Object.keys(distribution);
    const counts = Object.values(distribution);
    
    const ctx = document.getElementById('emotionDistributionChart').getContext('2d');
    charts.emotionDistribution = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: emotions.map(e => e.charAt(0).toUpperCase() + e.slice(1)),
            datasets: [{
                data: counts,
                backgroundColor: [
                    '#667eea',
                    '#764ba2',
                    '#f093fb',
                    '#f5576c',
                    '#4facfe',
                    '#43e97b',
                    '#38f9d7'
                ]
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
