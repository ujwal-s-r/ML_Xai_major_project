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
    
    // Fetch PHQ-8 & Game (existing)
    await loadData();
    // Fetch Video summary + per-frame streams for real charts
    await loadVideoData();
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
        
    // Render PHQ-8 and Game sections
    renderPHQ8(data.phq8);
    renderGame(data.game);
        
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
// Video Rendering (real per-frame data)
// ============================================

let videoSummary = null;
let blinkFrames = [];
let gazeFrames = [];
let pupilFrames = [];
let emotionFrames = [];

async function loadVideoData() {
    try {
        const sumRes = await fetch(`/api/video/summary/${encodeURIComponent(sessionId)}`);
        if (!sumRes.ok) {
            console.warn('No video summary for session');
            return;
        }
        videoSummary = await sumRes.json();

        // Fetch per-frame data for each stage
        const [blinkRes, gazeRes, pupilRes, emotionRes] = await Promise.all([
            fetch(`/api/video/frames/${encodeURIComponent(sessionId)}/blink`).catch(() => null),
            fetch(`/api/video/frames/${encodeURIComponent(sessionId)}/gaze`).catch(() => null),
            fetch(`/api/video/frames/${encodeURIComponent(sessionId)}/pupil`).catch(() => null),
            fetch(`/api/video/frames/${encodeURIComponent(sessionId)}/emotion`).catch(() => null),
        ]);

        blinkFrames = blinkRes && blinkRes.ok ? (await blinkRes.json()).frames : [];
        gazeFrames = gazeRes && gazeRes.ok ? (await gazeRes.json()).frames : [];
        pupilFrames = pupilRes && pupilRes.ok ? (await pupilRes.json()).frames : [];
        emotionFrames = emotionRes && emotionRes.ok ? (await emotionRes.json()).frames : [];

        renderVideoSummary(videoSummary, { blinkFrames, gazeFrames, pupilFrames, emotionFrames });
        renderEmotionTimelineFromFrames(emotionFrames);
        renderBlinkTimelineFromFrames(blinkFrames);
        renderPupilTimelineFromFrames(pupilFrames);
        renderGazeTimelineFromFrames(gazeFrames);
        renderEmotionDistributionFromSummary(videoSummary?.emotion);
    } catch (e) {
        console.warn('Video data load failed', e);
    }
}

function renderVideoSummary(summary, frames) {
    if (!summary) {
        document.getElementById('videoSection').innerHTML = '<div class="no-data">No video data available</div>';
        return;
    }
    // Dominant emotion and changes
    const domCounts = summary?.emotion?.dominant_counts || {};
    const dominantEmotion = Object.entries(domCounts).sort((a,b)=>b[1]-a[1])[0]?.[0] || '-';
    document.getElementById('videoDominantEmotion').textContent = dominantEmotion;
    // Count changes over emotionFrames dominant_emotion
    const domSeq = (frames.emotionFrames || []).map(f => f.dominant_emotion).filter(Boolean);
    let changes = 0; for (let i=1;i<domSeq.length;i++){ if (domSeq[i] !== domSeq[i-1]) changes++; }
    document.getElementById('videoEmotionChanges').textContent = `${changes} changes`;

    // Total blinks from blink summary
    const totalBlinks = summary?.blink?.total_blinks ?? '-';
    document.getElementById('videoTotalBlinks').textContent = totalBlinks;

    // Attention center % from gaze distribution
    const dist = summary?.gaze?.gaze_distribution || {};
    const totalGaze = (dist.LEFT||0)+(dist.CENTER||0)+(dist.RIGHT||0);
    const centerPct = totalGaze ? Math.round((dist.CENTER||0)/totalGaze*100) : 0;
    document.getElementById('videoAttention').textContent = `${centerPct}%`;

    // Avg pupil size from pupil summary
    const avgPupil = summary?.pupil?.avg_pupil_size;
    document.getElementById('videoAvgPupil').textContent = (avgPupil!=null)? avgPupil.toFixed(3): '-';
}

function renderEmotionTimelineFromFrames(frames) {
    if (!frames || !frames.length) return;
    // Build stacked area of probabilities for classes present
    const labels = frames.map(f => f.frame_index);
    // Collect unique emotions across frames
    const emoSet = new Set();
    frames.forEach(f => { const em = f.emotions || {}; Object.keys(em).forEach(k=>emoSet.add(k)); });
    const emotions = Array.from(emoSet);
    const colors = ['#667eea','#764ba2','#f093fb','#f5576c','#4facfe','#43e97b','#38f9d7','#f59e0b','#10b981'];
    const datasets = emotions.map((emo, idx) => ({
        label: emo,
        data: frames.map(f => (f.emotions && f.emotions[emo]!=null) ? Number(f.emotions[emo]) : 0),
        borderColor: colors[idx % colors.length],
        backgroundColor: colors[idx % colors.length] + '55',
        fill: true,
        tension: 0.1,
        stack: 'emotions'
    }));

    const ctx = document.getElementById('emotionTimelineChart').getContext('2d');
    charts.emotionTimeline = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top' } },
            scales: {
                x: { title: { display: true, text: 'Frame' } },
                y: { title: { display: true, text: 'Emotion probability (%)' }, stacked: true, min: 0, max: 100 }
            }
        }
    });
}

function renderBlinkTimelineFromFrames(frames) {
    if (!frames || !frames.length) return;
    const labels = frames.map(f => f.frame_index);
    const ear = frames.map(f => Number(f.avg_ear ?? 0));
    // Blink events when blink_count increments
    const events = [];
    for (let i=1;i<frames.length;i++){
        const prev = frames[i-1].blink_count||0; const curr = frames[i].blink_count||0;
        if (curr>prev) events.push({x: frames[i].frame_index, y: Math.max(...ear)*1.05});
    }
    const ctx = document.getElementById('blinkTimelineChart').getContext('2d');
    charts.blinkTimeline = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                { label: 'EAR', data: ear, borderColor: '#43e97b', backgroundColor: 'rgba(67,233,123,0.1)', fill: true, tension: 0.2 },
                { label: 'Blink', data: events, type: 'scatter', pointBackgroundColor: '#f5576c', pointRadius: 4, showLine: false }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: true } },
            scales: { x: { title: { display: true, text: 'Frame' } }, y: { title: { display: true, text: 'EAR' } } }
        }
    });
}

function renderPupilTimelineFromFrames(frames) {
    if (!frames || !frames.length) return;
    const labels = frames.map(f => f.frame_index);
    const sizes = frames.map(f => Number(f.avg_pupil_size ?? 0));
    const ctx = document.getElementById('pupilTimelineChart').getContext('2d');
    charts.pupilTimeline = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets: [{ label: 'Pupil Size', data: sizes, borderColor: '#f093fb', backgroundColor: 'rgba(240,147,251,0.1)', fill: true, tension: 0.3 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'Frame' } }, y: { title: { display: true, text: 'Pupil Size (normalized)' } } } }
    });
}

function renderGazeTimelineFromFrames(frames) {
    if (!frames || !frames.length) return;
    const labels = frames.map(f => f.frame_index);
    const left = frames.map(f => (f.label==='LEFT'?1:0));
    const center = frames.map(f => (f.label==='CENTER'?1:0));
    const right = frames.map(f => (f.label==='RIGHT'?1:0));
    const ctx = document.getElementById('gazeTimelineChart').getContext('2d');
    charts.gazeTimeline = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets: [
            { label: 'Left', data: left, borderColor:'#f5576c', backgroundColor:'rgba(245,87,108,0.3)', fill:true, stepped:true },
            { label: 'Center', data: center, borderColor:'#43e97b', backgroundColor:'rgba(67,233,123,0.3)', fill:true, stepped:true },
            { label: 'Right', data: right, borderColor:'#4facfe', backgroundColor:'rgba(79,172,254,0.3)', fill:true, stepped:true }
        ]},
        options: { responsive:true, maintainAspectRatio:false, plugins:{ legend:{ display:true } }, scales:{ x:{ title:{ display:true, text:'Frame' } }, y:{ beginAtZero:true, max:1, ticks:{ display:false }, title:{ display:true, text:'Gaze Direction' } } } }
    });
}

function renderEmotionDistributionFromSummary(emotionSummary) {
    if (!emotionSummary || !emotionSummary.dominant_counts) return;
    const distribution = emotionSummary.dominant_counts;
    const emotions = Object.keys(distribution);
    const counts = Object.values(distribution);
    const ctx = document.getElementById('emotionDistributionChart').getContext('2d');
    charts.emotionDistribution = new Chart(ctx, {
        type: 'doughnut',
        data: { labels: emotions.map(cap), datasets: [{ data: counts, backgroundColor: ['#667eea','#764ba2','#f093fb','#f5576c','#4facfe','#43e97b','#38f9d7'] }] },
        options: { responsive:true, maintainAspectRatio:false, plugins:{ legend:{ position:'right' } } }
    });
}

function cap(e){ return e.charAt(0).toUpperCase()+e.slice(1); }
