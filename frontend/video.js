/**
 * Video Analysis - Batch Processing Mode
 * 1. Captures all webcam frames while video plays
 * 2. Sends batch to backend for sequential processing
 * 3. Displays results after processing complete
 */

// Global state
let stream = null;
let videoElement = null;
let canvasElement = null;
let ctx = null;
let sessionId = null;
let isCapturing = false;

// Frame storage
let capturedFrames = [];
const FPS = 30;

// Results storage
let timeline = [];
let summary = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('Video analysis page loaded (BATCH MODE)');
    
    videoElement = document.getElementById('triggerVideo');
    canvasElement = document.getElementById('frameCanvas');
    ctx = canvasElement.getContext('2d');
    
    // Get session ID from localStorage
    sessionId = localStorage.getItem('session_id');
    if (!sessionId) {
        alert('No session ID found. Please complete the questionnaire first.');
        window.location.href = '/';
        return;
    }
    
    // Set up start button
    const startBtn = document.getElementById('startBtn');
    startBtn.addEventListener('click', startAnalysis);
});

async function startAnalysis() {
    console.log('Starting video analysis (BATCH MODE)...');
    
    const startBtn = document.getElementById('startBtn');
    const startOverlay = document.getElementById('startOverlay');
    const progressSection = document.getElementById('progressSection');
    
    startBtn.disabled = true;
    startBtn.textContent = 'Requesting camera access...';
    
    try {
        // Request webcam access
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: FPS }
            }
        });
        
        console.log('Webcam access granted');
        
        // Set canvas size
        canvasElement.width = 640;
        canvasElement.height = 480;
        
        // Hide overlay, show progress
        startOverlay.classList.add('hidden');
        progressSection.classList.remove('hidden');
        
        // Update progress text
        document.getElementById('progressText').textContent = 'Capturing frames from webcam...';
        document.getElementById('framesProcessed').textContent = '0';
        document.getElementById('blinksDetected').textContent = '-';
        document.getElementById('currentEmotion').textContent = 'Capturing...';
        document.getElementById('gazeDirection').textContent = '-';
        
        // Start video
        videoElement.play();
        
        // Start frame capture loop
        isCapturing = true;
        capturedFrames = [];
        captureFrameLoop();
        
        // Handle video end
        videoElement.addEventListener('ended', onVideoEnded, { once: true });
        
    } catch (error) {
        console.error('Failed to start analysis:', error);
        alert('Could not start analysis. Please check camera permissions.');
        startBtn.disabled = false;
        startBtn.textContent = 'Start Analysis';
    }
}

async function captureFrameLoop() {
    if (!isCapturing || !stream) return;
    
    try {
        // Capture frame from webcam
        const videoTrack = stream.getVideoTracks()[0];
        const imageCapture = new ImageCapture(videoTrack);
        const blob = await imageCapture.takePhoto();
        
        // Draw to canvas
        const img = await createImageBitmap(blob);
        ctx.drawImage(img, 0, 0, canvasElement.width, canvasElement.height);
        
        // Get base64 image and store it
        const base64Image = canvasElement.toDataURL('image/jpeg', 0.8);
        capturedFrames.push(base64Image);
        
        // Update progress
        const estimatedTotalFrames = videoElement.duration * FPS;
        const captureProgress = Math.min((capturedFrames.length / estimatedTotalFrames) * 50, 50);
        document.getElementById('progressBar').style.width = `${captureProgress}%`;
        document.getElementById('progressText').textContent = `Captured ${capturedFrames.length} frames...`;
        document.getElementById('framesProcessed').textContent = capturedFrames.length;
        
        // Continue loop if video is still playing
        if (!videoElement.ended && isCapturing) {
            setTimeout(captureFrameLoop, 1000 / FPS);
        }
        
    } catch (error) {
        console.error('Error in frame capture loop:', error);
        // Continue anyway
        if (!videoElement.ended && isCapturing) {
            setTimeout(captureFrameLoop, 1000 / FPS);
        }
    }
}

async function onVideoEnded() {
    console.log('Video ended, processing frames...');
    isCapturing = false;
    
    // Stop webcam
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    
    console.log(`Total frames captured: ${capturedFrames.length}`);
    
    // Update UI
    document.getElementById('progressBar').style.width = '50%';
    document.getElementById('progressText').textContent = `Processing ${capturedFrames.length} frames...`;
    document.getElementById('currentEmotion').textContent = 'Processing...';
    
    // Send batch to backend for processing
    try {
        console.log('Sending frames to backend...');
        
        const response = await fetch('/api/video/process-batch', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                frames_base64: capturedFrames,
                fps: FPS,
                session_id: sessionId
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to process frames');
        }
        
        const results = await response.json();
        console.log('Processing complete:', results);
        console.log('Timeline entries:', results.timeline ? results.timeline.length : 0);
        console.log('Summary:', JSON.stringify(results.summary, null, 2));
        
        // Store results
        timeline = results.timeline || [];
        summary = results.summary || {};
        
        console.log('Stored summary:', JSON.stringify(summary, null, 2));
        
        // Update progress to 100%
        document.getElementById('progressBar').style.width = '100%';
        document.getElementById('progressText').textContent = 'Processing complete!';
        
        // Submit to save
        await submitResults();
        
        // Show results
        console.log('Calling showResults with:', JSON.stringify(summary, null, 2));
        showResults(summary);
        
    } catch (error) {
        console.error('Error processing frames:', error);
        alert('Failed to process video. Please try again.');
        
        // Show partial results if available
        if (summary) {
            showResults(summary);
        }
    }
}

async function submitResults() {
    try {
        const response = await fetch('/api/video/submit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: sessionId,
                trigger_video: 'hack.mp4',
                duration_seconds: videoElement.duration,
                timeline: timeline,
                summary: summary
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to submit results');
        }
        
        const result = await response.json();
        console.log('Results saved:', result);
        
    } catch (error) {
        console.error('Error saving results:', error);
        // Continue anyway, show results
    }
}

function showResults(summary) {
    console.log('=== showResults() called ===');
    console.log('Summary object:', summary);
    console.log('Summary type:', typeof summary);
    console.log('Has blink?', !!summary.blink);
    console.log('Has emotion?', !!summary.emotion);
    console.log('Has gaze?', !!summary.gaze);
    console.log('Has pupil?', !!summary.pupil);
    
    // Hide progress, show results
    document.getElementById('progressSection').classList.add('hidden');
    document.getElementById('resultsSection').classList.remove('hidden');
    
    // Populate results
    if (summary.blink) {
        console.log('Setting blink data:', summary.blink);
        document.getElementById('totalBlinks').textContent = summary.blink.total_blinks || 0;
        document.getElementById('blinkRate').textContent = `${summary.blink.blink_rate_per_minute || 0}/min`;
    } else {
        console.warn('No blink data in summary');
    }
    
    if (summary.emotion) {
        console.log('Setting emotion data:', summary.emotion);
        document.getElementById('dominantEmotion').textContent = summary.emotion.dominant_emotion || '-';
        
        // Emotion distribution
        const emotionDist = document.getElementById('emotionDistribution');
        emotionDist.innerHTML = '';
        const distribution = summary.emotion.distribution || {};
        console.log('Emotion distribution:', distribution);
        for (const [emotion, count] of Object.entries(distribution)) {
            const badge = document.createElement('div');
            badge.className = 'emotion-badge';
            badge.textContent = `${emotion}: ${count}`;
            emotionDist.appendChild(badge);
        }
    } else {
        console.warn('No emotion data in summary');
    }
    
    if (summary.gaze) {
        console.log('Setting gaze data:', summary.gaze);
        const attentionScore = summary.gaze.attention_score || 0;
        document.getElementById('attentionScore').textContent = `${Math.round(attentionScore * 100)}%`;
        
        const gazePct = summary.gaze.distribution_percentage || {};
        console.log('Gaze percentages:', gazePct);
        document.getElementById('gazeLeft').textContent = `${Math.round(gazePct.left || 0)}%`;
        document.getElementById('gazeCenter').textContent = `${Math.round(gazePct.center || 0)}%`;
        document.getElementById('gazeRight').textContent = `${Math.round(gazePct.right || 0)}%`;
    } else {
        console.warn('No gaze data in summary');
    }
    
    if (summary.pupil) {
        console.log('Setting pupil data:', summary.pupil);
        document.getElementById('avgPupil').textContent = (summary.pupil.avg_pupil_size || 0).toFixed(4);
        document.getElementById('dilationEvents').textContent = summary.pupil.pupil_dilation_events || 0;
    } else {
        console.warn('No pupil data in summary');
    }
    
    console.log('=== Results display complete ===');
}
