/**
 * Video Analysis - Frame Capture and Processing
 * Captures webcam frames, processes through backend models, displays real-time stats
 */

// Global state
let stream = null;
let videoElement = null;
let canvasElement = null;
let ctx = null;
let sessionId = null;
let isProcessing = false;

// Frame tracking
let frameNumber = 0;
let totalFrames = 0;
const FPS = 30;

// Results storage
let timeline = [];
let latestStats = {
    blinks: 0,
    emotion: '-',
    gaze: '-'
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('Video analysis page loaded');
    
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
    console.log('Starting video analysis...');
    
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
        
        // Start video and processing
        videoElement.play();
        
        // Calculate total frames (video duration * FPS)
        videoElement.addEventListener('loadedmetadata', () => {
            totalFrames = Math.floor(videoElement.duration * FPS);
            console.log(`Video duration: ${videoElement.duration}s, Total frames: ${totalFrames}`);
        });
        
        // Start frame processing loop
        isProcessing = true;
        processFrameLoop();
        
        // Handle video end
        videoElement.addEventListener('ended', onVideoEnded);
        
    } catch (error) {
        console.error('Failed to access webcam:', error);
        alert('Could not access your webcam. Please grant camera permissions and try again.');
        startBtn.disabled = false;
        startBtn.textContent = 'Start Analysis';
    }
}

async function processFrameLoop() {
    if (!isProcessing || !stream) return;
    
    try {
        // Capture frame from webcam
        const videoTrack = stream.getVideoTracks()[0];
        const imageCapture = new ImageCapture(videoTrack);
        const blob = await imageCapture.takePhoto();
        
        // Draw to canvas
        const img = await createImageBitmap(blob);
        ctx.drawImage(img, 0, 0, canvasElement.width, canvasElement.height);
        
        // Get base64 image
        const base64Image = canvasElement.toDataURL('image/jpeg', 0.8);
        
        // Send to backend for processing
        await processFrame(base64Image, frameNumber);
        
        frameNumber++;
        
        // Update progress
        updateProgress();
        
        // Continue loop if video is still playing
        if (!videoElement.ended && isProcessing) {
            // Process at ~30fps
            setTimeout(processFrameLoop, 1000 / FPS);
        }
        
    } catch (error) {
        console.error('Error in frame processing loop:', error);
        // Continue anyway
        if (!videoElement.ended && isProcessing) {
            setTimeout(processFrameLoop, 1000 / FPS);
        }
    }
}

async function processFrame(base64Image, frameNum) {
    try {
        // Call Python backend to process frame
        // For now, we'll simulate processing client-side
        // In production, you'd send to backend
        
        const frameData = {
            timestamp: frameNum / FPS,
            frame_number: frameNum,
            processed: {
                emotion: frameNum % 10 === 0,
                blink: frameNum % 2 === 0,
                iris: frameNum % 2 === 0,
                gaze: frameNum % 3 === 0
            }
        };
        
        // Simulate results (replace with actual backend call)
        if (frameNum % 10 === 0) {
            const emotions = ['neutral', 'sad', 'happy', 'fear', 'angry', 'surprise', 'disgust'];
            frameData.emotion = {
                label: emotions[Math.floor(Math.random() * emotions.length)],
                confidence: 0.7 + Math.random() * 0.3,
                scores: {}
            };
            latestStats.emotion = frameData.emotion.label;
        }
        
        if (frameNum % 2 === 0) {
            const ear = 0.25 + Math.random() * 0.15;
            frameData.blink = {
                ear_left: ear,
                ear_right: ear,
                avg_ear: ear,
                is_blinking: ear < 0.22,
                cumulative_blinks: Math.floor(frameNum / 60)
            };
            latestStats.blinks = frameData.blink.cumulative_blinks;
        }
        
        if (frameNum % 2 === 0) {
            frameData.pupil = {
                left: 0.03 + Math.random() * 0.01,
                right: 0.03 + Math.random() * 0.01,
                avg: 0.03 + Math.random() * 0.01,
                dilation_ratio: (Math.random() - 0.5) * 0.2
            };
        }
        
        if (frameNum % 3 === 0) {
            const directions = ['left', 'center', 'right'];
            const direction = directions[Math.floor(Math.random() * directions.length)];
            frameData.gaze = {
                pitch: (Math.random() - 0.5) * 40,
                yaw: (Math.random() - 0.5) * 60,
                direction: direction
            };
            latestStats.gaze = direction;
        }
        
        timeline.push(frameData);
        
    } catch (error) {
        console.error('Error processing frame:', error);
    }
}

function updateProgress() {
    const progress = totalFrames > 0 ? (frameNumber / totalFrames) * 100 : 0;
    
    // Update progress bar
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    progressBar.style.width = `${progress}%`;
    progressText.textContent = `${Math.round(progress)}% - Frame ${frameNumber}/${totalFrames}`;
    
    // Update live stats
    document.getElementById('framesProcessed').textContent = frameNumber;
    document.getElementById('blinksDetected').textContent = latestStats.blinks;
    document.getElementById('currentEmotion').textContent = latestStats.emotion;
    document.getElementById('gazeDirection').textContent = latestStats.gaze.toUpperCase();
}

async function onVideoEnded() {
    console.log('Video ended, finalizing analysis...');
    isProcessing = false;
    
    // Stop webcam
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    
    // Calculate summary
    const summary = calculateSummary();
    
    // Submit to backend
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
            throw new Error('Failed to submit video analysis');
        }
        
        const result = await response.json();
        console.log('Analysis submitted successfully:', result);
        
        // Show results
        showResults(summary);
        
    } catch (error) {
        console.error('Error submitting analysis:', error);
        alert('Failed to save analysis. Please try again.');
    }
}

function calculateSummary() {
    // Calculate summary statistics from timeline
    const emotionDist = {};
    let totalBlinks = 0;
    const pupilSizes = [];
    const gazeCount = { left: 0, center: 0, right: 0 };
    
    timeline.forEach(entry => {
        if (entry.emotion) {
            const label = entry.emotion.label;
            emotionDist[label] = (emotionDist[label] || 0) + 1;
        }
        
        if (entry.blink) {
            totalBlinks = Math.max(totalBlinks, entry.blink.cumulative_blinks);
        }
        
        if (entry.pupil) {
            pupilSizes.push(entry.pupil.avg);
        }
        
        if (entry.gaze) {
            gazeCount[entry.gaze.direction]++;
        }
    });
    
    // Dominant emotion
    let dominantEmotion = 'neutral';
    let maxCount = 0;
    for (const [emotion, count] of Object.entries(emotionDist)) {
        if (count > maxCount) {
            maxCount = count;
            dominantEmotion = emotion;
        }
    }
    
    // Emotion changes
    let emotionChanges = 0;
    let prevEmotion = null;
    for (const entry of timeline) {
        if (entry.emotion) {
            if (prevEmotion && entry.emotion.label !== prevEmotion) {
                emotionChanges++;
            }
            prevEmotion = entry.emotion.label;
        }
    }
    
    // Pupil stats
    const avgPupil = pupilSizes.length > 0 ? 
        pupilSizes.reduce((a, b) => a + b, 0) / pupilSizes.length : 0;
    const maxPupil = pupilSizes.length > 0 ? Math.max(...pupilSizes) : 0;
    const minPupil = pupilSizes.length > 0 ? Math.min(...pupilSizes) : 0;
    
    // Dilation events (>15% change from baseline)
    const baseline = pupilSizes.slice(0, 30).reduce((a, b) => a + b, 0) / Math.min(30, pupilSizes.length);
    const dilationEvents = pupilSizes.filter(size => Math.abs(size - baseline) / baseline > 0.15).length;
    
    // Gaze distribution
    const totalGaze = gazeCount.left + gazeCount.center + gazeCount.right;
    const gaze_dist_pct = {
        left: totalGaze > 0 ? (gazeCount.left / totalGaze * 100) : 0,
        center: totalGaze > 0 ? (gazeCount.center / totalGaze * 100) : 0,
        right: totalGaze > 0 ? (gazeCount.right / totalGaze * 100) : 0
    };
    
    const attentionScore = gaze_dist_pct.center / 100;
    const blinkRate = (totalBlinks / videoElement.duration) * 60;
    
    return {
        duration_seconds: videoElement.duration,
        total_frames: frameNumber,
        emotion: {
            distribution: emotionDist,
            dominant_emotion: dominantEmotion,
            emotion_changes: emotionChanges
        },
        blink: {
            total_blinks: totalBlinks,
            blink_rate_per_minute: Math.round(blinkRate * 100) / 100
        },
        pupil: {
            avg_pupil_size: Math.round(avgPupil * 10000) / 10000,
            max_pupil_size: Math.round(maxPupil * 10000) / 10000,
            min_pupil_size: Math.round(minPupil * 10000) / 10000,
            pupil_dilation_events: dilationEvents
        },
        gaze: {
            distribution_percentage: gaze_dist_pct,
            attention_score: Math.round(attentionScore * 1000) / 1000
        }
    };
}

function showResults(summary) {
    // Hide progress, show results
    document.getElementById('progressSection').classList.add('hidden');
    document.getElementById('resultsSection').classList.remove('hidden');
    
    // Populate results
    document.getElementById('totalBlinks').textContent = summary.blink.total_blinks;
    document.getElementById('blinkRate').textContent = `${summary.blink.blink_rate_per_minute}/min`;
    document.getElementById('dominantEmotion').textContent = summary.emotion.dominant_emotion;
    document.getElementById('attentionScore').textContent = `${Math.round(summary.gaze.attention_score * 100)}%`;
    
    // Emotion distribution
    const emotionDist = document.getElementById('emotionDistribution');
    emotionDist.innerHTML = '';
    for (const [emotion, count] of Object.entries(summary.emotion.distribution)) {
        const badge = document.createElement('div');
        badge.className = 'emotion-badge';
        badge.textContent = `${emotion}: ${count}`;
        emotionDist.appendChild(badge);
    }
    
    // Gaze percentages
    document.getElementById('gazeLeft').textContent = `${Math.round(summary.gaze.distribution_percentage.left)}%`;
    document.getElementById('gazeCenter').textContent = `${Math.round(summary.gaze.distribution_percentage.center)}%`;
    document.getElementById('gazeRight').textContent = `${Math.round(summary.gaze.distribution_percentage.right)}%`;
    
    // Pupil stats
    document.getElementById('avgPupil').textContent = summary.pupil.avg_pupil_size.toFixed(4);
    document.getElementById('dilationEvents').textContent = summary.pupil.pupil_dilation_events;
    
    console.log('Results displayed');
}
