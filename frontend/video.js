/**
 * Video Analysis Module
 * Records webcam while playing trigger video, then processes through ML pipeline
 */

// Global state
let stream = null;
let mediaRecorder = null;
let recordedChunks = [];
let videoElement = null;
let sessionId = null;
let isRecording = false;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Video analysis page loaded');
    
    videoElement = document.getElementById('triggerVideo');
    
    // Get or create session ID
    sessionId = localStorage.getItem('session_id');
    if (!sessionId) {
        console.warn('No session ID found, creating temporary session');
        sessionId = 'session_' + Date.now();
        localStorage.setItem('session_id', sessionId);
    }
    
    console.log('Session ID:', sessionId);
    
    // Set up start button
    const startBtn = document.getElementById('startBtn');
    startBtn.addEventListener('click', startAnalysis);
});

async function startAnalysis() {
    console.log('Starting video analysis...');
    
    const startBtn = document.getElementById('startBtn');
    const instructionsScreen = document.getElementById('instructionsScreen');
    const videoScreen = document.getElementById('videoScreen');
    const webcamPreview = document.getElementById('webcamPreview');
    
    startBtn.disabled = true;
    startBtn.textContent = 'Requesting camera access...';
    
    try {
        // Request webcam access
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                frameRate: { ideal: 30 }
            },
            audio: false
        });
        
        console.log('✓ Webcam access granted');
        
        // Set up webcam preview (hidden, but recording)
        webcamPreview.srcObject = stream;
        
        // Hide instructions, show video screen
        instructionsScreen.classList.add('hidden');
        videoScreen.classList.add('active');
        
        // Enter fullscreen
        try {
            if (videoScreen.requestFullscreen) {
                await videoScreen.requestFullscreen();
            } else if (videoScreen.webkitRequestFullscreen) {
                await videoScreen.webkitRequestFullscreen();
            } else if (videoScreen.msRequestFullscreen) {
                await videoScreen.msRequestFullscreen();
            }
            console.log('✓ Entered fullscreen mode');
        } catch (fsError) {
            console.warn('Fullscreen not available:', fsError);
        }
        
        // Set up MediaRecorder
        recordedChunks = [];
        
        // Use vp9 codec if available, otherwise vp8
        let mimeType = 'video/webm;codecs=vp9';
        if (!MediaRecorder.isTypeSupported(mimeType)) {
            mimeType = 'video/webm;codecs=vp8';
            console.log('vp9 not supported, using vp8');
        }
        
        mediaRecorder = new MediaRecorder(stream, { 
            mimeType: mimeType,
            videoBitsPerSecond: 2500000 // 2.5 Mbps
        });
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                recordedChunks.push(event.data);
                console.log(`Recording chunk: ${event.data.size} bytes`);
            }
        };
        
        mediaRecorder.onstop = async () => {
            console.log('Recording stopped');
            
            // Exit fullscreen
            if (document.fullscreenElement) {
                try {
                    await document.exitFullscreen();
                } catch (e) {
                    console.warn('Could not exit fullscreen:', e);
                }
            }
            
            // Hide video screen
            videoScreen.classList.remove('active');
            
            // Stop webcam
            if (stream) {
                stream.getTracks().forEach(track => {
                    track.stop();
                    console.log('Stopped track:', track.kind);
                });
            }
            
            // Create blob and upload
            const blob = new Blob(recordedChunks, { type: mimeType });
            console.log(`Total recording size: ${(blob.size / 1024 / 1024).toFixed(2)} MB`);
            
            await uploadAndProcess(blob);
        };
        
        // Start recording
        isRecording = true;
        mediaRecorder.start(1000); // Capture in 1-second chunks
        console.log('✓ Recording started');
        
        // Play trigger video
        videoElement.play();
        console.log('✓ Trigger video playing');
        
        // When video ends, stop recording
        videoElement.addEventListener('ended', () => {
            console.log('Trigger video ended');
            
            if (isRecording && mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                isRecording = false;
            }
        }, { once: true });
        
        // Handle manual fullscreen exit
        document.addEventListener('fullscreenchange', () => {
            if (!document.fullscreenElement && isRecording) {
                console.log('User exited fullscreen early');
                videoElement.pause();
                if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                    mediaRecorder.stop();
                    isRecording = false;
                }
            }
        }, { once: true });
        
    } catch (error) {
        console.error('Failed to start analysis:', error);
        alert('Could not access camera. Please check permissions and try again.');
        startBtn.disabled = false;
        startBtn.textContent = 'Start Analysis';
        
        // Clean up
        if (document.fullscreenElement) {
            document.exitFullscreen();
        }
        instructionsScreen.classList.remove('hidden');
        videoScreen.classList.remove('active');
    }
}

async function uploadAndProcess(videoBlob) {
    console.log('Starting upload and processing...');
    
    const loadingScreen = document.getElementById('loadingScreen');
    const progressBar = document.getElementById('progressBar');
    
    // Show loading screen
    loadingScreen.classList.add('active');
    progressBar.style.width = '0%';
    
    // Update stages
    const updateStage = (stageId, status) => {
        const stage = document.getElementById(stageId);
        const statusIcon = stage.querySelector('.stage-status');
        
        if (status === 'active') {
            stage.classList.add('active');
            statusIcon.textContent = '⏳';
        } else if (status === 'complete') {
            stage.classList.remove('active');
            stage.classList.add('complete');
            statusIcon.textContent = '✅';
        } else if (status === 'error') {
            stage.classList.remove('active');
            statusIcon.textContent = '❌';
        }
    };
    
    try {
        // Stage 1: Upload
        updateStage('stageUpload', 'active');
        progressBar.style.width = '10%';
        
        const formData = new FormData();
        formData.append('session_id', sessionId);
        formData.append('video_file', videoBlob, 'recording.webm');
        
        console.log('Uploading video...');
        
        const response = await fetch('/api/video/upload-and-process', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Upload failed: ${response.status} - ${errorText}`);
        }
        
        updateStage('stageUpload', 'complete');
        progressBar.style.width = '20%';
        
        // Stage 2: Blink detection
        updateStage('stageBlink', 'active');
        progressBar.style.width = '40%';
        
        // Simulate processing stages (backend processes all at once)
        await new Promise(resolve => setTimeout(resolve, 1000));
        updateStage('stageBlink', 'complete');
        
        // Stage 3: Gaze tracking
        updateStage('stageGaze', 'active');
        progressBar.style.width = '60%';
        await new Promise(resolve => setTimeout(resolve, 1000));
        updateStage('stageGaze', 'complete');
        
        // Stage 4: Pupil analysis
        updateStage('stagePupil', 'active');
        progressBar.style.width = '80%';
        await new Promise(resolve => setTimeout(resolve, 1000));
        updateStage('stagePupil', 'complete');
        
        // Stage 5: Emotion recognition
        updateStage('stageEmotion', 'active');
        progressBar.style.width = '90%';
        
        // Get results
        const results = await response.json();
        console.log('Processing complete:', results);
        
        updateStage('stageEmotion', 'complete');
        progressBar.style.width = '100%';
        
        // Wait a moment to show completion
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Save results
        await saveResults(results);
        
        // Show completion screen with options
        showCompletionScreen();
        
    } catch (error) {
        console.error('Processing failed:', error);
        alert(`Failed to process video: ${error.message}`);
        
        // Mark current stage as error
        document.querySelectorAll('.stage-item.active').forEach(stage => {
            updateStage(stage.id, 'error');
        });
        
        // Allow retry
        setTimeout(() => {
            location.reload();
        }, 3000);
    }
}

async function saveResults(results) {
    try {
        const response = await fetch('/api/video/submit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: sessionId,
                trigger_video: 'hack.mp4',
                duration_seconds: videoElement.duration || 0,
                timeline: results.timeline || [],
                summary: results.summary || {}
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to save results');
        }
        
        const result = await response.json();
        console.log('✓ Results saved:', result);
        
    } catch (error) {
        console.error('Error saving results:', error);
        // Continue anyway
    }
}

function showCompletionScreen() {
    const loadingScreen = document.getElementById('loadingScreen');
    const loadingContent = loadingScreen.querySelector('.loading-content');
    
    // Replace loading content with completion UI
    loadingContent.innerHTML = `
        <div style="text-align: center;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">✅</div>
            <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">Analysis Complete!</h1>
            <p style="font-size: 1.2rem; opacity: 0.9; margin-bottom: 3rem;">
                Your video has been processed successfully.<br>
                All facial expressions, gaze, and emotions have been analyzed.
            </p>
            
            <div style="display: flex; gap: 2rem; justify-content: center; flex-wrap: wrap; max-width: 600px; margin: 0 auto;">
                <button onclick="window.location.href='/dashboard'" 
                        style="padding: 1.5rem 3rem; font-size: 1.3rem; background: white; color: #667eea; 
                               border: none; border-radius: 50px; cursor: pointer; font-weight: 700; 
                               box-shadow: 0 10px 30px rgba(0,0,0,0.3); transition: all 0.3s ease;
                               min-width: 250px;">
                    📊 View Dashboard
                </button>
                
                <button onclick="window.location.href='/ai-analysis'" 
                        style="padding: 1.5rem 3rem; font-size: 1.3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                               color: white; border: 2px solid white; border-radius: 50px; cursor: pointer; font-weight: 700; 
                               transition: all 0.3s ease; min-width: 250px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
                    🤖 AI Analysis
                </button>
            </div>
            
            <p style="margin-top: 3rem; opacity: 0.7; font-size: 0.9rem;">
                Session ID: ${sessionId}
            </p>
        </div>
    `;
}
