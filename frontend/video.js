(() => {
  const startBtn = document.getElementById('startBtn');
  const triggerVideo = document.getElementById('triggerVideo');
  const webcamPreview = document.getElementById('webcamPreview');
  const statusEl = document.getElementById('status');
  const sessionEl = document.getElementById('sessionId');
  const resultCard = document.getElementById('resultCard');
  const resultJson = document.getElementById('resultJson');

  function ensureSessionId() {
    let id = localStorage.getItem('session_id');
    if (!id) {
      id = (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`);
      localStorage.setItem('session_id', id);
    }
    return id;
  }

  const sessionId = ensureSessionId();
  sessionEl.textContent = sessionId;

  let mediaStream = null;
  let mediaRecorder = null;
  let chunks = [];
  let recordingStarted = false;

  function logStatus(msg) {
    console.log('[video]', msg);
    statusEl.textContent = msg;
  }

  async function startWebcam() {
    if (mediaStream) return mediaStream;
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    webcamPreview.srcObject = mediaStream;
    await webcamPreview.play().catch(() => {});
    return mediaStream;
  }

  function pickMimeType() {
    const candidates = [
      'video/webm;codecs=vp9',
      'video/webm;codecs=vp8',
      'video/webm',
      'video/mp4'
    ];
    for (const t of candidates) {
      if (MediaRecorder.isTypeSupported(t)) return t;
    }
    return '';
  }

  function setupRecorder(stream) {
    const mimeType = pickMimeType();
    const options = mimeType ? { mimeType, videoBitsPerSecond: 2_500_000 } : { videoBitsPerSecond: 2_500_000 };
    const rec = new MediaRecorder(stream, options);
    chunks = [];
    rec.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    };
    rec.onstart = () => { recordingStarted = true; logStatus('Recording webcam...'); };
    rec.onstop = async () => {
      logStatus('Recording stopped. Preparing upload...');
      const blob = new Blob(chunks, { type: rec.mimeType || 'video/webm' });
      await uploadRecording(blob);
    };
    return rec;
  }

  async function goFullscreen(elem) {
    if (elem.requestFullscreen) return elem.requestFullscreen();
    if (elem.webkitRequestFullscreen) return elem.webkitRequestFullscreen();
    if (elem.msRequestFullscreen) return elem.msRequestFullscreen();
  }

  async function exitFullscreen() {
    if (document.fullscreenElement) return document.exitFullscreen();
    if (document.webkitFullscreenElement) return document.webkitExitFullscreen();
    if (document.msFullscreenElement) return document.msExitFullscreen();
  }

  async function uploadRecording(blob) {
    try {
      logStatus('Uploading recording to server...');
      const fd = new FormData();
      const filename = (blob.type && blob.type.includes('mp4')) ? 'webcam.mp4' : 'webcam.webm';
      fd.append('file', blob, filename);
      fd.append('session_id', sessionId);
      const res = await fetch('/api/video/upload', { method: 'POST', body: fd });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(`Upload failed ${res.status}: ${t}`);
      }
      const json = await res.json();
      logStatus('Upload complete. Frames extracted.');
      resultCard.classList.remove('hidden');
      resultJson.textContent = JSON.stringify(json, null, 2);

      // Start processing (blink detection) and show progress
      const fd2 = new FormData();
      fd2.append('session_id', sessionId);
      fd2.append('frames_dir', json.frames_dir);
      const startRes = await fetch('/api/video/process', { method: 'POST', body: fd2 });
      if (!startRes.ok) {
        const t = await startRes.text();
        throw new Error(`Process start failed ${startRes.status}: ${t}`);
      }
      logStatus('Started processing: Blink detection...');
      pollStatus();
    } catch (err) {
      console.error(err);
      logStatus(`Error: ${err.message}`);
    } finally {
      // cleanup media
      try { mediaRecorder && mediaRecorder.state !== 'inactive' && mediaRecorder.stop(); } catch {}
      if (mediaStream) {
        mediaStream.getTracks().forEach(t => t.stop());
        mediaStream = null;
      }
    }
  }

  async function pollStatus() {
    let done = false;
    while (!done) {
      try {
        const res = await fetch(`/api/video/status/${encodeURIComponent(sessionId)}`);
        if (!res.ok) throw new Error('status fetch failed');
        const st = await res.json();
        const { stage, processed, total, state } = st;
        const progress = total ? `${processed}/${total}` : `${processed}`;
        logStatus(`Processing: ${stage} ${progress} ${state === 'done' ? '(done)' : ''}`);
        if (state === 'done' || state === 'error') {
          done = true;
          // On completion, fetch combined summary and navigate to analysis hub
          try {
            const sumRes = await fetch(`/api/video/summary/${encodeURIComponent(sessionId)}`);
            if (sumRes.ok) {
              const summary = await sumRes.json();
              sessionStorage.setItem('video_summary', JSON.stringify(summary));
            }
          } catch (e) {
            console.warn('failed to fetch video summary', e);
          }
          // Redirect to analysis hub (no actions wired yet)
          window.location.href = '/analysis';
          break;
        }
      } catch (e) {
        console.warn('status error', e);
      }
      await new Promise(r => setTimeout(r, 1000));
    }
  }

  async function start() {
    try {
      startBtn.disabled = true;
      resultCard.classList.add('hidden');
      logStatus('Requesting webcam access...');
      const stream = await startWebcam();

      logStatus('Preparing recorder...');
      mediaRecorder = setupRecorder(stream);

      // Prepare trigger video
      triggerVideo.src = '/api/video/trigger';
      triggerVideo.currentTime = 0;

      // On end: stop recording and exit fullscreen
      triggerVideo.onended = async () => {
        logStatus('Trigger video ended. Stopping recording...');
        try { await exitFullscreen(); } catch {}
        try { mediaRecorder && mediaRecorder.stop(); } catch {}
      };

      // Start recording right before playing
      mediaRecorder.start(1000); // gather data every second

      // Play and request fullscreen
      await triggerVideo.play();
      await goFullscreen(triggerVideo);
      logStatus('Playing trigger video in fullscreen and recording webcam...');

    } catch (err) {
      console.error(err);
      logStatus(`Error: ${err.message}`);
      startBtn.disabled = false;
    }
  }

  startBtn.addEventListener('click', start);
})();
