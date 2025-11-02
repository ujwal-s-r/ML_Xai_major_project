(() => {
  const sessionEl = document.getElementById('sessionId');
  const aiBtn = document.getElementById('aiBtn');
  const dashBtn = document.getElementById('dashBtn');

  function getSessionId() {
    return localStorage.getItem('session_id') || '';
  }

  function getStoredSummary() {
    try {
      const raw = sessionStorage.getItem('video_summary');
      return raw ? JSON.parse(raw) : null;
    } catch { return null; }
  }

  async function fetchSummaryIfMissing(sessionId) {
    // Fallback: fetch from server if not present
    if (!sessionId) return null;
    try {
      const res = await fetch(`/api/video/summary/${encodeURIComponent(sessionId)}`);
      if (!res.ok) return null;
      const json = await res.json();
      sessionStorage.setItem('video_summary', JSON.stringify(json));
      return json;
    } catch { return null; }
  }

  async function init() {
    const sessionId = getSessionId();
    sessionEl.textContent = sessionId || '-';

    let summary = getStoredSummary();
    if (!summary) {
      summary = await fetchSummaryIfMissing(sessionId);
    }

    // Buttons are placeholders for now; no actions wired per request
    aiBtn.addEventListener('click', (e) => {
      e.preventDefault();
      // Intentionally left unimplemented
    });
    dashBtn.addEventListener('click', (e) => {
      e.preventDefault();
      // Intentionally left unimplemented
    });
  }

  init();
})();
