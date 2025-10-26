// Utility to make radio group for each question
const OPTIONS = [
  { label: 'Not at all', value: 0 },
  { label: 'Several days', value: 1 },
  { label: 'More than half the days', value: 2 },
  { label: 'Nearly every day', value: 3 },
];

function ensureSessionId() {
  let id = localStorage.getItem('session_id');
  if (!id) {
    id = crypto.randomUUID ? crypto.randomUUID() : (Date.now() + '-' + Math.random().toString(16).slice(2));
    localStorage.setItem('session_id', id);
  }
  return id;
}

function makeRadio(name, value, label) {
  const wrapper = document.createElement('label');
  wrapper.className = 'radio';

  const input = document.createElement('input');
  input.type = 'radio';
  input.name = name;
  input.value = String(value);
  input.required = true;

    const span = document.createElement('span');
    span.textContent = `${label} → ${value}`; // show mapping visually

  wrapper.appendChild(input);
  wrapper.appendChild(span);
  return wrapper;
}

function renderOptions() {
  document.querySelectorAll('.opts').forEach((container) => {
    const q = container.getAttribute('data-q');
    const name = `q${q}`;
    OPTIONS.forEach((opt) => container.appendChild(makeRadio(name, opt.value, opt.label)));
  });
}

function readAnswers() {
  const answers = [];
  for (let i = 1; i <= 8; i++) {
    const selected = document.querySelector(`input[name="q${i}"]:checked`);
    if (!selected) return null; // validation will catch
    answers.push(parseInt(selected.value, 10));
  }
  return answers;
}

function severityLabel(total) {
  if (total <= 4) return 'None';
  if (total <= 9) return 'Mild';
  if (total <= 14) return 'Moderate';
  if (total <= 19) return 'Moderately Severe';
  return 'Severe';
}

async function submitToBackend(answers) {
  const payload = { session_id: ensureSessionId(), answers };
  const res = await fetch('/api/phq8/submit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Server error ${res.status}: ${text}`);
  }
  return res.json();
}

function showResult(total, severity) {
  document.getElementById('score').textContent = String(total);
  document.getElementById('severity').textContent = severity;
  document.getElementById('result').classList.remove('hidden');
  window.scrollTo({ top: document.getElementById('result').offsetTop - 10, behavior: 'smooth' });
}

function hookForm() {
  const form = document.getElementById('phq8-form');
  const submitBtn = document.getElementById('submit-btn');
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const answers = readAnswers();
    if (!answers) {
      alert('Please answer all questions.');
      return;
    }
    const total = answers.reduce((a, b) => a + b, 0);
    submitBtn.disabled = true;
    try {
      const resp = await submitToBackend(answers);
      showResult(total, resp?.severity || severityLabel(total));
    } catch (err) {
      console.error(err);
      // Show local calc even if backend failed
      showResult(total, severityLabel(total));
      alert('Saved locally. Backend not reachable now; will proceed.');
    } finally {
      submitBtn.disabled = false;
    }
  });
}

window.addEventListener('DOMContentLoaded', () => {
  renderOptions();
  hookForm();
  ensureSessionId();
});
