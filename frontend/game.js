(function(){
  const DURATION_MS = 60000; // 60s
  const TARGET_TTL_INITIAL = 1500; // adaptive target lifetime
  const ARENA_PADDING = 20; // px safety border
  const INITIAL_SPAWN_MS = 1500; // start spawn interval (slower start)
  const MIN_SPAWN_MS = 600; // min interval
  const ACCEL_PER_SEC = 8; // slower natural accel; adaptive will adjust
  
  // Fixed distraction timing: exactly 2 distractions at 20s and 40s
  const DISTRACTION_TIMES = [20000, 40000];
  const WINDOW_MS = 5000; // pre/post window duration
  const MIN_OPPORTUNITIES_PER_WINDOW = 5; // guarantee at least 5 eligible targets per window

  const arena = document.getElementById('arena');
  const overlay = document.getElementById('overlay');
  const startBtn = document.getElementById('startBtn');
  const uiTime = document.getElementById('time');
  const uiHits = document.getElementById('hits');
  const uiErrors = document.getElementById('errors');
  const uiAvgRt = document.getElementById('avgRt');
  const ruleBanner = document.getElementById('ruleBanner');

  const summaryEl = document.getElementById('summary');
  const sumAcc = document.getElementById('sumAcc');
  const sumRt = document.getElementById('sumRt');
  const sumImp = document.getElementById('sumImp');
  const distractionSummary = document.getElementById('distractionSummary');

  const sessionId = (() => {
    let id = localStorage.getItem('session_id');
    if (!id) {
      id = crypto.randomUUID ? crypto.randomUUID() : (Date.now() + '-' + Math.random().toString(16).slice(2));
      localStorage.setItem('session_id', id);
    }
    return id;
  })();

  const state = {
    running: false,
    startTime: 0,
    lastTick: 0,
    nextSpawnAt: 0,
    spawnInterval: INITIAL_SPAWN_MS,
    targetTtl: TARGET_TTL_INITIAL,
    fakeProb: 0.20, // base chance for red fakes (adaptive)
    pAllowed: 0.70, // during rule windows, share of allowed green vs disallowed green (adaptive)
    targets: new Map(), // id -> {id, x, y, vx, vy, type, fake, spawnTs}
    targetSeq: 0,
    hitCount: 0,
    errorCount: 0,
    rtList: [],
    recentOutcomes: [], // last N booleans (true hit, false error/miss)
    taps: [], // {ts, x, y, targetId, correct, premature, ruleAtTap}
    targetEvents: [], // {id, spawnTs, despawnTs, type, fake, ruleAtSpawn}
    distractions: [], // {id, kind, startTs, duration}
    distractionIndex: 0, // track which distraction is next
    rule: 'normal', // 'normal' | 'onlySquares' | 'onlyCircles' | 'onlyTriangles'
    ruleUntil: 0,
    ruleStart: 0,
    ruleWindows: [], // {rule, startTs, endTs}
    inOpportunityWindow: false, // flag when we're in a pre/post window needing opportunity floor
  };

  function now() { return performance.now(); }
  function elapsed() { return Math.max(0, now() - state.startTime); }
  function remainingMs() { return Math.max(0, DURATION_MS - elapsed()); }

  function randInt(min, max) { return Math.floor(Math.random() * (max - min + 1)) + min; }
  function pick(arr) { return arr[Math.floor(Math.random()*arr.length)]; }

  function randomPos() {
    const rect = arena.getBoundingClientRect();
    const size = 40;
    const x = randInt(ARENA_PADDING, Math.max(ARENA_PADDING, rect.width - size - ARENA_PADDING));
    const y = randInt(ARENA_PADDING, Math.max(ARENA_PADDING, rect.height - size - ARENA_PADDING));
    return {x,y};
  }

  function showOverlay(msg, duration=700) {
    overlay.textContent = msg;
    overlay.classList.remove('hidden');
    overlay.classList.add('show');
    setTimeout(()=>{
      overlay.classList.remove('show');
      overlay.classList.add('hidden');
      overlay.textContent = '';
    }, duration);
  }

  function scheduleNextDistraction() {
    // No-op now; we use fixed DISTRACTION_TIMES
  }

  function checkAndTriggerDistraction(t) {
    if (state.distractionIndex < DISTRACTION_TIMES.length) {
      const nextTime = DISTRACTION_TIMES[state.distractionIndex];
      if (t >= nextTime) {
        triggerDistraction(nextTime);
        state.distractionIndex++;
      }
    }
  }

  function triggerDistraction(startTs) {
    // No audio per requirements
    const kind = pick(['visual','text']);
    const duration = randInt(500, 1000);
    const id = state.distractionIndex; // 0-indexed
    const rec = { id, kind, startTs, duration };
    state.distractions.push(rec);

    if (kind === 'visual') showOverlay('Ignore this!', duration);
    if (kind === 'text') {
      // pick a rule and set a timer
      state.rule = pick(['onlySquares','onlyCircles','onlyTriangles']);
      state.ruleStart = rec.startTs;
      state.ruleUntil = rec.startTs + duration + 2500; // rule lingers slightly beyond message
      state.ruleWindows.push({ rule: state.rule, startTs: state.ruleStart, endTs: null });
      updateRuleBanner();
      const txt = state.rule === 'onlySquares' ? 'Rule: ONLY squares!' : (state.rule === 'onlyCircles' ? 'Rule: ONLY circles!' : 'Rule: ONLY triangles!');
      showOverlay(txt, duration);
    }

    // During distraction, spawn a fake target once
    const makeFake = () => spawnTarget(true, null);
    setTimeout(makeFake, Math.min(200, duration - 100));
  }

  function spawnTarget(fake=false, forcedType=null) {
    const id = ++state.targetSeq;
    const {x,y} = randomPos();
  const type = forcedType || pick(['circle','square','triangle']);
    const el = document.createElement('div');
    el.className = `target ${fake? 'fake':'valid'} ${type}`;
    el.style.left = `${x}px`;
    el.style.top = `${y}px`;
    el.dataset.id = String(id);
    arena.appendChild(el);

    // gentle movement
    const vx = (Math.random()*0.12 - 0.06); // px per ms
    const vy = (Math.random()*0.12 - 0.06);

  const rec = { id, x, y, vx, vy, type, fake, spawnTs: elapsed(), ruleAtSpawn: state.rule };
    state.targets.set(id, rec);
  state.targetEvents.push({ id, spawnTs: rec.spawnTs, despawnTs: null, type, fake, ruleAtSpawn: state.rule });

    const onClick = (ev) => {
      if (!state.running) return;
      const ts = elapsed();
      // correctness considers fake and current rule
      let correct = !fake;
      if (correct && state.rule !== 'normal') {
        if (state.rule === 'onlySquares' && type !== 'square') correct = false;
        if (state.rule === 'onlyCircles' && type !== 'circle') correct = false;
        if (state.rule === 'onlyTriangles' && type !== 'triangle') correct = false;
      }
      const rt = ts - rec.spawnTs;
      if (correct) {
        state.hitCount += 1;
        state.rtList.push(rt);
      } else {
        state.errorCount += 1;
      }
  state.taps.push({ ts, x: ev.offsetX, y: ev.offsetY, targetId: id, correct, premature: false, ruleAtTap: state.rule });
      pushOutcome(correct);
      removeTarget(id);
      updateHud();
      ev.stopPropagation();
    };

    el.addEventListener('click', onClick, { once:true });

    setTimeout(()=>{
      // despawn if still present (miss)
      if (state.targets.has(id)) {
        pushOutcome(false); // treat as miss in adaptive window
        removeTarget(id);
        updateHud();
      }
    }, state.targetTtl);
  }

  function removeTarget(id) {
    const rec = state.targets.get(id);
    if (!rec) return;
    state.targets.delete(id);
    const el = arena.querySelector(`.target[data-id="${id}"]`);
    if (el && el.parentNode) el.parentNode.removeChild(el);
    // mark despawn
    const ev = state.targetEvents.find(e=>e.id===id);
    if (ev) ev.despawnTs = elapsed();
  }

  function onArenaClick(ev) {
    if (!state.running) return;
    // click not on a target -> error/premature
    const ts = elapsed();
    const anyTargets = state.targets.size > 0;
    const premature = !anyTargets;
    state.errorCount += 1;
  state.taps.push({ ts, x: ev.offsetX, y: ev.offsetY, targetId: null, correct:false, premature, ruleAtTap: state.rule });
    pushOutcome(false);
    updateHud();
  }

  function updateHud() {
    uiHits.textContent = String(state.hitCount);
    uiErrors.textContent = String(state.errorCount);
    const avg = state.rtList.length ? Math.round(state.rtList.reduce((a,b)=>a+b,0) / state.rtList.length) : '-';
    uiAvgRt.textContent = String(avg);
  }

  function tick() {
    if (!state.running) return;
    const t = elapsed();
    uiTime.textContent = (Math.max(0, (DURATION_MS - t)/1000)).toFixed(1);

    // rule expiration
    if (state.rule !== 'normal' && t >= state.ruleUntil) {
      // close last rule window
      const last = state.ruleWindows[state.ruleWindows.length - 1];
      if (last && last.endTs == null) last.endTs = t;
      state.rule = 'normal';
      state.ruleUntil = 0;
      state.ruleStart = 0;
      updateRuleBanner();
    }

    // spawn logic (accelerates + opportunity floor for pre/post windows)
    if (t >= state.nextSpawnAt) {
      // Check if we're in an opportunity window (pre/post of distractions)
      let inOpportunityWindow = false;
      for (const dt of DISTRACTION_TIMES) {
        const preStart = dt - WINDOW_MS;
        const postEnd = dt + WINDOW_MS;
        if (t >= preStart && t < postEnd) {
          inOpportunityWindow = true;
          break;
        }
      }

      // rule-aware mixed spawning: choose fake and shape based on current rule
      const makeFake = Math.random() < state.fakeProb;
      let chosenType;
      if (state.rule === 'normal') {
        chosenType = pick(['circle','square','triangle']);
      } else {
        const allowed = state.rule === 'onlySquares' ? 'square' : (state.rule === 'onlyCircles' ? 'circle' : 'triangle');
        const others = ['circle','square','triangle'].filter(s => s !== allowed);
        // In opportunity windows, ensure 100% allowed to meet floor
        const pAllowedNow = inOpportunityWindow ? 1.0 : state.pAllowed;
        const allow = Math.random() < pAllowedNow;
        chosenType = allow ? allowed : pick(others);
      }
      spawnTarget(makeFake, chosenType);
      
      // Accelerate spawn interval unless in opportunity window (then keep it faster)
      const intervalNow = inOpportunityWindow ? Math.min(state.spawnInterval, 800) : state.spawnInterval;
      state.spawnInterval = Math.max(MIN_SPAWN_MS, intervalNow - (t/1000)*ACCEL_PER_SEC);
      state.nextSpawnAt = t + (inOpportunityWindow ? 800 : state.spawnInterval);
      // occasionally adapt based on recent outcomes
      if ((state.targetSeq % 5) === 0) adaptDifficulty();
    }

    // distraction trigger
    checkAndTriggerDistraction(t);

    // move targets gently
    moveTargets();

    if (t >= DURATION_MS) {
      endGame();
      return;
    }

    state.lastTick = t;
    requestAnimationFrame(tick);
  }

  function startGame() {
    // reset
    arena.innerHTML = '';
    overlay.classList.add('hidden');
    state.running = true;
    state.startTime = now();
    state.lastTick = 0;
    state.nextSpawnAt = 0;
  state.spawnInterval = INITIAL_SPAWN_MS;
  state.targetTtl = TARGET_TTL_INITIAL;
  state.fakeProb = 0.20;
  state.pAllowed = 0.70;
    state.targets.clear();
    state.targetSeq = 0;
    state.hitCount = 0;
    state.errorCount = 0;
    state.rtList = [];
  state.recentOutcomes = [];
    state.taps = [];
    state.targetEvents = [];
    state.distractions = [];
    state.distractionIndex = 0;
  state.rule = 'normal';
  state.ruleUntil = 0;
  state.ruleStart = 0;
  state.ruleWindows = [];
  state.inOpportunityWindow = false;
  updateRuleBanner();
    scheduleNextDistraction();

    updateHud();
    summaryEl.classList.add('hidden');
    requestAnimationFrame(tick);
  }

  function endGame() {
    state.running = false;
    // remove any remaining targets
    [...state.targets.keys()].forEach(removeTarget);

    const totals = computeSummary();
    renderSummary(totals);
    submitResults(totals).catch(err=>console.error('submit error', err));
  }

  function computeSummary() {
    // Clean, simple overall metrics
    // 1. Collect ALL eligible targets: non-fake AND allowed under rule at spawn time
    const eligible = [];
    state.targetEvents.forEach(t => {
      if (t.fake) return;
      const ruleAtSpawn = t.ruleAtSpawn || 'normal';
      const allowed = _isAllowedShape(t.type, ruleAtSpawn);
      if (allowed) eligible.push(t);
    });

    // 2. Collect first valid hit per eligible target
    const hitByTarget = new Map();
    state.taps.forEach(tap => {
      const tid = tap.targetId;
      if (tid == null) return;
      const tgt = state.targetEvents.find(e => e.id === tid);
      if (!tgt || tgt.fake) return;
      const ruleAtTap = tap.ruleAtTap || 'normal';
      const allowed = _isAllowedShape(tgt.type, ruleAtTap);
      if (allowed && tap.correct && !hitByTarget.has(tid)) {
        hitByTarget.set(tid, { tapTs: tap.ts, spawnTs: tgt.spawnTs });
      }
    });

    // 3. Overall stats
    const totalEligible = eligible.length;
    const hits = hitByTarget.size;
    const misses = totalEligible - hits;
    const accuracy = totalEligible > 0 ? Math.round((hits / totalEligible) * 100) : 0;

    const rts = [];
    hitByTarget.forEach(h => rts.push(h.tapTs - h.spawnTs));
    const avgRt = rts.length > 0 ? Math.round(rts.reduce((a, b) => a + b, 0) / rts.length) : null;

    // Impulsivity: premature taps + taps on fakes
    const prematureTaps = state.taps.filter(t => t.premature).length;
    const fakeTaps = state.taps.filter(t => {
      if (t.targetId == null) return false;
      const tgt = state.targetEvents.find(e => e.id === t.targetId);
      return tgt && tgt.fake;
    }).length;
    const impulsive = prematureTaps + fakeTaps;

    // Errors: wrong taps (tapped disallowed shapes under rule + fakes + premature)
    const errors = state.taps.filter(t => !t.correct).length;

    // 4. Per-distraction pre/post with CORRECT calculation
    const perDistraction = state.distractions.map(d => {
      const preStart = d.startTs - WINDOW_MS;
      const preEnd = d.startTs;
      const postStart = d.startTs;
      const postEnd = d.startTs + WINDOW_MS;

      // Pre window eligible targets
      const preElig = eligible.filter(t => t.spawnTs >= preStart && t.spawnTs < preEnd);
      const preHits = preElig.filter(t => hitByTarget.has(t.id) && hitByTarget.get(t.id).tapTs >= preStart && hitByTarget.get(t.id).tapTs < preEnd);
      const preAcc = preElig.length > 0 ? preHits.length / preElig.length : null;
      const preRts = preHits.map(t => hitByTarget.get(t.id).tapTs - t.spawnTs);
      const preAvgRt = preRts.length > 0 ? Math.round(preRts.reduce((a,b)=>a+b,0)/preRts.length) : null;

      // Post window eligible targets
      const postElig = eligible.filter(t => t.spawnTs >= postStart && t.spawnTs < postEnd);
      const postHits = postElig.filter(t => hitByTarget.has(t.id) && hitByTarget.get(t.id).tapTs >= postStart && hitByTarget.get(t.id).tapTs < postEnd);
      const postAcc = postElig.length > 0 ? postHits.length / postElig.length : null;
      const postRts = postHits.map(t => hitByTarget.get(t.id).tapTs - t.spawnTs);
      const postAvgRt = postRts.length > 0 ? Math.round(postRts.reduce((a,b)=>a+b,0)/postRts.length) : null;

      return {
        id: d.id,
        kind: d.kind,
        startTs: d.startTs,
        duration: d.duration,
        pre: { targets: preElig.length, hits: preHits.length, acc: preAcc, avgRt: preAvgRt },
        post: { targets: postElig.length, hits: postHits.length, acc: postAcc, avgRt: postAvgRt },
      };
    });

    return {
      totalEligible,
      hits,
      misses,
      errors,
      accuracy,
      avgRt,
      impulsive,
      perDistraction,
    };
  }

  function _isAllowedShape(shape, rule) {
    if (rule === 'normal') return true;
    if (rule === 'onlySquares') return shape === 'square';
    if (rule === 'onlyCircles') return shape === 'circle';
    if (rule === 'onlyTriangles') return shape === 'triangle';
    return true;
  }

  function avg(list) {
    if (!list || !list.length) return null;
    return Math.round(list.reduce((a,b)=>a+b,0)/list.length);
  }

  // Helper to get RTs within a time window using taps mapped to their target spawn times
  state.rtListPerWindow = function(start, end) {
    const rts = [];
    state.taps.forEach(t => {
      if (!t.correct || t.targetId==null) return;
      const te = state.targetEvents.find(e=>e.id===t.targetId);
      if (!te) return;
      if (te.spawnTs>=start && te.spawnTs<end) {
        rts.push(t.ts - te.spawnTs);
      }
    });
    return rts;
  }

  function renderSummary(totals) {
    const acc = isFinite(totals.accuracy) ? totals.accuracy : 0;
    sumAcc.textContent = String(Math.round(acc));
    sumRt.textContent = totals.avgRt==null ? '-' : String(totals.avgRt);
    sumImp.textContent = String(totals.impulsive);

    // per-distraction table with opportunity counts shown
    const parts = totals.perDistraction.map(d => {
      const preAcc = d.pre.acc==null? '-' : Math.round(d.pre.acc*100)+'%';
      const postAcc = d.post.acc==null? '-' : Math.round(d.post.acc*100)+'%';
      const preRt = d.pre.avgRt==null? '-' : d.pre.avgRt+'ms';
      const postRt = d.post.avgRt==null? '-' : d.post.avgRt+'ms';
      const preCount = `(${d.pre.hits}/${d.pre.targets})`;
      const postCount = `(${d.post.hits}/${d.post.targets})`;
      return `<div class="pill">#${d.id+1} ${d.kind} at ${Math.round(d.startTs/1000)}s • pre: ${preAcc} ${preCount} / ${preRt} • post: ${postAcc} ${postCount} / ${postRt}</div>`;
    });
    distractionSummary.innerHTML = parts.join(' ');

    summaryEl.classList.remove('hidden');
  }

  async function submitResults(totals) {
    const payload = {
      session_id: sessionId,
      duration_ms: DURATION_MS,
  taps: state.taps,
  targets: state.targetEvents,
  distractions: state.distractions,
  rule_windows: state.ruleWindows,
      summary: totals,
      timestamp: new Date().toISOString(),
      version: 1,
    };
    const res = await fetch('/api/game/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      console.warn('submit failed', await res.text());
    }
  }

  arena.addEventListener('click', onArenaClick);
  startBtn.addEventListener('click', () => {
    const modal = document.getElementById('instructions');
    if (modal) modal.style.display = 'none';
    if (!state.running) startGame();
  });
  const startFromModal = document.getElementById('startFromModal');
  if (startFromModal) {
    startFromModal.addEventListener('click', () => {
      const modal = document.getElementById('instructions');
      if (modal) modal.style.display = 'none';
      if (!state.running) startGame();
    });
  }

  function pushOutcome(correct){
    state.recentOutcomes.push(!!correct);
    const MAX = 15;
    if (state.recentOutcomes.length > MAX) state.recentOutcomes.shift();
  }

  function adaptDifficulty(){
    const n = state.recentOutcomes.length;
    const hits = state.recentOutcomes.filter(Boolean).length;
    const hitRate = n ? hits/n : 0;
    const lastRt = state.rtList.slice(-8);
    const avgRt = lastRt.length ? lastRt.reduce((a,b)=>a+b,0)/lastRt.length : 1200;

    // If doing well: faster spawns, shorter TTL, more fakes
    if (hitRate >= 0.75 && avgRt < 950) {
      state.spawnInterval = Math.max(MIN_SPAWN_MS, state.spawnInterval - 60);
      state.targetTtl = Math.max(800, state.targetTtl - 50);
      state.fakeProb = Math.min(0.35, state.fakeProb + 0.03);
      state.pAllowed = Math.max(0.60, state.pAllowed - 0.05);
    }
    // If struggling: slow down, longer TTL, fewer fakes
    if (hitRate <= 0.45 || avgRt > 1250) {
      state.spawnInterval = Math.min(1800, state.spawnInterval + 80);
      state.targetTtl = Math.min(1800, state.targetTtl + 60);
      state.fakeProb = Math.max(0.10, state.fakeProb - 0.05);
      state.pAllowed = Math.min(0.90, state.pAllowed + 0.10);
    }
  }

  function moveTargets(){
    const rect = arena.getBoundingClientRect();
    state.targets.forEach((rec, id) => {
      // dt approximated per-frame; velocities are tiny so it's fine
      rec.x += rec.vx * 16; // ~16ms/frame
      rec.y += rec.vy * 16;

      const size = 40;
      if (rec.x < ARENA_PADDING) { rec.x = ARENA_PADDING; rec.vx *= -1; }
      if (rec.y < ARENA_PADDING) { rec.y = ARENA_PADDING; rec.vy *= -1; }
      if (rec.x > rect.width - size - ARENA_PADDING) { rec.x = rect.width - size - ARENA_PADDING; rec.vx *= -1; }
      if (rec.y > rect.height - size - ARENA_PADDING) { rec.y = rect.height - size - ARENA_PADDING; rec.vy *= -1; }

      const el = arena.querySelector(`.target[data-id="${id}"]`);
      if (el) { el.style.left = `${rec.x}px`; el.style.top = `${rec.y}px`; }
    });
  }

  function updateRuleBanner(){
    if (!ruleBanner) return;
    let label = 'Normal';
    if (state.rule === 'onlySquares') label = 'Only squares';
    if (state.rule === 'onlyCircles') label = 'Only circles';
    if (state.rule === 'onlyTriangles') label = 'Only triangles';
    ruleBanner.textContent = label;
  }
})();
