const built = {};
let kurtChart, latChart;
const MCOLORS = ['#60a5fa', '#2dd4bf', '#fbbf24'];
const SCENARIOS = [['auto', 'Auto story (healthy → fault)'], ['normal', 'Force: healthy'], ['inner_race_fault', 'Force: inner-race fault'],
  ['outer_race_fault', 'Force: outer-race fault'], ['ball_fault', 'Force: ball fault']];
const SNRS = [['off', 'Sensor noise: off'], ['30', 'Noise 30 dB'], ['20', 'Noise 20 dB'], ['10', 'Noise 10 dB'], ['5', 'Noise 5 dB']];

function cmd(mid, body) { return post('/api/command', Object.assign({ machine_id: mid }, body)).catch(e => alert('Command failed: ' + e.message)); }
function fleetCmd(body) { MACHINES.forEach(m => cmd(m, body)); }

function buildCard(mid, sev) {
  const el = document.createElement('div'); el.className = 'card machine'; el.id = 'm-' + mid;
  el.innerHTML = `
  <div class="mhead"><div><h3>${mid}</h3><span class="muted small">defect size ${sev}</span></div><span class="badge b-off" data-f="badge">WAITING</span></div>
  <div class="pred" data-f="pred">–</div>
  <div class="confbar"><div data-f="conf" style="width:0"></div></div><div class="muted small" data-f="conftxt">confidence –</div>
  <div class="probs">${LABELS.map((l, i) => `<div class="prow"><span>${l}</span><div class="bar"><div data-f="p${i}" style="width:0;background:${CLS_COLOR[i]}"></div></div><span data-f="pt${i}">–</span></div>`).join('')}</div>
  <div class="timeline" data-f="tl"></div><div class="muted small">last 40 windows: <span style="color:var(--ok)">■</span> ok <span style="color:var(--fault)">■</span> fault <span style="color:var(--warn)">■</span> uncertain</div>
  <div class="mrow"><span>Latency</span><b data-f="lat">–</b></div>
  <div class="mrow"><span>Kurtosis / crest factor</span><b data-f="kurt">–</b></div>
  <div class="mrow"><span>Live accuracy · windows</span><b data-f="acc">–</b></div>
  <div class="mrow"><span>Process memory (RSS)</span><b data-f="rss">–</b></div>
  <div class="mrow"><span>Model · integrity (SHA-256)</span><b data-f="model" class="mono">–</b></div>
  <div class="mrow"><span>On-device alarm</span><b data-f="local">–</b></div>
  <div class="mrow"><span>Store-and-forward / suppressed</span><b data-f="sf">–</b></div>
  <div class="controls">
    <div style="grid-column:1/3"><select data-f="scen">${SCENARIOS.map(s => `<option value="${s[0]}">${s[1]}</option>`).join('')}</select></div>
    <select data-f="model-sel"><option value="full">Model: full INT8</option><option value="pruned">Model: pruned + INT8</option></select>
    <select data-f="snr">${SNRS.map(s => `<option value="${s[0]}">${s[1]}</option>`).join('')}</select>
    <select data-f="report"><option value="all">Report: every window</option><option value="on_change">Report: on change only</option></select>
    <button class="ghost" data-f="uplink">✂ Cut uplink</button>
    <button class="ghost" data-f="pause">⏸ Pause</button>
    <a class="btn ghost" data-f="explain" href="/explain?machine=${mid}">🔍 Explain</a>
  </div>`;
  const q = f => el.querySelector(`[data-f="${f}"]`);
  q('scen').onchange = e => cmd(mid, { scenario: e.target.value });
  q('model-sel').onchange = e => cmd(mid, { model: e.target.value });
  q('snr').onchange = e => cmd(mid, { snr: e.target.value === 'off' ? null : Number(e.target.value) });
  q('report').onchange = e => cmd(mid, { report: e.target.value });
  q('uplink').onclick = () => { const down = q('uplink').dataset.down === '1'; cmd(mid, { uplink: down }); };
  q('pause').onclick = () => { const p = q('pause').dataset.paused === '1'; cmd(mid, { paused: !p }); };
  built[mid] = { el, q };
  $('#machines').appendChild(el);
}

function badge(m) {
  if (!m.seen) return ['WAITING', 'b-off'];
  if (!m.online) return [m.paused ? 'PAUSED' : (m.commanded && m.commanded.uplink === false ? 'UPLINK CUT' : 'OFFLINE'), 'b-off'];
  if (m.alarm) return ['ALARM', 'b-alarm'];
  const s = m.latest.status;
  return s === 'OK' ? ['HEALTHY', 'b-ok'] : s === 'FAULT' ? ['FAULT', 'b-fault'] : ['UNCERTAIN', 'b-warn'];
}

function updateCard(m) {
  const c = built[m.machine_id]; if (!c) return; const q = c.q;
  const [bt, bc] = badge(m); const b = q('badge'); b.textContent = bt; b.className = 'badge ' + bc;
  if (!m.seen) return;
  const L = m.latest;
  q('pred').textContent = LABELS[L.class_id]; q('pred').style.color = L.class_id === 0 ? 'var(--ok)' : CLS_COLOR[L.class_id];
  q('conf').style.width = (L.confidence * 100) + '%'; q('conftxt').textContent = 'confidence ' + pct(L.confidence) + (L.snr != null ? '  ·  sensor noise ' + L.snr + ' dB' : '');
  (L.probs || []).forEach((p, i) => { q('p' + i).style.width = (p * 100) + '%'; q('pt' + i).textContent = pct(p, 0); });
  const h = m.history.slice(-40); const tl = q('tl');
  while (tl.children.length < 40) tl.appendChild(document.createElement('i'));
  [...tl.children].forEach((i, k) => { const x = h[k - (40 - h.length)]; i.style.background = !x ? '' : x.s === 'OK' ? 'var(--ok)' : x.s === 'FAULT' ? 'var(--fault)' : 'var(--warn)'; });
  q('lat').textContent = num(L.latency_ms, 3) + ' ms (mean ' + num(m.mean_latency_ms, 3) + ')';
  q('kurt').textContent = num(L.kurtosis, 1) + ' / ' + num(L.crest_factor, 1);
  q('acc').textContent = pct(m.live_acc) + ' · ' + m.windows;
  q('rss').textContent = num(L.rss_mb, 0) + ' MB (incl. TF runtime)';
  q('model').textContent = (L.model_kind || '–') + ' · ' + (L.model_sha || '–');
  q('local').innerHTML = L.local_alarm ? '<span style="color:var(--fault)">RAISED (on device)</span>' : 'clear';
  q('sf').textContent = 'backlog ' + (L.backlog || 0) + ' · flushed ' + m.delayed + ' · suppressed ' + m.suppressed;
  const cm = m.commanded || {};
  const fo = document.activeElement;
  if (cm.scenario && fo !== q('scen')) q('scen').value = cm.scenario;
  if (cm.model && fo !== q('model-sel')) q('model-sel').value = cm.model;
  if (cm.report && fo !== q('report')) q('report').value = cm.report;
  const down = cm.uplink === false; q('uplink').dataset.down = down ? '1' : '0'; q('uplink').textContent = down ? '🔌 Restore uplink' : '✂ Cut uplink';
  q('pause').dataset.paused = m.paused ? '1' : '0'; q('pause').textContent = m.paused ? '▶ Resume' : '⏸ Pause';
  q('explain').href = `/explain?machine=${m.machine_id}&idx=${L.window_idx}`;
}

function setupCharts() {
  const mk = (id, y) => mkChart(id, { type: 'line', data: { labels: [], datasets: MACHINES.map((m, i) => ({ label: m, data: [], borderColor: MCOLORS[i], backgroundColor: MCOLORS[i], borderWidth: 1.6, pointRadius: 0, tension: .25 })) },
    options: { scales: { y: { title: { display: true, text: y } }, x: { display: false } }, plugins: { legend: { position: 'bottom' } } } });
  kurtChart = mk('ch-kurt', 'kurtosis'); latChart = mk('ch-lat', 'ms');
}
function updateCharts(machines) {
  [[kurtChart, h => h.k], [latChart, h => h.lat]].forEach(([ch, fn]) => {
    if (!ch) return; const n = 60; ch.data.labels = Array.from({ length: n }, (_, i) => i);
    MACHINES.forEach((mid, i) => { const hs = (machines[mid].history || []); const arr = hs.map(fn); ch.data.datasets[i].data = Array(Math.max(0, n - arr.length)).fill(null).concat(arr); });
    ch.update('none');
  });
}

async function refresh() {
  let d; try { d = await api('/api/status'); } catch (e) { return; }
  const k = d.kpi;
  $('#k-online').textContent = k.online + ' / ' + k.total; $('#k-online-d').textContent = k.online ? 'reporting live' : 'start the edge nodes';
  $('#k-alarms').textContent = k.alarms; $('#k-alarms').style.color = k.alarms ? 'var(--fault)' : '';
  $('#k-win').textContent = k.windows; $('#k-win-d').textContent = k.messages + ' msgs sent' + (k.suppressed ? ' · ' + k.suppressed + ' suppressed' : '');
  $('#k-lat').textContent = k.mean_latency_ms == null ? '–' : num(k.mean_latency_ms, 3) + ' ms';
  $('#k-rt').textContent = k.mean_latency_ms ? `window = ${num(k.window_ms, 1)} ms of signal → ${Math.round(k.window_ms / k.mean_latency_ms)}× faster than real time` : '';
  const kb = x => x >= 1000 ? (x / 1000).toFixed(1) + ' KB/s' : Math.round(x) + ' B/s';
  $('#k-bw').textContent = k.raw_Bps ? '−' + num((1 - k.uplink_Bps / k.raw_Bps) * 100, 1) + '%' : '–';
  $('#k-bw-d').textContent = k.raw_Bps ? `${kb(k.uplink_Bps)} sent vs ${kb(k.raw_Bps)} to stream the raw 48 kHz sensor` : 'start the edge nodes';
  $('#k-acc').textContent = pct(k.live_acc);
  MACHINES.forEach(m => { if (!built[m]) buildCard(m, d.machines[m].severity); updateCard(d.machines[m]); });
  const alarms = MACHINES.map(m => d.machines[m]).filter(m => m.alarm);
  const bn = $('#banner'); bn.classList.toggle('show', alarms.length > 0);
  bn.innerHTML = alarms.map(m => `⚠ ${m.machine_id}: ${LABELS[m.alarm_class] || 'fault'} detected on-device (${Math.round(d.server_time - m.alarm_since)} s ago)`).join(' &nbsp;·&nbsp; ');
  updateCharts(d.machines);
  if (d.events.length) $('#payload').textContent = JSON.stringify(Object.fromEntries(Object.entries(d.events[0]).filter(([k]) => k !== 'true_label')), null, 1);
  $('#events').innerHTML = d.events.slice(0, 14).map(e => `<tr><td>${fmtTime(e.timestamp)}</td><td>${e.machine_id}</td><td>${LABELS[e.class_id]}</td>
    <td><span class="badge ${e.status === 'OK' ? 'b-ok' : e.status === 'FAULT' ? 'b-fault' : 'b-warn'}">${e.status}</span>${e.delayed ? ' <span class="tag">delayed</span>' : ''}</td>
    <td class="n">${pct(e.confidence)}</td><td class="n">${num(e.kurtosis, 1)}</td><td class="n">${num(e.latency_ms, 3)} ms</td><td>${e.model_kind || ''}</td>
    <td><a href="/explain?machine=${e.machine_id}&idx=${e.window_idx}">explain →</a></td></tr>`).join('');
}
document.addEventListener('DOMContentLoaded', () => { setupCharts(); refresh(); setInterval(refresh, 1500); });
