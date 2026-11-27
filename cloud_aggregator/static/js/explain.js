const METHODS = { saliency: 'Gradient saliency', smoothgrad: 'SmoothGrad', ig: 'Integrated Gradients', occlusion: 'Occlusion' };
const SEV = { edge01: '0.007 in', edge02: '0.014 in', edge03: '0.021 in' };
let sigChart, delChart, xaiChart;
const qs = new URLSearchParams(location.search);

/* draws the importance shading behind the signal */
const shade = { id: 'shade', beforeDatasetsDraw(chart) {
  const a = chart.$attr; if (!a) return; const { ctx, chartArea: ca, scales: { x } } = chart; const seg = a.seg, n = a.vals.length;
  const mx = Math.max(...a.vals) || 1; ctx.save();
  for (let i = 0; i < n; i++) { const x0 = x.getPixelForValue(i * seg), x1 = x.getPixelForValue((i + 1) * seg);
    ctx.fillStyle = `rgba(251,146,60,${0.75 * a.vals[i] / mx})`; ctx.fillRect(x0, ca.top, Math.max(x1 - x0, 1), ca.bottom - ca.top); }
  ctx.restore(); } };

function segs(attr, seg) { const out = []; for (let i = 0; i + seg <= attr.length; i += seg) { let s = 0; for (let j = i; j < i + seg; j++) s += attr[j]; out.push(s); } return out; }
function interp(xs, ys, x) { for (let i = 1; i < xs.length; i++) if (x <= xs[i]) { const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]); return ys[i - 1] + t * (ys[i] - ys[i - 1]); } return ys[ys.length - 1]; }

function render(d) {
  const e = d.edge, f = d.fp32, lab = LABELS;
  const agree = e.pred === f.pred;
  $('#decision').innerHTML = `
   <div class="mrow"><span>Ground truth</span><b>${lab[d.label]}</b></div>
   <div class="mrow"><span>Edge model (INT8, ${num(e.latency_ms, 3)} ms)</span><b style="color:${e.pred === d.label ? 'var(--ok)' : 'var(--fault)'}">${lab[e.pred]} · ${pct(e.probs[e.pred])}</b></div>
   <div class="mrow"><span>Cloud model (FP32, used for XAI)</span><b>${lab[f.pred]} · ${pct(f.probs[f.pred])}</b></div>
   <div class="probs">${lab.map((l, i) => `<div class="prow"><span>${l}</span><div class="bar"><div style="width:${e.probs[i] * 100}%;background:${CLS_COLOR[i]}"></div></div><span>${pct(e.probs[i], 0)}</span></div>`).join('')}</div>
   <div class="small ${agree ? 'muted' : ''}" style="${agree ? '' : 'color:var(--warn)'}">${agree ? 'INT8 and FP32 models agree — quantization did not change this decision.' : '⚠ INT8 and FP32 models disagree on this window (a quantization effect).'}</div>
   ${d.snr != null ? `<div class="small muted">Window corrupted with white noise at ${d.snr} dB SNR.</div>` : ''}`;
  const ex = d.explanation, seg = ex.seg, vals = segs(ex.attribution, seg);
  $('#xtitle').textContent = `— ${METHODS[ex.method]} · computed in ${num(ex.ms, 1)} ms for class “${lab[f.pred]}”`;
  const labels = d.window.map((_, i) => i);
  if (sigChart) sigChart.destroy();
  sigChart = mkChart('ch-sig', { type: 'line', plugins: [shade], data: { labels, datasets: [{ data: d.window, borderColor: '#60a5fa', borderWidth: 1, pointRadius: 0 }] },
    options: { plugins: { legend: { display: false }, tooltip: { enabled: false } }, scales: { x: { title: { display: true, text: 'sample (48 kHz)' }, ticks: { maxTicksLimit: 9 } }, y: { title: { display: true, text: 'normalised amplitude' } } } } });
  sigChart.$attr = { vals, seg };
  const dl = ex.deletion, fr = dl.fractions.map(v => v * 100);
  if (delChart) delChart.destroy();
  delChart = mkChart('ch-del', { type: 'line', data: { labels: fr, datasets: [{ label: 'remove most-important first', data: dl.guided, borderColor: '#fb923c', backgroundColor: '#fb923c', pointRadius: 2 }, { label: 'remove at random', data: dl.random, borderColor: '#8d99b3', borderDash: [5, 4], pointRadius: 0 }] },
    options: { plugins: { legend: { position: 'bottom' } }, scales: { y: { min: 0, max: 1, title: { display: true, text: 'confidence' } }, x: { title: { display: true, text: '% of signal removed' } } } } });
  const cross = (xs, ys) => { for (let i = 0; i < ys.length; i++) if (ys[i] < 0.5) return xs[i]; return null; };
  const gx = cross(dl.fractions, dl.guided), rx = cross(dl.fractions, dl.random);
  const fmt = v => v == null ? 'never' : Math.round(v * 100) + '%';
  let verdict;
  if (gx == null) verdict = 'The model stays confident even with everything removed — unusually redundant evidence, so this test cannot judge this window.';
  else if (rx == null || gx < rx - 0.05) verdict = 'The explanation points at parts of the signal the model really relies on.';
  else verdict = '<span style="color:var(--warn)">No better than removing random parts — treat this explanation with suspicion.</span>';
  $('#del-text').innerHTML = `Confidence falls below 50% after removing <b>${fmt(gx)}</b> of the signal (most-important first) versus <b>${fmt(rx)}</b> when removing at random. ${verdict}`;
}

async function run() {
  $('#err').textContent = 'computing…';
  try {
    const snr = $('#snr').value;
    const d = await post('/api/explain', { machine: $('#machine').value, idx: +$('#idx').value, method: $('#method').value, snr: snr === '' ? null : +snr });
    render(d); $('#err').textContent = '';
    history.replaceState(null, '', `?machine=${d.machine}&idx=${d.idx}&method=${$('#method').value}`);
  } catch (e) { $('#err').textContent = 'Could not explain: ' + e.message; }
}
async function randomWindow() {
  const d = await api(`/api/window_index?machine=${$('#machine').value}&cls=${$('#cls').value}`); $('#idx').value = d.idx; run();
}

async function loadReport() {
  const r = await api('/api/results/xai');
  if (!r.available) { notAvailable($('#xai-table'), 'python model/explainability.py'); return; }
  const rows = Object.entries(r.methods);
  $('#xai-table').innerHTML = `<table><thead><tr><th>Method</th><th class="n">Cost</th><th class="n">Deletion AUC ↓</th><th class="n">Confidence after 10% removed</th><th>vs random</th></tr></thead><tbody>` +
    rows.map(([k, m]) => { const better = m.deletion_auc < m.random_auc - .01;
      return `<tr><td>${m.label}</td><td class="n">${num(m.ms_mean, 1)} ms</td><td class="n">${num(m.deletion_auc, 3)}</td><td class="n">${pct(m.conf_after_10pct_removed, 0)}</td><td style="color:${better ? 'var(--ok)' : 'var(--fault)'}">${better ? '✔ faithful' : '✘ not better than random'}</td></tr>`; }).join('') +
    `<tr><td><i>random removal</i></td><td class="n">–</td><td class="n">${num(rows[0][1].random_auc, 3)}</td><td class="n">${pct(rows[0][1].random_conf_after_10pct_removed, 0)}</td><td></td></tr></tbody></table>
    <div class="muted small" style="margin-top:6px">Lower AUC = confidence collapses faster when the “important” parts are removed = more faithful.</div>`;
  const cols = ['#60a5fa', '#2dd4bf', '#fb923c', '#a78bfa'];
  xaiChart = mkChart('ch-xai', { type: 'line', data: { labels: r.fractions.map(v => v * 100), datasets: rows.map(([k, m], i) => ({ label: m.label, data: m.curve, borderColor: cols[i], pointRadius: 0, borderWidth: 2 })).concat([{ label: 'random', data: r.random_curve, borderColor: '#8d99b3', borderDash: [5, 4], pointRadius: 0 }]) },
    options: { plugins: { legend: { position: 'bottom' } }, scales: { y: { min: 0, max: 1, title: { display: true, text: 'mean confidence' } }, x: { title: { display: true, text: '% of signal removed' } } } } });
  const ic = r.impulse_correlation || {}, bad = rows.filter(([k, m]) => m.deletion_auc >= m.random_auc - .01).map(([k, m]) => m.label);
  $('#xai-notes').innerHTML = `<div class="callout ${bad.length ? 'warn' : ''}"><b>Finding.</b> ${bad.length ? `${bad.join(' and ')} failed the deletion test — they highlight where the output is <i>sensitive</i>, not where the evidence is, so they look plausible but are no better than random. ` : ''}
   Integrated Gradients and occlusion pass. (Occlusion is nearly the deletion metric itself, so its win is partly by construction.)</div>
   <div class="callout"><b>Physical sanity check.</b> Integrated-Gradients importance correlates with the local impulse amplitude of the signal (Spearman ρ): ${CLS.map((c, i) => ic[c] != null ? `${LABELS[i]} <b>${num(ic[c], 2)}</b>` : '').filter(Boolean).join(' · ')}. Bearing faults produce repetitive impulses, and the model attends to them — most strongly for the fault classes.</div>`;
}

document.addEventListener('DOMContentLoaded', async () => {
  MACHINES.forEach(m => $('#machine').add(new Option(`${m} (${SEV[m]})`, m)));
  CLS.forEach((c, i) => $('#cls').add(new Option(LABELS[i], c)));
  Object.entries(METHODS).forEach(([k, v]) => $('#method').add(new Option(v, k)));
  $('#method').value = qs.get('method') || 'ig'; if (qs.get('machine')) $('#machine').value = qs.get('machine');
  $('#idx').value = qs.get('idx') || 0;
  $('#go').onclick = run; $('#rand').onclick = randomWindow;
  loadReport().catch(e => console.error(e));
  if (qs.get('idx')) run(); else randomWindow();
});
