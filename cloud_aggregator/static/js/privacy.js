const ALPHAS = [1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128, 256];
function rdpEps(sigma, T, delta) { if (!(sigma > 0)) return Infinity; let best = Infinity;
  for (const a of ALPHAS) { const e = T * a / (2 * sigma * sigma) + Math.log((a - 1) / a) - (Math.log(delta) + Math.log(a)) / (a - 1); best = Math.min(best, e); } return Math.max(best, 0); }
function calc() {
  const s = +$('#c-sigma').value, T = +$('#c-T').value, d = +$('#c-delta').value, K = +$('#c-K').value, D = 11748;
  const eps = rdpEps(s, T, d), noisePerCoord = s / K, ratio = (s / K) * Math.sqrt(D);
  $('#c-out').innerHTML = `<div class="grid g3"><div class="card kpi flat"><div class="l">ε after ${T} rounds</div><div class="v">${isFinite(eps) ? eps.toFixed(1) : '∞'}</div><div class="d">${eps < 1 ? 'strong' : eps < 10 ? 'reasonable' : eps < 50 ? 'weak' : 'practically meaningless'} guarantee</div></div>
   <div class="card kpi flat"><div class="l">noise per weight (on the average)</div><div class="v">${noisePerCoord.toFixed(4)}×C</div><div class="d">σ ÷ K, in units of the clip norm C</div></div>
   <div class="card kpi flat"><div class="l">total noise ÷ one update</div><div class="v">${ratio.toFixed(1)}×</div><div class="d">σ·√d ÷ K with d = ${D} weights; ≳1 drowns the signal</div></div></div>`;
}
async function main() {
  ['sigma', 'T', 'delta', 'K'].forEach(k => $('#c-' + k).oninput = calc); calc();
  try { const s = await api('/api/status');
    if (s.events.length) { const e = Object.fromEntries(Object.entries(s.events[0]).filter(([k]) => k !== 'true_label')); $('#payload').textContent = JSON.stringify(e, null, 1); }
    const k = s.kpi; const msg = k.avg_msg_bytes || 500, raw = k.raw_window_bytes;
    $('#bw').innerHTML = `<div class="callout">A raw window is <b>${raw} bytes</b> (1024 × 16-bit samples); one status message is about <b>${Math.round(msg)} bytes</b>. Reporting every window already cuts traffic by ${((1 - msg / raw) * 100).toFixed(0)}% — and a real node classifying 47 windows/s would report once a second or on change, a >99% cut. Raw streaming would also expose the machine's full vibration signature.</div>`; } catch (e) {}
  const r = await api('/api/results/dp');
  if (!r.available || !r.rows.length) { notAvailable($('#dp-table'), 'python federated/compare_dp.py'); return; }
  const Ks = [...new Set(r.rows.map(x => x.K))].sort((a, b) => a - b), cols = ['#f87171', '#fbbf24', '#2dd4bf'];
  const ds = [];
  Ks.forEach((K, i) => { const rr = r.rows.filter(x => x.K === K && x.eps != null).sort((a, b) => a.eps - b.eps);
    ds.push({ label: `${K} clients`, data: rr.map(x => ({ x: x.eps, y: x.accs.reduce((a, b) => a + b, 0) / x.accs.length * 100 })), borderColor: cols[i % 3], backgroundColor: cols[i % 3], showLine: true, tension: .2, pointRadius: 5 });
    const base = r.rows.find(x => x.K === K && x.eps == null); if (base) ds.push({ label: `${K} clients, no noise`, data: [{ x: 1, y: base.accs.reduce((a, b) => a + b, 0) / base.accs.length * 100 }, { x: 1e6, y: base.accs.reduce((a, b) => a + b, 0) / base.accs.length * 100 }], borderColor: cols[i % 3], borderDash: [3, 4], showLine: true, pointRadius: 0, borderWidth: 1 }); });
  mkChart('ch-dp', { type: 'scatter', data: { datasets: ds }, options: { plugins: { legend: { position: 'bottom', labels: { filter: i => !i.text.includes('no noise') } } }, scales: { x: { type: 'logarithmic', min: 1, max: 1e6, title: { display: true, text: 'ε (log scale) — lower is more private' } }, y: { min: 0, max: 100, title: { display: true, text: 'accuracy %' } } } } });
  $('#dp-table').innerHTML = `<table><thead><tr><th class="n">Clients</th><th class="n">σ</th><th class="n">ε</th><th class="n">Accuracy</th></tr></thead><tbody>` +
    r.rows.map(x => { const a = x.accs.reduce((p, q) => p + q, 0) / x.accs.length; return `<tr><td class="n">${x.K}</td><td class="n">${x.sigma || 'clip only'}</td><td class="n">${x.eps == null ? '∞' : x.eps.toFixed(0)}</td><td class="n">${pct(a)}${x.accs.length > 1 ? ' (n=' + x.accs.length + ')' : ''}</td></tr>`; }).join('') + `</tbody></table>`;
  const best = r.rows.filter(x => x.eps != null && x.eps < 100).map(x => ({ ...x, a: x.accs.reduce((p, q) => p + q, 0) / x.accs.length })).sort((a, b) => b.a - a.a)[0];
  $('#dp-verdict').innerHTML = `<div class="callout warn"><b>Honest reading.</b> DP noise is added to the <i>sum</i> of client updates and divided by the number of clients, so it fades as 1/K — but this model has ~12 000 trainable weights, so the total noise grows with √d. With only 3 clients there is <b>no</b> setting that is both accurate and private. More clients help a lot (compare the curves), yet even then ${best ? `the best accuracy at ε &lt; 100 is ${pct(best.a, 0)} (K=${best.K}, σ=${best.sigma}, ε≈${best.eps.toFixed(0)})` : 'accuracy collapses before ε reaches a meaningful value'} — ε in the tens or hundreds is a weak guarantee. Single-digit ε needs hundreds-to-thousands of clients, as in real cross-device FL. We report this rather than hide it.</div>`;
}
document.addEventListener('DOMContentLoaded', () => main().catch(e => console.error(e)));
