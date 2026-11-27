function faWidget() {
  const d = [[1600, 0.60], [800, 0.40], [2400, 0.80]];
  $('#fa-inputs').innerHTML = d.map((v, i) => `<div class="card flat"><b>edge0${i + 1}</b>
    <span class="lbl">local training windows</span><input type="number" id="fa-n${i}" value="${v[0]}" min="1">
    <span class="lbl">weight after local training</span><input type="number" id="fa-w${i}" value="${v[1]}" step="0.05"></div>`).join('');
  const upd = () => { const n = [0, 1, 2].map(i => +$('#fa-n' + i).value || 0), w = [0, 1, 2].map(i => +$('#fa-w' + i).value || 0);
    const tot = n.reduce((a, b) => a + b, 0), num_ = n.reduce((a, x, i) => a + x * w[i], 0);
    $('#fa-out').textContent = `global_w = Σ nᵢ·wᵢ / Σ nᵢ\n         = (${n.map((x, i) => `${x}×${w[i]}`).join(' + ')}) / ${tot}\n         = ${n.map((x, i) => (x * w[i]).toFixed(1)).join(' + ')}  /  ${tot}\n         = ${(num_ / tot).toFixed(4)}\n\nplain (unweighted) mean would be ${(w.reduce((a, b) => a + b, 0) / 3).toFixed(4)}`; };
  $$('#fa-inputs input').forEach(e => e.oninput = upd); upd();
}

function band(label, color, mean, std) {
  const lo = mean.map((m, i) => m - std[i]), hi = mean.map((m, i) => m + std[i]);
  return [{ label: label + ' −σ', data: lo, borderWidth: 0, pointRadius: 0, backgroundColor: color + '33', fill: false, hidden: false, _hideLegend: true },
          { label: label + ' +σ', data: hi, borderWidth: 0, pointRadius: 0, backgroundColor: color + '33', fill: '-1', _hideLegend: true },
          { label, data: mean, borderColor: color, backgroundColor: color, borderWidth: 2.2, pointRadius: 2 }];
}

async function main() {
  faWidget();
  const [fl, off, gen] = await Promise.all([api('/api/results/fl'), api('/api/results/official'), api('/api/results/generalization')]);
  if (fl.available) {
    const S = fl.summary, runs = Object.values(fl.runs), R = runs[0].fedavg.curve.length;
    const mean = Array.from({ length: R }, (_, r) => runs.reduce((a, x) => a + x.fedavg.curve[r], 0) / runs.length);
    const std = mean.map((m, r) => Math.sqrt(runs.reduce((a, x) => a + (x.fedavg.curve[r] - m) ** 2, 0) / Math.max(runs.length - 1, 1)));
    const lines = (v, label, c) => ({ label, data: Array(R).fill(v), borderColor: c, borderDash: [6, 4], pointRadius: 0, borderWidth: 1.5 });
    mkChart('ch-conv', { type: 'line', data: { labels: mean.map((_, i) => i + 1), datasets: [...band('FedAvg', '#60a5fa', mean, std), lines(S.centralized.acc.mean, 'Centralized (pooled data — no privacy)', '#34d399'), lines(S.local_only.acc.mean, 'Local-only average', '#f87171')] },
      options: { plugins: { legend: { position: 'bottom', labels: { filter: i => !i.text.includes('σ') } } }, scales: { y: { min: 0.2, max: 1, title: { display: true, text: 'test accuracy' } }, x: { title: { display: true, text: 'federated round' } } } } });
    $('#conv-sub').textContent = `Shaded band = ± 1 std across ${S.n_seeds} seeds. Local clients: ${fl.config.local_epochs} epochs/round, cosine LR decay.`;
    const rows = [['Local-only (no federation)', S.local_only, '#f87171'], ['FedAvg (3 nodes)', S.fedavg, '#60a5fa'], ['Centralized (pooled)', S.centralized, '#34d399']];
    mkChart('ch-cmp', { type: 'bar', data: { labels: rows.map(r => r[0]), datasets: [{ data: rows.map(r => r[1].acc.mean * 100), backgroundColor: rows.map(r => r[2]) }] },
      options: { indexAxis: 'y', plugins: { legend: { display: false } }, scales: { x: { min: 0, max: 100, title: { display: true, text: 'accuracy %' } } } } });
    $('#cmp-table').innerHTML = `<table><thead><tr><th></th><th class="n">Accuracy</th><th class="n">Macro-F1</th><th>Privacy</th></tr></thead><tbody>` +
      rows.map((r, i) => `<tr><td>${r[0]}</td><td class="n">${msd(r[1].acc)}</td><td class="n">${msd(r[1].f1)}</td><td>${i === 2 ? '✘ raw data pooled' : '✔ raw data stays local'}</td></tr>`).join('') + '</tbody></table>';
    const gapClosed = (S.fedavg.acc.mean - S.local_only.acc.mean) / (S.centralized.acc.mean - S.local_only.acc.mean);
    $('#headline').innerHTML = [
      ['FedAvg accuracy', msd(S.fedavg.acc), 'mean ± std, ' + S.n_seeds + ' seeds'],
      ['Gain over training alone', '+' + ((S.fedavg.acc.mean - S.local_only.acc.mean) * 100).toFixed(1) + ' pts', 'same data, but shared as weights'],
      ['Gap to pooled data', '−' + ((S.centralized.acc.mean - S.fedavg.acc.mean) * 100).toFixed(1) + ' pts', 'the price of keeping data private'],
      ['Of the possible gain captured', (gapClosed * 100).toFixed(0) + '%', 'between local-only and centralized']
    ].map(h => `<div class="card kpi"><div class="l">${h[0]}</div><div class="v">${h[1]}</div><div class="d">${h[2]}</div></div>`).join('');
    const own = S.local_only.acc_on_own_data, fleet = S.local_only.acc_per_node_on_fleet, fed = S.fedavg_per_node;
    mkChart('ch-node', { type: 'bar', data: { labels: MACHINES, datasets: [
      { label: 'Local model, on its own machine', data: MACHINES.map(m => own[m].mean * 100), backgroundColor: '#fbbf24' },
      { label: 'Local model, on the whole fleet', data: MACHINES.map(m => fleet[m].mean * 100), backgroundColor: '#f87171' },
      { label: 'FedAvg model, on that machine', data: MACHINES.map(m => fed[m].mean * 100), backgroundColor: '#60a5fa' }] },
      options: { plugins: { legend: { position: 'bottom' } }, scales: { y: { min: 0, max: 100, title: { display: true, text: 'accuracy %' } } } } });
  } else {
    ['#headline', '#cmp-table', '#conv-sub'].forEach(s => notAvailable($(s), 'python experiments/fl_comparison.py'));
  }
  if (off.available) {
    const cm = off.confusion_matrix, mx = Math.max(...cm.flat());
    $('#cm').innerHTML = `<table class="cm"><tr><th>true ↓ / pred →</th>${LABELS.map(l => `<th>${l.replace(' fault', '')}</th>`).join('')}</tr>` +
      cm.map((row, i) => `<tr><th>${LABELS[i].replace(' fault', '')}</th>${row.map((v, j) => `<td style="background:rgba(${i === j ? '52,211,153' : '248,113,113'},${Math.min(.85, v / mx + (v ? .08 : 0))})">${v}</td>`).join('')}</tr>`).join('') + '</table>';
    $('#perclass').innerHTML = CLS.map((c, i) => `${LABELS[i]}: recall <b>${pct(off.per_class[c].recall, 0)}</b>, precision <b>${pct(off.per_class[c].precision, 0)}</b>`).join('<br>') + `<br><span class="muted">Single official run (seed ${off.seed}); accuracy ${pct(off.acc)}, macro-F1 ${pct(off.macro_f1)}.</span>`;
  }
  if (gen.available && gen.summary) {
    const SV = { edge01: '0.007 in', edge02: '0.014 in', edge03: '0.021 in' };
    $('#gen').innerHTML = `<table><thead><tr><th>Held-out machine (never seen)</th><th class="n">FedAvg</th><th class="n">Centralized</th></tr></thead><tbody>` +
      MACHINES.map(m => `<tr><td>${m} — defect ${SV[m]}</td><td class="n">${msd(gen.summary[m].fedavg)}</td><td class="n">${msd(gen.summary[m].centralized)}</td></tr>`).join('') + `</tbody></table>
      <div class="callout warn" style="margin-top:10px">Accuracy on an unseen defect size is far below the standard-split number. A model trained on two severities cannot be trusted blindly on a third — a real deployment must keep training on new machines (which is what federated learning is for).</div>`;
  } else notAvailable($('#gen'), 'python experiments/generalization.py');
}
document.addEventListener('DOMContentLoaded', () => main().catch(e => console.error(e)));
