const DEVICES = [['Arduino Nano 33 BLE (Cortex-M4F, 64 MHz)', 1024, 256, 64], ['STM32F407 (Cortex-M4F, 168 MHz)', 1024, 192, 168], ['ESP32 (Xtensa LX6, 240 MHz)', 4096, 520, 240], ['Raspberry Pi Zero 2 W (Cortex-A53, 1 GHz)', 1e6, 5e5, 1000]];
const OVERHEAD_KB = 20;
function fitTable(S, name) {
  const v = S[name];
  $('#fit').innerHTML = `<table><thead><tr><th>Device</th><th class="n">Flash needed / have</th><th class="n">RAM needed / have</th><th class="n">Est. time / window</th><th>Fits?</th></tr></thead><tbody>` +
    DEVICES.map(d => { const ram = v.ram_kb + OVERHEAD_KB, ok = v.size_kb < d[1] && ram < d[2], ms_ = v.macs / (d[3] * 1e6) * 1000;
      return `<tr><td>${d[0]}</td><td class="n">${num(v.size_kb, 1)} / ${d[1] >= 1e6 ? '16 GB+' : d[1] + ' KB'}</td><td class="n">${num(ram, 0)} / ${d[2] >= 1e5 ? '512 MB' : d[2] + ' KB'}</td><td class="n">${ms_ < 1 ? '<1' : num(ms_, 0)} ms</td><td class="${ok ? 'fit-ok' : 'fit-no'}">${ok ? '✔ fits' : '✘ too big'}</td></tr>`; }).join('') +
    `</tbody></table><div class="small muted">A 1024-sample window is 21.3 ms of signal, so “est. time” below ~21 ms keeps up with real time.</div>`;
}
async function main() {
  const [c, b] = await Promise.all([api('/api/results/compression'), api('/api/results/benchmark')]);
  if (c.available && c.summary) {
    const S = c.summary, names = Object.keys(S);
    const base = S['FP32 baseline'], q = S['INT8 (quantized)'], p = S['Pruned 50% + INT8'], ctl = S['FP32 + FL fine-tune (control)'], p50 = S['Pruned 50% (FP32)'];
    $('#headline').innerHTML = [
      ['Model size', `${num(base.size_kb, 0)} → ${num(p.size_kb, 0)} KB`, `${((1 - p.size_kb / base.size_kb) * 100).toFixed(0)}% smaller (pruned 50% + INT8)`],
      ['Parameters', `${base.params} → ${p.params}`, `${((1 - p.params / base.params) * 100).toFixed(0)}% fewer`],
      ['Compute / window', `${(base.macs / 1e3).toFixed(0)}K → ${(p.macs / 1e3).toFixed(0)}K MACs`, `${(base.macs / p.macs).toFixed(1)}× fewer multiply-accumulates`],
      ['Accuracy', msd(p.acc), `baseline ${msd(base.acc)}`]].map(h => `<div class="card kpi"><div class="l">${h[0]}</div><div class="v" style="font-size:20px">${h[1]}</div><div class="d">${h[2]}</div></div>`).join('');
    $('#tbl').innerHTML = `<table><thead><tr><th>Variant</th><th class="n">Size KB</th><th class="n">Params</th><th class="n">MACs</th><th class="n">Peak RAM KB</th><th class="n">Accuracy</th><th class="n">Macro-F1</th>${c.snrs.map(s => `<th class="n">@${s} dB</th>`).join('')}</tr></thead><tbody>` +
      names.map(n => { const v = S[n]; return `<tr><td>${n}</td><td class="n">${num(v.size_kb, 1)}</td><td class="n">${v.params}</td><td class="n">${(v.macs / 1e3).toFixed(0)}K</td><td class="n">${num(v.ram_kb, 1)}</td><td class="n">${msd(v.acc)}</td><td class="n">${msd(v.f1)}</td>${c.snrs.map(s => `<td class="n">${pct(v.robustness[s].mean, 0)}</td>`).join('')}</tr>`; }).join('') +
      `</tbody></table><div class="muted small" style="margin-top:6px">${Object.keys(c.runs).length} seeds. Sizes are TFLite file sizes (FP32 or INT8); “Peak RAM” is the largest single-layer activation footprint at 4 B (FP32) or 1 B (INT8) per element.</div>`;
    const cols = n => n.includes('Pruned') ? '#2dd4bf' : n.includes('INT8') ? '#60a5fa' : n.includes('control') ? '#fbbf24' : '#a78bfa';
    mkChart('ch-pareto', { type: 'scatter', data: { datasets: names.map(n => ({ label: n, data: [{ x: S[n].size_kb, y: S[n].acc.mean * 100 }], backgroundColor: cols(n), pointRadius: 7 })) },
      options: { plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.parsed.x.toFixed(1)} KB, ${ctx.parsed.y.toFixed(1)}%` } } }, scales: { x: { title: { display: true, text: 'size on disk (KB)' } }, y: { title: { display: true, text: 'accuracy %' } } } } });
    const pick = ['FP32 baseline', 'INT8 (quantized)', 'Pruned 50% + INT8'];
    const cc = ['#a78bfa', '#60a5fa', '#2dd4bf'];
    mkChart('ch-noise', { type: 'line', data: { labels: ['clean', ...c.snrs.map(s => s + ' dB')].reverse().reverse(), datasets: pick.map((n, i) => ({ label: n, data: [S[n].acc.mean * 100, ...c.snrs.map(s => S[n].robustness[s].mean * 100)], borderColor: cc[i], backgroundColor: cc[i], tension: .2 })) },
      options: { plugins: { legend: { position: 'bottom' } }, scales: { y: { min: 0, max: 100, title: { display: true, text: 'accuracy %' } } } } });
    const dq = (q.acc.mean - base.acc.mean) * 100, dp = (p50.acc.mean - ctl.acc.mean) * 100;
    const noiseDrop = (base.acc.mean - base.robustness['5'].mean) * 100;
    $('#verdict').innerHTML = `<div class="callout"><b>Quantization</b> costs ${dq.toFixed(1)} pts of accuracy for a ${((1 - q.size_kb / base.size_kb) * 100).toFixed(0)}% size cut — essentially free.</div>
      <div class="callout ${Math.abs(dp) > 2 ? '' : 'warn'}"><b>Pruning, tested fairly.</b> Pruned-50% (FP32) reaches ${msd(p50.acc)} against ${msd(ctl.acc)} for the unpruned model given the <i>same</i> extra federated rounds: a difference of ${dp >= 0 ? '+' : ''}${dp.toFixed(1)} pts. ${Math.abs(dp) <= 2 ? 'So pruning 72% of the parameters is essentially accuracy-neutral here — but it does <b>not</b> improve accuracy; an earlier version of this project appeared to show that only because the pruned model got extra centralized training the baseline never received.' : ''}</div>
      <div class="callout bad"><b>Noise is the weak spot.</b> At 5 dB SNR the baseline falls by ${noiseDrop.toFixed(0)} points (to ${pct(base.robustness['5'].mean, 0)}). The model was trained only on clean lab recordings. Training with noise augmentation is the obvious next step before trusting it on a real factory floor.</div>`;
    const sel = $('#fit-variant'); names.forEach(n => sel.add(new Option(n, n))); sel.value = 'Pruned 50% + INT8';
    sel.onchange = () => fitTable(S, sel.value); fitTable(S, sel.value);
  } else { ['#tbl', '#headline'].forEach(s => notAvailable($(s), 'python experiments/compression_study.py')); }
  if (b.available) {
    $('#bench').innerHTML = `<table><thead><tr><th>Variant</th><th class="n">Size KB</th><th class="n">Params</th><th class="n">MACs</th><th class="n">Latency mean / p95</th><th class="n">Accuracy</th><th class="n">Macro-F1</th></tr></thead><tbody>` +
      b.rows.map(r => `<tr><td>${r.name}</td><td class="n">${num(r.size_kb, 1)}</td><td class="n">${r.params}</td><td class="n">${(r.macs / 1e3).toFixed(0)}K</td><td class="n">${num(r.latency_ms, 3)} / ${num(r.latency_p95_ms, 3)} ms</td><td class="n">${pct(r.acc)}</td><td class="n">${pct(r.f1)}</td></tr>`).join('') + '</tbody></table>';
  } else notAvailable($('#bench'), 'python benchmarking/benchmark.py');
}
document.addEventListener('DOMContentLoaded', () => main().catch(e => console.error(e)));
