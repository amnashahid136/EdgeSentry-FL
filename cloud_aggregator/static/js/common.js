/* shared helpers */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const CLS = ['normal', 'inner_race_fault', 'outer_race_fault', 'ball_fault'];
const LABELS = ['Normal', 'Inner-race fault', 'Outer-race fault', 'Ball fault'];
const CLS_COLOR = ['#34d399', '#f87171', '#fb923c', '#a78bfa'];
const MACHINES = ['edge01', 'edge02', 'edge03'];

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) { let t = ''; try { t = (await r.text()).replace(/<[^>]+>/g, ' ').trim().slice(0, 200); } catch (e) {} throw new Error(r.status + ' ' + t); }
  return r.json();
}
const post = (path, body) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
const pct = (x, d = 1) => (x == null || isNaN(x)) ? '–' : (x * 100).toFixed(d) + '%';
const num = (x, d = 1) => (x == null || isNaN(x)) ? '–' : Number(x).toFixed(d);
const msd = (o, d = 1) => o ? `${(o.mean * 100).toFixed(d)} ± ${(o.std * 100).toFixed(d)}%` : '–';
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmtTime = t => new Date(t * 1000).toLocaleTimeString();
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

/* theme */
(function () {
  const t = localStorage.getItem('es-theme') || 'dark';
  document.documentElement.setAttribute('data-theme', t);
})();
function toggleTheme() {
  const n = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  localStorage.setItem('es-theme', n); location.reload();
}
function chartDefaults() {
  if (!window.Chart) return;
  Chart.defaults.color = css('--muted'); Chart.defaults.borderColor = css('--line');
  Chart.defaults.font.family = '-apple-system,Segoe UI,Roboto,sans-serif'; Chart.defaults.font.size = 12;
  Chart.defaults.maintainAspectRatio = false; Chart.defaults.animation = { duration: 250 };
}
function mkChart(id, cfg) { const el = document.getElementById(id); if (!el || !window.Chart) return null; return new Chart(el, cfg); }
function notAvailable(el, script) {
  el.innerHTML = `<div class="empty">Results not generated yet. Run <code>${script}</code> (or <code>./run_pipeline.sh</code>) and refresh.</div>`;
}
document.addEventListener('DOMContentLoaded', chartDefaults);
