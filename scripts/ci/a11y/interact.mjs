// interact.mjs: browser behaviour gate for the frozen site (CI job "a11y", after a11y.mjs).
//
//   node scripts/ci/a11y/interact.mjs <site-dir> [--out report.json]
//
// a11y.mjs checks what axe-core and a scroll width can see. This script checks what a
// reader sees once the page's own script has run, with the published CSP in force
// (no bypassCSP: desk.js runs exactly as it does on the site):
//   * signed bars: every bar of a chart series marked signed:true is drawn in the
//     loss fill (--neg-fill) when its value is below zero and in the gain fill
//     (--pos-fill) otherwise, read from the ECharts model of the drawn chart
//     (on the horizontal sleeve chart every bar was drawn in the gain colour);
//   * KPI tones: every .kpi-value.pos / .kpi-value.neg tile computes to --pos / --neg
//     (the .kpi-value rule overrode the tone classes, so no tile showed its sign);
//   * the keyboard dialog: '?' opens #keysDialog centred in the viewport, at desk and
//     phone width (the universal margin reset opened it in the top-left corner).
// Values are read from each chart's own JSON spec and colours from the CSS tokens,
// so the checks hold on every daily rebuild.
import { createServer } from 'node:http';
import { readFile, stat, writeFile } from 'node:fs/promises';
import { extname, join, normalize, resolve } from 'node:path';
import { chromium } from 'playwright';

const args = process.argv.slice(2);
const site = resolve(args[0] || '.site-build');
const outIdx = args.indexOf('--out');
const outPath = outIdx >= 0 ? args[outIdx + 1] : null;

const PAGES = ['index.html', 'surface.html', 'calibration.html', 'risk.html', 'research.html',
  'corrections.html', 'model-risk.html'];
const TONE_PAGES = ['index.html', 'risk.html', 'research.html'];
const DIALOG_WIDTHS = [{ width: 1440, height: 900 }, { width: 390, height: 844 }];
const CHART_TIMEOUT_MS = 20000;
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css',
  '.png': 'image/png', '.svg': 'image/svg+xml', '.json': 'application/json', '.xml': 'application/xml',
  '.txt': 'text/plain', '.woff2': 'font/woff2', '.ico': 'image/x-icon' };

const server = createServer(async (req, res) => {
  try {
    let path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    if (path.endsWith('/')) path += 'index.html';
    const file = normalize(join(site, path));
    if (!file.startsWith(site)) { res.writeHead(403).end(); return; }
    let target = file;
    try { if (!(await stat(target)).isFile()) throw new Error(); } catch { target = file + '.html'; }
    const body = await readFile(target);
    res.writeHead(200, { 'content-type': TYPES[extname(target)] || 'application/octet-stream' }).end(body);
  } catch { res.writeHead(404).end('not found'); }
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}/`;

const browser = await chromium.launch();
const report = { site, checks: [], failures: [] };

async function openPage(name, viewport = { width: 1440, height: 900 }) {
  const ctx = await browser.newContext({ viewport, reducedMotion: 'reduce' });
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.route('**/*', route => {
    const u = route.request().url();
    return (u.startsWith(base) || u.startsWith('data:')) ? route.continue() : route.abort();
  });
  await page.goto(base + name, { waitUntil: 'load' });
  await page.evaluate(() => document.fonts.ready.then(() => true));
  return { ctx, page, errors };
}

// '#4c90f0' / 'rgb(76, 144, 240)' -> 'rgb(76,144,240)'
function norm(c) {
  c = String(c || '').trim().toLowerCase();
  const h = c.match(/^#([0-9a-f]{6})$/);
  if (h) return `rgb(${parseInt(h[1].slice(0, 2), 16)},${parseInt(h[1].slice(2, 4), 16)},${parseInt(h[1].slice(4, 6), 16)})`;
  return c.replace(/\s+/g, '').replace(/^rgba\((\d+,\d+,\d+),1\)$/, 'rgb($1)');
}

// -- signed bars ----
async function checkSignedBars() {
  const check = { name: 'signed bars take the colour of their sign', charts: [], failures: [] };
  for (const name of PAGES) {
    const { ctx, page, errors } = await openPage(name);
    const ids = await page.evaluate(() => Array.from(document.querySelectorAll('.chart[data-spec]'))
      .filter(el => {
        const s = document.getElementById(el.getAttribute('data-spec'));
        return s && (JSON.parse(s.textContent).series || []).some(x => x.signed);
      })
      .map(el => ({ id: el.id, hidden: !!el.closest('[hidden]') })));
    for (const { id, hidden } of ids) {
      if (hidden) { check.charts.push({ page: name, id, skipped: 'in a hidden tab panel' }); continue; }
      await page.locator('#' + id).scrollIntoViewIfNeeded();
      const res = await page.evaluate(([cid, ms]) => new Promise(done => {
        const t0 = Date.now();
        (function poll() {
          const c = window.Desk && Desk.charts[cid];
          if (!c) {
            if (Date.now() - t0 > ms) return done({ error: 'not drawn in ' + ms + ' ms' });
            return setTimeout(poll, 50);
          }
          const root = getComputedStyle(document.documentElement);
          const pos = root.getPropertyValue('--pos-fill').trim(), neg = root.getPropertyValue('--neg-fill').trim();
          const items = [];
          (c.spec.series || []).forEach((s, i) => {
            if (!s.signed) return;
            const data = c.chart.getModel().getSeriesByIndex(i).getData();
            for (let j = 0; j < data.count(); j++) {
              const raw = s.data[j];
              const v = Array.isArray(raw) ? raw[1] : raw;
              if (v === null || v === undefined) continue;
              const cat = (c.spec.xs || [])[j];
              items.push({ series: s.name, label: cat === undefined ? j : cat, value: v,
                           fill: data.getItemVisual(j, 'style').fill, want: v < 0 ? neg : pos });
            }
          });
          done({ items });
        })();
      }), [id, CHART_TIMEOUT_MS]);
      const entry = { page: name, id, bars: 0 };
      if (res.error) check.failures.push(`${name} #${id}: ${res.error}`);
      for (const it of res.items || []) {
        entry.bars++;
        if (norm(it.fill) !== norm(it.want)) {
          check.failures.push(`${name} #${id} ${it.series} '${it.label}' = ${it.value}: fill ${it.fill}, expected ${it.want}`);
        }
      }
      check.charts.push(entry);
    }
    for (const e of errors) check.failures.push(`${name}: JavaScript error: ${e}`);
    await ctx.close();
  }
  if (!check.charts.some(c => c.bars)) check.failures.push('no signed bar chart was found or drawn');
  return check;
}

// -- KPI tones ----
async function checkKpiTones() {
  const check = { name: 'KPI tiles show their tone', tiles: [], failures: [] };
  for (const name of TONE_PAGES) {
    const { ctx, page } = await openPage(name);
    const got = await page.evaluate(() => {
      const root = getComputedStyle(document.documentElement);
      const probe = document.createElement('span');
      document.body.appendChild(probe);
      const want = {};
      for (const t of ['pos', 'neg']) { probe.style.color = root.getPropertyValue('--' + t).trim(); want[t] = getComputedStyle(probe).color; }
      probe.remove();
      return Array.from(document.querySelectorAll('.kpi-value.pos, .kpi-value.neg')).map(el => ({
        tone: el.classList.contains('neg') ? 'neg' : 'pos', text: el.textContent.trim().slice(0, 40),
        color: getComputedStyle(el).color, want: want[el.classList.contains('neg') ? 'neg' : 'pos'] }));
    });
    for (const t of got) {
      check.tiles.push({ page: name, ...t });
      if (norm(t.color) !== norm(t.want)) check.failures.push(`${name}: .kpi-value.${t.tone} '${t.text}' is ${t.color}, expected ${t.want}`);
    }
    await ctx.close();
  }
  if (!check.tiles.length) check.failures.push('no toned KPI tile on ' + TONE_PAGES.join(', '));
  return check;
}

// -- keyboard dialog ----
async function checkKeysDialog() {
  const check = { name: "'?' opens the shortcut list centred", views: [], failures: [] };
  for (const vp of DIALOG_WIDTHS) {
    const { ctx, page } = await openPage('surface.html', vp);
    await page.locator('body').click({ position: { x: 2, y: 2 } }).catch(() => {});
    await page.keyboard.press('Shift+Slash');
    const box = await page.evaluate(() => {
      const d = document.getElementById('keysDialog');
      const r = d.getBoundingClientRect();
      return { open: d.open, left: r.left, top: r.top, width: r.width, height: r.height,
               vw: document.documentElement.clientWidth, vh: window.innerHeight };
    });
    check.views.push({ width: vp.width, ...box });
    if (!box.open) check.failures.push(`surface.html @${vp.width}px: '?' did not open #keysDialog`);
    else if (Math.abs(box.left - (box.vw - box.width) / 2) > 2) {
      check.failures.push(`surface.html @${vp.width}px: #keysDialog left ${box.left.toFixed(1)} px, centred would be ${((box.vw - box.width) / 2).toFixed(1)} px`);
    }
    await ctx.close();
  }
  return check;
}

for (const run of [checkSignedBars, checkKpiTones, checkKeysDialog]) {
  const c = await run();
  report.checks.push(c);
  report.failures.push(...c.failures);
  console.log(`${(c.failures.length ? 'FAIL' : 'ok').padEnd(5)} ${c.name}` +
    (c.charts ? `: ${c.charts.filter(x => x.bars).length} chart(s), ${c.charts.reduce((a, x) => a + (x.bars || 0), 0)} bars` : '') +
    (c.tiles ? `: ${c.tiles.length} tile(s)` : '') + (c.views ? `: ${c.views.length} width(s)` : ''));
}

await browser.close();
server.close();
if (outPath) await writeFile(outPath, JSON.stringify(report, null, 1) + '\n');
console.log(`\ninteraction checks: ${report.failures.length} failure(s)`);
for (const f of report.failures) console.log('  ' + f);
process.exit(report.failures.length ? 1 : 0);
