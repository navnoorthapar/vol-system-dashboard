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
//   * KPI strips on a phone (320 and 390 px): no value pair of a range under a KPI
//     ('12.51–25.79', '−9.62 to −3.48') breaks across lines, and no strip leaves an
//     empty cell beside its last tile;
//   * the keyboard dialog: '?' opens #keysDialog centred in the viewport, at desk and
//     phone width (the universal margin reset opened it in the top-left corner);
//   * grid rows: at 1024, 1280, 1440 and 1920 px no row of a .g grid leaves more than
//     60 px unused (an s7 + s5 row became 12 + 6 below 1181 px: half-empty rows);
//   * panel bottoms: no .panel ends more than 96 px below its last visible content
//     (a short panel stretched beside a tall one), every chart drawn first; the body
//     of a closed <details> is not visible content; #carry and #forward are left to
//     their own fixes;
//   * heatmap row heads: at 390 px a heatmap's row header stays in view after its
//     region is scrolled sideways;
//   * Model Risk status filters: each chip of the summary strip shows exactly the
//     rows of its status, as many as it counts, and is the only one pressed; 'All'
//     shows every row again;
//   * the quote inspector (P4.1): hovering a quote of the Surface smile (the
//     expiry the page shows, every model toggled on) gives a tooltip with its
//     strike, a bid and an ask, one row with z per model and no raw series name;
//     the keyboard readout says the same; the residual strip shares the smile's
//     x range; the forward-delta switch renames the axis and keeps the expiry; a
//     Calibration smile's tooltip names its quote the same way;
//   * the smile's expiry switch is fast (P4.2): the median of five expiry-tab
//     clicks, timed by the Event Timing API (PerformanceObserver 'event',
//     durationThreshold 16, from the input to the next paint), is at most 50 ms,
//     and at most 150 ms with the CPU throttled 4× (CDP); a legend toggle at most
//     30 ms. The chart itself is drawn in the task after that paint, so the time
//     from the input to the frame that shows the new expiry's chart is measured
//     too (the chart's 'finished' event, then the next frame): its median at most
//     100 ms, 300 ms at 4×. (It re-parsed a 250 KB spec and rebuilt every series
//     on each click: 160 ms, 640 ms at 4×, measured on the previous desk.js.)
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

// -- KPI strips on a phone ----
// R3-D11: at 320 px a range under a KPI broke inside a value pair ('12.51–' / '25.79'),
// and an odd last tile of a two-column strip left the strip's grey background showing.
const KPI_WIDTHS = [{ width: 320, height: 800 }, { width: 390, height: 844 }];
async function checkKpiStrips() {
  const check = { name: 'KPI ranges keep each value pair on a line; strips have no empty cell', views: [], failures: [] };
  for (const name of PAGES) {
    for (const vp of KPI_WIDTHS) {
      const { ctx, page } = await openPage(name, vp);
      const got = await page.evaluate(() => {
        const pair = /[+\u2212-]?[\d.,]+%?(?:–| to )[+\u2212-]?[\d.,]+%?/g;
        const broken = [];
        for (const el of document.querySelectorAll('.kpi-range')) {
          if (!el.getClientRects().length) continue;
          const walk = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
          let text = '', nodes = [];
          for (let n = walk.nextNode(); n; n = walk.nextNode()) { nodes.push([n, text.length]); text += n.data; }
          for (const m of text.matchAll(pair)) {
            const at = (i) => { const k = nodes.findLastIndex(([, s]) => s <= i); return [nodes[k][0], i - nodes[k][1]]; };
            const r = document.createRange();
            r.setStart(...at(m.index)); r.setEnd(...at(m.index + m[0].length - 1)); r.setEnd(r.endContainer, r.endOffset + 1);
            const tops = new Set([...r.getClientRects()].filter(q => q.width > 0).map(q => Math.round(q.top)));
            if (tops.size > 1) broken.push(m[0]);
          }
        }
        const holes = [];
        for (const g of document.querySelectorAll('.kpis')) {
          const tiles = [...g.children].filter(k => k.getClientRects().length);
          if (!tiles.length) continue;
          const gr = g.getBoundingClientRect();
          const lastTop = Math.max(...tiles.map(k => Math.round(k.getBoundingClientRect().top)));
          const right = Math.max(...tiles.filter(k => Math.round(k.getBoundingClientRect().top) === lastTop)
            .map(k => k.getBoundingClientRect().right));
          if (gr.right - right > 24) holes.push(`${g.id || g.closest('[id]')?.id || '.kpis'}: ${Math.round(gr.right - right)} px empty`);
        }
        return { broken, holes };
      });
      check.views.push({ page: name, width: vp.width, ...got });
      for (const b of got.broken) check.failures.push(`${name} @${vp.width}px: KPI range '${b}' breaks across lines`);
      for (const h of got.holes) check.failures.push(`${name} @${vp.width}px: KPI strip ${h} in its last row`);
      await ctx.close();
    }
  }
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

// -- grid rows and panel bottoms (P2.7) ----
const LAYOUT_WIDTHS = [1024, 1280, 1440, 1920];
const MAX_UNUSED_PX = 60;
const MAX_BLANK_PX = 96;
const BLANK_EXEMPT = ['carry', 'forward'];
async function checkLayout() {
  const check = { name: 'grid rows are full and panels end with their content', views: [], failures: [] };
  for (const width of LAYOUT_WIDTHS) {
    for (const name of PAGES) {
      const { ctx, page, errors } = await openPage(name, { width, height: 900 });
      // draw every chart (the IntersectionObserver draws them as they come into view)
      const h = await page.evaluate(() => document.documentElement.scrollHeight);
      for (let y = 0; y < h; y += 700) { await page.evaluate(v => window.scrollTo(0, v), y); await page.waitForTimeout(40); }
      await page.waitForTimeout(300);
      const r = await page.evaluate(([maxUnused, maxBlank, exempt]) => {
        const vis = el => { const b = el.getBoundingClientRect(), s = getComputedStyle(el);
          return b.width > 0 && b.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
        // the body of a closed <details> keeps a layout box in Chromium although nothing of it
        // shows: counting it hid a blank panel bottom below a closed disclosure
        const shut = el => { const d = el.closest('details:not([open])'), s = el.closest('summary');
          return !!d && d !== el && !(s && s.parentElement === d); };
        const holes = [], blanks = [];
        document.querySelectorAll('.g').forEach(g => {
          const gap = parseFloat(getComputedStyle(g).columnGap) || 0, rows = {};
          [...g.children].filter(vis).forEach(k => { const t = Math.round(k.getBoundingClientRect().top); (rows[t] = rows[t] || []).push(k); });
          Object.values(rows).forEach(ks => {
            const used = ks.reduce((a, k) => a + k.getBoundingClientRect().width, 0) + gap * (ks.length - 1);
            const unused = g.getBoundingClientRect().width - used;
            if (unused > maxUnused) holes.push(`${ks.map(k => '#' + (k.id || k.className)).join(' + ')} leaves ${Math.round(unused)} px`);
          });
        });
        document.querySelectorAll('.panel').forEach(p => {
          if (!vis(p) || exempt.includes(p.id)) return;
          let last = -Infinity;
          p.querySelectorAll('.panel-body *').forEach(el => {
            if (!vis(el) || el.closest('[hidden]') || el.closest('.sr-only') || shut(el)) return;
            last = Math.max(last, el.getBoundingClientRect().bottom);
          });
          if (last === -Infinity) return;
          const blank = p.getBoundingClientRect().bottom - last;
          if (blank > maxBlank) blanks.push(`#${p.id} ends ${Math.round(blank)} px below its content`);
        });
        return { holes, blanks };
      }, [MAX_UNUSED_PX, MAX_BLANK_PX, BLANK_EXEMPT]);
      check.views.push({ name, width, holes: r.holes.length, blanks: r.blanks.length });
      for (const x of r.holes) check.failures.push(`${name} @${width}px: row ${x} unused`);
      for (const x of r.blanks) check.failures.push(`${name} @${width}px: ${x}`);
      for (const e of errors) check.failures.push(`${name} @${width}px: JavaScript error: ${e}`);
      await ctx.close();
    }
  }
  return check;
}

// -- heatmap row heads on a phone (P2.7) ----
async function checkStickyHeatHeads() {
  const check = { name: 'heatmap row heads stay in view while the region scrolls', heatmaps: [], failures: [] };
  for (const name of ['surface.html', 'research.html', 'risk.html']) {
    const { ctx, page } = await openPage(name, { width: 390, height: 844 });
    const res = await page.evaluate(async () => {
      const out = [];
      for (const wrap of document.querySelectorAll('.heat-wrap')) {
        if (wrap.closest('[hidden]') || wrap.scrollWidth <= wrap.clientWidth + 1) continue;
        wrap.scrollIntoView({ block: 'center' });
        wrap.scrollLeft = Math.min(200, wrap.scrollWidth - wrap.clientWidth);
        await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
        const th = wrap.querySelector('tbody th[scope="row"]');
        if (!th) continue;
        const a = th.getBoundingClientRect(), w = wrap.getBoundingClientRect();
        out.push({ id: wrap.querySelector('table').id, left: a.left, right: a.right, wl: w.left, vw: innerWidth });
      }
      return out;
    });
    for (const h of res) {
      check.heatmaps.push({ name, ...h });
      if (h.left < h.wl - 1 || h.right > h.vw + 1) check.failures.push(`${name} #${h.id} @390px: row header at ${h.left.toFixed(0)}–${h.right.toFixed(0)} px after a sideways scroll, region starts at ${h.wl.toFixed(0)} px`);
    }
    await ctx.close();
  }
  if (!check.heatmaps.length) check.failures.push('no heatmap scrolls sideways at 390 px');
  return check;
}

// -- Model Risk status filters (P3.3) ----
async function checkMonitorFilters() {
  const check = { name: 'Model Risk status chips filter the checks', filters: [], failures: [] };
  const { ctx, page, errors } = await openPage('model-risk.html');
  const chips = await page.evaluate(() => Array.from(document.querySelectorAll('[data-filter-for="monitor-table"]'))
    .map(b => ({ filter: b.getAttribute('data-filter'), count: b.getAttribute('data-count') })));
  const total = await page.evaluate(() => document.querySelectorAll('#monitor-table tbody tr').length);
  if (chips.filter(c => c.filter.startsWith('status:')).length < 2) check.failures.push('model-risk.html: fewer than two status chips');
  for (const c of [...chips.filter(x => x.filter !== 'all'), chips.find(x => x.filter === 'all')]) {
    if (!c) continue;
    await page.locator(`[data-filter-for="monitor-table"][data-filter="${c.filter}"]`).click();
    const r = await page.evaluate(f => ({
      shown: Array.from(document.querySelectorAll('#monitor-table tbody tr')).filter(tr => !tr.hidden && getComputedStyle(tr).display !== 'none')
        .map(tr => tr.getAttribute('data-status')),
      pressed: Array.from(document.querySelectorAll('[data-filter-for="monitor-table"][aria-pressed="true"]')).map(b => b.getAttribute('data-filter')),
    }), c.filter);
    check.filters.push({ filter: c.filter, shown: r.shown.length });
    const want = c.filter === 'all' ? total : Number(c.count);
    const st = c.filter.slice(7);
    if (r.shown.length !== want) check.failures.push(`model-risk.html ${c.filter}: ${r.shown.length} rows shown, the chip counts ${want}`);
    if (c.filter !== 'all' && r.shown.some(x => x !== st)) check.failures.push(`model-risk.html ${c.filter}: shows rows of other statuses (${r.shown.join(', ')})`);
    if (r.pressed.length !== 1 || r.pressed[0] !== c.filter) check.failures.push(`model-risk.html ${c.filter}: pressed ${r.pressed.join(', ')}`);
  }
  for (const e of errors) check.failures.push(`model-risk.html: JavaScript error: ${e}`);
  await ctx.close();
  return check;
}

// -- the quote inspector (P4.1) ----
const TIP_JS = id => { const t = [...document.querySelectorAll('#' + id + ' > div')].filter(d => getComputedStyle(d).display !== 'none'
  && d.style.position === 'absolute' && d.innerText.startsWith('K '))[0]; return t ? t.innerText : null; };
const joinLines = t => t.split('\n').map(x => x.trim()).filter(Boolean).join(' \u00b7 ');
async function waitChart(page, id) {
  return page.evaluate(([cid, ms]) => new Promise(done => {
    const t0 = Date.now();
    (function poll() {
      if (window.Desk && Desk.charts[cid] && document.fonts.status === 'loaded') return done(true);
      if (Date.now() - t0 > ms) return done(false);
      setTimeout(poll, 50);
    })();
  }), [id, CHART_TIMEOUT_MS]);
}
async function hoverQuote(page, id, frac) {
  const pos = await page.evaluate(([cid, f]) => {
    const c = Desk.charts[cid], rows = c.spec._rows[Object.keys(c.spec._rows).length > 1 ? Desk.filters[cid] : Object.keys(c.spec._rows)[0]];
    const q = rows[Math.floor((rows.length - 1) * f)];
    const px = c.chart.convertToPixel({ xAxisIndex: 0, yAxisIndex: 0 }, [q.k, q.mid]);
    const r = c.el.getBoundingClientRect();
    return { x: r.left + px[0], y: r.top + px[1] };
  }, [id, frac]);
  await page.mouse.move(pos.x, pos.y);
  await page.waitForTimeout(250);
  return page.evaluate(TIP_JS, id);
}
async function checkQuoteInspector() {
  const check = { name: 'the quote inspector names the quote under the pointer and from the keyboard', views: [], failures: [] };
  const { ctx, page, errors } = await openPage('surface.html');
  const id = 'chart-slices';
  await page.locator('#' + id).scrollIntoViewIfNeeded();
  if (!(await waitChart(page, id))) { check.failures.push(`surface.html #${id}: not drawn`); await ctx.close(); return check; }
  const exp = await page.evaluate(cid => Desk.filters[cid], id);
  const models = await page.evaluate(cid => Desk.charts[cid].spec.quotes.models.map(m => m.label), id);
  for (const label of models) {
    const b = page.locator(`[data-legend-for="${id}"][data-series="${label}"]`);
    if ((await b.getAttribute('aria-pressed')) === 'false') await b.click();
  }
  const tip = await hoverQuote(page, id, 0.5);
  check.views.push({ page: 'surface.html', expiry: exp, tooltip: tip });
  if (!tip) check.failures.push(`surface.html #${id} ${exp}: no tooltip on a quote`);
  else {
    const lines = tip.split('\n').map(x => x.trim()).filter(Boolean);
    if (!/^K [\d,]+ [PC] /.test(lines[0])) check.failures.push(`surface.html #${id}: tooltip head '${lines[0]}' names no strike`);
    if (!/Bid \/ mid \/ ask \d+\.\d\d \/ \d+\.\d\d \/ \d+\.\d\d%/.test(tip)) check.failures.push(`surface.html #${id}: tooltip has no bid / ask`);
    const zRows = lines.filter(x => / \u00b7 z [\u2212+]?\d+\.\d/.test(x)).length;
    if (zRows !== models.length) check.failures.push(`surface.html #${id}: ${zRows} model rows with z, ${models.length} models shown`);
    if (/\d{4}-\d{2}-\d{2} mid/.test(tip)) check.failures.push(`surface.html #${id}: tooltip leaks a raw series name`);
    const ro = await page.locator('#' + id + '-readout').innerText();
    if (ro !== joinLines(tip)) check.failures.push(`surface.html #${id}: the readout ('${ro.slice(0, 60)}…') is not the tooltip's text`);
  }
  await page.focus('#' + id + '-readout');
  await page.keyboard.press('ArrowRight');
  await page.waitForTimeout(200);
  const ro2 = await page.locator('#' + id + '-readout').innerText(), tip2 = await page.evaluate(TIP_JS, id);
  if (!tip2 || ro2 !== joinLines(tip2)) check.failures.push(`surface.html #${id}: after an arrow key the readout and the tooltip differ`);
  const ax = await page.evaluate(cid => Desk.charts[cid].chart.getOption().xAxis.map(a => [a.min, a.max]), id);
  if (ax.length !== 2 || ax[0][0] !== ax[1][0] || ax[0][1] !== ax[1][1]) check.failures.push(`surface.html #${id}: residual strip x range ${JSON.stringify(ax)}`);
  await page.click(`[data-xmode-for="${id}"][data-xmode="delta"]`);
  await page.waitForTimeout(200);
  const names = await page.evaluate(cid => Desk.charts[cid].chart.getOption().xAxis.map(a => a.name), id);
  if (!names.includes('Forward delta')) check.failures.push(`surface.html #${id}: the delta switch left the x axis named ${JSON.stringify(names)}`);
  const sel = await page.locator(`[data-chart-filter="${id}"] [data-value="${exp}"]`).getAttribute('aria-selected');
  if (sel !== 'true') check.failures.push(`surface.html #${id}: the delta switch lost the ${exp} expiry`);
  for (const e of errors) check.failures.push(`surface.html: JavaScript error: ${e}`);
  await ctx.close();
  // a Calibration smile
  const c2 = await openPage('calibration.html');
  const cid = await c2.page.evaluate(() => { const e = document.querySelector('.chart[id^="chart-smile-"]'); return e ? e.id : null; });
  if (cid) {
    await c2.page.locator('#' + cid).scrollIntoViewIfNeeded();
    const has = await c2.page.evaluate(i => { const s = document.getElementById(i + '-spec'); return !!(s && JSON.parse(s.textContent).quotes); }, cid);
    if (has && await waitChart(c2.page, cid)) {
      const t = await hoverQuote(c2.page, cid, 0.8);
      check.views.push({ page: 'calibration.html', id: cid, tooltip: t });
      if (!t || !/^K [\d,]+ [PC] /.test(t) || !/ \u00b7 z /.test(t)) check.failures.push(`calibration.html #${cid}: tooltip '${t}' names no quote`);
    }
  }
  for (const e of c2.errors) check.failures.push(`calibration.html: JavaScript error: ${e}`);
  await c2.ctx.close();
  return check;
}

// -- the smile's expiry switch (P4.2) ----
const SWITCH_MAX_MS = { 1: 50, 4: 150 };
const DRAWN_MAX_MS = { 1: 100, 4: 300 };
const LEGEND_MAX_MS = 30;
const median = xs => { const s = [...xs].sort((a, b) => a - b); return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2; };
// one interaction's duration: the longest Event Timing entry it produced (none:
// under the 16 ms threshold), and the time from its click to the frame after the
// chart's next 'finished' event (null: the chart was not drawn again)
async function timed(page, act, id) {
  await page.evaluate(cid => {
    window.__ev = []; window.__drawn = null;
    const ch = Desk.charts[cid].chart;
    const done = () => { ch.off('finished', done); requestAnimationFrame(() => { window.__drawn = performance.now(); }); };
    ch.on('finished', done);
  }, id);
  await act();
  await page.waitForTimeout(400);
  const { ev, drawn } = await page.evaluate(() => ({ ev: window.__ev, drawn: window.__drawn }));
  const click = ev.find(e => e.n === 'click'), timing = ev.filter(e => e !== click);
  return { ms: timing.length ? Math.max(...timing.map(e => e.d)) : 16,
           drawn: drawn !== null && click ? Math.round(drawn - click.t) : null };
}
async function checkSwitchSpeed() {
  const check = { name: 'the smile switches expiry and toggles a model quickly', views: [], failures: [] };
  for (const rate of [1, 4]) {
    const { ctx, page, errors } = await openPage('surface.html');
    const id = 'chart-slices';
    await page.locator('#' + id).scrollIntoViewIfNeeded();
    if (!(await waitChart(page, id))) { check.failures.push(`surface.html #${id}: not drawn`); await ctx.close(); continue; }
    if (rate > 1) { const cdp = await ctx.newCDPSession(page); await cdp.send('Emulation.setCPUThrottlingRate', { rate }); }
    await page.evaluate(() => {
      new PerformanceObserver(l => { for (const e of l.getEntries()) (window.__ev = window.__ev || []).push({ n: e.name + ' (timing)', d: e.duration }); })
        .observe({ type: 'event', durationThreshold: 16 });
      // the click's own start, under the threshold or not
      document.addEventListener('click', e => { (window.__ev = window.__ev || []).push({ n: 'click', d: 0, t: e.timeStamp }); }, true);
    });
    const tabs = await page.evaluate(cid => [...document.querySelectorAll(`[data-chart-filter="${cid}"] [role="tab"]`)]
      .filter(t => t.getAttribute('aria-selected') !== 'true').map(t => t.getAttribute('data-value')), id);
    const picks = [0.1, 0.3, 0.5, 0.7, 0.9].map(f => tabs[Math.floor(f * (tabs.length - 1))]);
    const runs = [];
    for (const v of picks) runs.push(await timed(page, () => page.click(`[data-chart-filter="${id}"] [data-value="${v}"]`), id));
    const clicks = runs.map(r => r.ms), drawn = runs.map(r => r.drawn);
    const view = { rate, clicks, median: median(clicks), drawn };
    if (view.median > SWITCH_MAX_MS[rate]) check.failures.push(`surface.html #${id} at ${rate}× CPU: median expiry switch ${view.median} ms (${clicks.join(', ')}), limit ${SWITCH_MAX_MS[rate]} ms`);
    if (drawn.some(d => d === null)) check.failures.push(`surface.html #${id} at ${rate}× CPU: an expiry switch did not draw the chart again (${drawn.join(', ')})`);
    else {
      view.drawnMedian = median(drawn);
      if (view.drawnMedian > DRAWN_MAX_MS[rate]) check.failures.push(`surface.html #${id} at ${rate}× CPU: median click-to-chart ${view.drawnMedian} ms (${drawn.join(', ')}), limit ${DRAWN_MAX_MS[rate]} ms`);
    }
    if (rate === 1) {
      const btn = page.locator(`[data-legend-for="${id}"][aria-pressed="false"]`).first();
      const key = await btn.getAttribute('data-series');
      const toggles = [];
      for (let i = 0; i < 5; i++) toggles.push((await timed(page, () => page.click(`[data-legend-for="${id}"][data-series="${key}"]`), id)).ms);
      view.legend = toggles; view.legendMedian = median(toggles);
      if (view.legendMedian > LEGEND_MAX_MS) check.failures.push(`surface.html #${id}: median legend toggle ${view.legendMedian} ms (${toggles.join(', ')}), limit ${LEGEND_MAX_MS} ms`);
    }
    check.views.push(view);
    for (const e of errors) check.failures.push(`surface.html: JavaScript error: ${e}`);
    await ctx.close();
  }
  return check;
}

const ONLY = process.env.INTERACT_ONLY ? process.env.INTERACT_ONLY.split(',') : null;
for (const run of [checkSignedBars, checkKpiTones, checkKpiStrips, checkKeysDialog, checkLayout, checkStickyHeatHeads, checkMonitorFilters,
  checkQuoteInspector, checkSwitchSpeed]) {
  if (ONLY && !ONLY.includes(run.name)) continue;
  const c = await run();
  report.checks.push(c);
  report.failures.push(...c.failures);
  console.log(`${(c.failures.length ? 'FAIL' : 'ok').padEnd(5)} ${c.name}` +
    (c.charts ? `: ${c.charts.filter(x => x.bars).length} chart(s), ${c.charts.reduce((a, x) => a + (x.bars || 0), 0)} bars` : '') +
    (c.tiles ? `: ${c.tiles.length} tile(s)` : '') + (c.views ? `: ${c.views.length} width(s)` : '') +
    (run === checkSwitchSpeed ? ' ' + JSON.stringify(c.views.map(v => ({ rate: v.rate, median: v.median, drawn: v.drawnMedian, legend: v.legendMedian }))) : ''));
}

await browser.close();
server.close();
if (outPath) await writeFile(outPath, JSON.stringify(report, null, 1) + '\n');
console.log(`\ninteraction checks: ${report.failures.length} failure(s)`);
for (const f of report.failures) console.log('  ' + f);
process.exit(report.failures.length ? 1 : 0);
