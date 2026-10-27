// a11y.mjs: accessibility and layout gate for the frozen site (`make a11y`, CI job "a11y").
//
//   node scripts/ci/a11y/a11y.mjs <site-dir> [--out report.json]
//
// Serves <site-dir> (the output of dashboard/freeze_site.py) on 127.0.0.1, loads every
// page in headless Chromium (Playwright) and fails if
//   * axe-core reports any WCAG 2.0/2.1/2.2 A or AA violation, at 1440 px (desktop) or
//     390 px (phone), in the dark and the light colour scheme;
//   * a page scrolls horizontally at any width from 320 to 1920 px, or at 320 px with every
//     glyph spaced 0.06em wider (a label must not fit only with this machine's fonts);
//   * a page throws a JavaScript error;
//   * a page needs the network: every request to another origin is aborted, so the
//     published pages must work from their own files (the CDN fallback for the
//     charting library is never exercised here; the vendored copy is).
// The report lists every violation with its rule, impact and node count.
import { createServer } from 'node:http';
import { readFile, stat, writeFile } from 'node:fs/promises';
import { extname, join, normalize, resolve } from 'node:path';
import { createRequire } from 'node:module';
import { chromium } from 'playwright';

const require = createRequire(import.meta.url);
const AXE = await readFile(require.resolve('axe-core/axe.min.js'), 'utf8');
const AXE_VERSION = require('axe-core/package.json').version;

const args = process.argv.slice(2);
const site = resolve(args[0] || '.site-build');
const outIdx = args.indexOf('--out');
const outPath = outIdx >= 0 ? args[outIdx + 1] : null;

// The site's seven pages and its 404 (greeks.html and backtest.html are meta-refresh
// redirects to risk.html and research.html since C23, so they are not audited themselves).
const PAGES = ['index.html', 'surface.html', 'calibration.html', 'risk.html', 'research.html',
  'corrections.html', 'model-risk.html', '404.html'];
const AXE_VIEWPORTS = [{ width: 1440, height: 900 }, { width: 390, height: 844 }];
const OVERFLOW_WIDTHS = [320, 390, 768, 1024, 1440, 1920];
const STRESS_SPACING = '0.06em';            // the font-metric margin check below
const TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'];
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
const report = { site, axe: AXE_VERSION, tags: TAGS, pages: [], failures: [] };

async function openPage(ctx, name) {
  const page = await ctx.newPage();
  const errors = [], blocked = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.route('**/*', route => {
    const u = route.request().url();
    if (u.startsWith(base) || u.startsWith('data:')) return route.continue();
    blocked.push(u);
    return route.abort();
  });
  await page.goto(base + name, { waitUntil: 'load' });
  await page.waitForTimeout(500);           // chart animations settle
  return { page, errors, blocked };
}

for (const name of PAGES) {
  const entry = { page: name, axe: [], overflow: [], errors: [], blocked: [] };
  for (const scheme of ['dark', 'light']) {
    for (const vp of AXE_VIEWPORTS) {
      // bypassCSP: the pages' CSP (script-src 'self') would block the injected axe-core;
      // only the audit script is exempted, the page's own scripts run as published.
      const ctx = await browser.newContext({ viewport: vp, colorScheme: scheme, reducedMotion: 'reduce',
                                             bypassCSP: true });
      const { page, errors, blocked } = await openPage(ctx, name);
      await page.addScriptTag({ content: AXE });
      const violations = await page.evaluate(async tags => {
        const r = await window.axe.run(document, { runOnly: { type: 'tag', values: tags } });
        return r.violations.map(v => ({ id: v.id, impact: v.impact, nodes: v.nodes.length, help: v.help }));
      }, TAGS);
      for (const v of violations) entry.axe.push({ ...v, width: vp.width, scheme });
      entry.errors.push(...errors);
      entry.blocked.push(...blocked);
      await ctx.close();
    }
  }
  for (const width of OVERFLOW_WIDTHS) {
    const ctx = await browser.newContext({ viewport: { width, height: 900 } });
    const { page } = await openPage(ctx, name);
    const sw = await page.evaluate(() => document.documentElement.scrollWidth);
    if (sw > width + 1) entry.overflow.push({ width, scrollWidth: sw });
    await ctx.close();
  }
  {
    // Font-metric margin: the narrowest width again with every glyph spaced
    // wider. A label that fits only with this machine's fonts (a fallback
    // glyph is wider on another runner) scrolls the page here first: CI's
    // runner overflowed surface.html by 7 px at 320 px while this machine did not.
    const ctx = await browser.newContext({ viewport: { width: OVERFLOW_WIDTHS[0], height: 900 } });
    const { page } = await openPage(ctx, name);
    await page.addStyleTag({ content: `* { letter-spacing: ${STRESS_SPACING} !important; }` });
    await page.waitForTimeout(200);
    const sw = await page.evaluate(() => document.documentElement.scrollWidth);
    if (sw > OVERFLOW_WIDTHS[0] + 1) entry.overflow.push({ width: OVERFLOW_WIDTHS[0], scrollWidth: sw, stressed: true });
    await ctx.close();
  }
  entry.blocked = [...new Set(entry.blocked)];
  entry.errors = [...new Set(entry.errors)];
  report.pages.push(entry);
  for (const v of entry.axe) report.failures.push(`${name} @${v.width}px ${v.scheme}: axe ${v.id} (${v.impact}, ${v.nodes} node(s)): ${v.help}`);
  for (const o of entry.overflow) report.failures.push(`${name}: scrolls horizontally at ${o.width}px` +
    (o.stressed ? ` with letter-spacing +${STRESS_SPACING}` : '') + ` (scrollWidth ${o.scrollWidth})`);
  for (const e of entry.errors) report.failures.push(`${name}: JavaScript error: ${e}`);
  for (const b of entry.blocked) report.failures.push(`${name}: requests another origin: ${b}`);
  const status = entry.axe.length + entry.overflow.length + entry.errors.length + entry.blocked.length ? 'FAIL' : 'ok';
  console.log(`${status.padEnd(5)} ${name.padEnd(18)} axe violations ${entry.axe.length}, overflow ${entry.overflow.length}, ` +
              `js errors ${entry.errors.length}, external requests ${entry.blocked.length}`);
}

await browser.close();
server.close();
if (outPath) await writeFile(outPath, JSON.stringify(report, null, 1) + '\n');
console.log(`\naxe-core ${AXE_VERSION}, tags ${TAGS.join(',')}: ${report.failures.length} failure(s)`);
for (const f of report.failures) console.log('  ' + f);
process.exit(report.failures.length ? 1 : 0);
