// node build.mjs <out.js> — deterministic minified IIFE bundle of src/entry.js.
//
// Rebuild both vendored files (versions pinned in package-lock.json):
//   cd dashboard/echarts_build && npm ci --ignore-scripts && npm run build
// then update the sha384 SRI values in dashboard/app.py (VENDORED); the
// freeze refuses a vendored file that does not match its SRI. Rebuilt on
// 2026-10-20 with Node 22: both outputs matched the committed files byte for
// byte (sha256 e917e082... and 2946cabd...).
import { build } from 'esbuild';
const out = process.argv[2] || 'echarts-desk.min.js';
await build({
  entryPoints: ['src/entry.js'], bundle: true, minify: true, format: 'iife',
  target: ['es2019'], legalComments: 'eof', outfile: out, logLevel: 'info',
  define: { 'process.env.NODE_ENV': '"production"', '__DEV__': 'false' },
});
