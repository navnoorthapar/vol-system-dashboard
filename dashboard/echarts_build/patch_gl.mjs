// node patch_gl.mjs <out.js> -- echarts-gl 2.1.0's npm dist file with one CSP patch.
//
// echarts-gl always builds claygl's post-effect compositor, which turns size
// strings such as "expr(width * dpr / 4)" into functions with `new Function`.
// The dashboard's pages declare script-src 'self' (no 'unsafe-eval'), which
// blocks that call and the 3D view fails with "Invalid expression.". This
// replaces the single `new Function("width","height","dpr", ...)` with a small
// recursive-descent evaluator for exactly that grammar (numbers, width / height
// / dpr, + - * /, parentheses, [a, b] arrays), so the CSP stays strict. The
// input must be the published npm file (sha256 checked); the patch must match
// exactly once; the output is deterministic.
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';

const SRC = 'node_modules/echarts-gl/dist/echarts-gl.min.js';
const SRC_SHA256 = 'd6d60f2165ce3eb1ea360ef62e8490835e0ae2c1e0c1788a772fff918e350842';
const NEEDLE = 'new Function("width","height","dpr","return "+t[1])';
const out = process.argv[2] || 'echarts-gl-2.1.0-csp.min.js';

const src = readFileSync(SRC, 'utf8');
const sha = createHash('sha256').update(src, 'utf8').digest('hex');
if (sha !== SRC_SHA256) throw new Error(`unexpected echarts-gl dist: sha256 ${sha}`);
const n = src.split(NEEDLE).length - 1;
if (n !== 1) throw new Error(`expected exactly one compositor new Function, found ${n}`);

const evaluator = `/* desk CSP patch (dashboard/echarts_build/patch_gl.mjs): claygl compositor size expressions are parsed, not eval'd */
var __deskExprFn=function(s){var k=s.match(/\\d+(?:\\.\\d+)?|[A-Za-z_]+|[-+*\\/()[\\],]/g)||[];if(k.join("")!==s.replace(/\\s+/g,""))throw new Error("Invalid expression.");var V={width:0,height:1,dpr:2};return function(w,h,d){var E=[w,h,d],i=0;function P(){return k[i]}function N(){return k[i++]}function A(){var t=N();if(t==="("){var v=S();if(N()!==")")throw new Error("Invalid expression.");return v}if(t==="["){var a=[];if(P()!=="]"){a.push(S());while(P()===","){N();a.push(S())}}if(N()!=="]")throw new Error("Invalid expression.");return a}if(t==="-")return-A();if(t==="+")return+A();if(/^\\d/.test(t))return parseFloat(t);if(t!==undefined&&Object.prototype.hasOwnProperty.call(V,t))return E[V[t]];throw new Error("Invalid expression.")}function M(){var v=A();while(P()==="*"||P()==="/"){var o=N(),r=A();v=o==="*"?v*r:v/r}return v}function S(){var v=M();while(P()==="+"||P()==="-"){var o=N(),r=M();v=o==="+"?v+r:v-r}return v}var r=S();if(i!==k.length)throw new Error("Invalid expression.");return r}};
`;
writeFileSync(out, evaluator + src.replace(NEEDLE, '__deskExprFn(t[1])'));
console.log(`wrote ${out}`);
