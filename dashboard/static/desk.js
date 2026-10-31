/* desk.js -- the dashboard's only page script (no inline JavaScript anywhere).
 *
 * Every chart is described by a JSON spec that the server writes into a
 * <script type="application/json"> block next to a server-rendered data table;
 * this file turns the spec into an Apache ECharts option. Without JavaScript,
 * or if ECharts fails to load, the tables are shown instead (.charts-failed /
 * <noscript>). Numbers are formatted here exactly as the server formats them:
 * a true minus sign (U+2212), fixed decimals, explicit + on signed values.
 */
(function () {
  'use strict';
  var doc = document, root = doc.documentElement;
  var Desk = window.Desk = { charts: {}, filters: {}, xmode: {} };
  var MINUS = '\u2212';

  function css(name) { return getComputedStyle(root).getPropertyValue(name).trim(); }
  function $(sel, el) { return (el || doc).querySelector(sel); }
  function $$(sel, el) { return Array.prototype.slice.call((el || doc).querySelectorAll(sel)); }

  // -- number formatting ----
  function num(v, o) {
    o = o || {};
    if (v === null || v === undefined || v === '' || !isFinite(v)) return '\u2014';
    var d = o.digits === undefined ? 2 : o.digits;
    var x = Number(v) * (o.scale || 1);
    var s = Math.abs(x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    var zero = Number(s.replace(/,/g, '')) === 0;
    var sign = x < 0 && !zero ? MINUS : (o.signed && x > 0 && !zero ? '+' : '');
    return (o.prefix ? sign + o.prefix : sign) + s + (o.unit || '');
  }
  Desk.num = num;
  function axisFmt(a) { return function (v) { return num(v, { digits: a.digits === undefined ? 1 : a.digits, signed: a.signed, unit: a.tickUnit || '', prefix: a.prefix }); }; }

  // -- theme from the CSS tokens ----
  function theme(print) {
    var t = {
      text: css('--text'), text2: css('--text2'), text3: css('--text3'),
      line: css('--line'), line2: css('--line2'), grid: css('--grid'),
      surface: css('--surface'), tooltip: css('--tooltip'),
      sans: css('--sans'), mono: css('--mono'),
      colors: {
        headline: css('--series-headline'), muted: css('--series-muted'),
        pos: css('--pos-fill'), neg: css('--neg-fill'), ok: css('--ok'), watch: css('--watch'),
        breach: css('--breach'), info: css('--info'),
        cat0: css('--cat-0'), cat1: css('--cat-1'), cat2: css('--cat-2'),
        cat3: css('--cat-3'), cat4: css('--cat-4'), cat5: css('--cat-5'),
        ramp0: css('--ramp-0'), ramp1: css('--ramp-1'), ramp2: css('--ramp-2'),
        ramp3: css('--ramp-3'), ramp4: css('--ramp-4'), ramp5: css('--ramp-5'),
        'tail-down': css('--tail-down'), 'tail-up': css('--tail-up'),
      },
    };
    if (print) {
      t.text = '#111111'; t.text2 = '#333333'; t.text3 = '#555555';
      t.line = '#bbbbbb'; t.line2 = '#999999'; t.grid = '#dddddd'; t.surface = '#ffffff'; t.tooltip = '#ffffff';
      t.colors.headline = '#111111'; t.colors.muted = '#666666';
      t.colors.cat3 = '#8a7a00'; t.colors['tail-down'] = '#a3369f'; t.colors['tail-up'] = '#4d7a00';
      // the maturity ramp on paper: light to dark violet that still reads on white
      ['#9c8ad8', '#8a76cf', '#7762c4', '#6450b4', '#523fa0', '#412f88'].forEach(function (h, i) { t.colors['ramp' + i] = h; });
    }
    return t;
  }
  function color(c, T) { return !c ? T.colors.cat0 : (c.charAt(0) === '#' ? c : (T.colors[c] || c)); }
  function alpha(hex, a) {
    var h = hex.replace('#', '');
    if (h.length !== 6) return hex;
    return 'rgba(' + parseInt(h.slice(0, 2), 16) + ',' + parseInt(h.slice(2, 4), 16) + ',' + parseInt(h.slice(4, 6), 16) + ',' + a + ')';
  }

  // -- quote-level smiles: spec.quotes.groups[expiry], packed by desk_quotes.py;
  // decodeQuotes / quotePoints are the twins of its decode / series_data ----
  var QP = { iv: 1e5, spread: 1e8, model: 1e4, floor: 0.01, days: 365 };
  // the standard normal CDF to double precision (Hart 1968, as given by West 2005)
  function ndtr(x) {
    var z = Math.abs(x), c = 0;
    if (z <= 37) {
      var e = Math.exp(-z * z / 2);
      if (z < 7.07106781186547) {
        var n = ((((((3.52624965998911e-2 * z + 0.700383064443688) * z + 6.37396220353165) * z + 33.912866078383) * z
          + 112.079291497871) * z + 221.213596169931) * z + 220.206867912376);
        var d = (((((((8.83883476483184e-2 * z + 1.75566716318264) * z + 16.064177579207) * z + 86.7807322029461) * z
          + 296.564248779674) * z + 637.333633378831) * z + 793.826512519948) * z + 440.413735824752);
        c = e * n / d;
      } else {
        var f = z + 1 / (z + 2 / (z + 3 / (z + 4 / (z + 0.65))));
        c = e / f / 2.506628274631;
      }
    }
    return x > 0 ? 1 - c : c;
  }
  Desk.ndtr = ndtr;
  function decodeQuotes(g) {
    var out = [], K = g.K0, F = g.F, T = g.T, models = Object.keys(g.m || {});
    for (var i = 0; i < g.mid.length; i++) {
      if (i) K += g.dK[i - 1] * g.step;
      var mid = g.mid[i] / QP.iv, db = g.db[i] / QP.spread, da = g.da[i] / QP.spread, hs = (db + da) / 2;
      var k = Math.log(K / F), sw = Math.sqrt(Math.pow(mid / 100, 2) * T / QP.days), d1 = -k / sw + sw / 2;
      var right = g.r.charAt(i), fl = g.f ? +g.f.charAt(i) : 0;
      var q = { i: i, strike: K, right: right, k: k, T: T, mid: mid, bid: mid - db, ask: mid + da, hs: hs,
        delta: right === 'P' ? ndtr(d1) - 1 : ndtr(d1), xd: ndtr(-d1),
        exec: !!(fl & 1), repaired: !!(fl & 2), outside: !!(fl & 4), models: {} };
      models.forEach(function (m) {
        var e = g.m[m][i], st = g.s && g.s[m] ? +g.s[m].charAt(i) : 0;
        q.models[m] = { iv: e === null ? null : mid + e / QP.model,
          z: e === null ? null : (e / QP.model) / Math.max(hs, QP.floor),
          inside: e === null ? null : !!(st & 2), fit: !!(st & 1) };
      });
      out.push(q);
    }
    return out;
  }
  Desk.decodeQuotes = decodeQuotes;
  function quotePoints(rows, s, xmode, clip) {
    var f = s.quote, out = [];
    rows.forEach(function (q) {
      var x = xmode === 'delta' ? q.xd : q.k, mm = s.model ? q.models[s.model] : null;
      if (f === 'ba') out.push([x, q.bid, q.ask]);
      else if (f === 'mid') out.push([x, q.mid]);
      else if (f === 'model') out.push([x, mm && mm.iv !== null ? mm.iv : null]);
      else if (f === 'fit' || f === 'oos') { if (mm && mm.fit === (f === 'fit')) out.push([x, q.mid]); }
      else if ((f === 'z' || f === 'zfit') && mm && mm.z !== null && (f === 'z' || mm.fit)) {
        out.push([x, Math.max(-clip, Math.min(clip, mm.z)), mm.z]);
      }
    });
    return out;
  }
  // the quote series of expiry f (no f: every group's) in one x coordinate: 'k' = ln(K/F), 'delta' = N(−d₁)
  function fillQuotes(spec, xmode, f) {
    spec._xmode = xmode;
    var Q = spec.quotes, clip = Q.zClip || 10, all = f === undefined || f === null;
    (spec.series || []).forEach(function (s) {
      if (s.quote && (Q.template || all || String(s.group) === String(f))) s.data = quotePoints(rowsOf(spec, Q.template ? f : s.group), s, xmode, clip);
    });
  }
  // the group a quote chart shows: the tab's, or its first
  function quoteGroup(spec, f) { return f !== undefined && f !== null ? String(f) : Object.keys(spec.quotes.groups)[0]; }
  // one expiry's quotes, decoded when first used (all 45 at load: 174 ms at 4× CPU)
  function rowsOf(spec, f) {
    var g = quoteGroup(spec, f), b = spec.quotes.groups[g];
    return spec._rows[g] || (b ? spec._rows[g] = decodeQuotes(b) : []);
  }
  Desk.rowsOf = rowsOf;
  // a same-origin data file (named by its sha256), checked by its sha384, schema and
  // session; fetched once, and again after a failure
  var files = {};
  Desk.data = function (src, sri, session) {
    if (!files[src]) {
      files[src] = fetch(src, { integrity: sri }).then(function (r) { if (!r.ok) throw Error('HTTP ' + r.status); return r.json(); })
        .then(function (j) { if (j.schema !== 1 || j.session !== session) throw Error('another session’s file'); return j; });
      files[src].catch(function () { delete files[src]; });
    }
    return files[src];
  };
  // a book (Surface): the page holds the default expiry and one series per role
  // (quotes.template); its file, quotes.src, every expiry of quotes.order
  function loadBook(spec) {
    var Q = spec.quotes;
    return !Q.src ? Promise.resolve(spec) : Desk.data(Q.src, Q.integrity, Q.session).then(function (j) {
      if (j.quotes.order.join() !== Q.order.join()) throw Error('other expiries');
      Q.order.forEach(function (g) { Q.groups[g] = Q.groups[g] || j.quotes.groups[g]; });
      return spec;
    });
  }
  function hasRows(spec, f) { return !!spec.quotes.groups[quoteGroup(spec, f)]; }
  // no file (a tab open across a deploy, a file:// preview): a note, and the expiry's table
  function bookFailed(id, err) {
    var n = doc.getElementById(id + '-note'), t = doc.getElementById(id + '-data');
    if (n) n.textContent = err ? 'This expiry’s quotes did not load (' + err.message + '): its medians are in the table below. Reload the page to try again.' : '';
    if (err && t && !t.classList.contains('show')) $('[data-table-for="' + id + '-data"]').click();
  }
  Desk.quotesReady = function (id) {
    var c = Desk.charts[id], el = doc.getElementById(id), spec = c ? c.spec : el && specOf(el);
    return spec && spec.quotes ? loadBook(spec).then(function () { return true; }) : Promise.resolve(false);
  };
  // the x range of the shown group (the smile and its residual strip): the quotes'
  // extent padded and rounded out to a tick step; a spec's x.min / x.max wins
  function quoteRange(spec, f) {
    if (spec._xmode === 'delta') return [0, 1];
    var x = spec.x || {};
    if (x.min !== undefined && x.min !== null && x.max !== undefined && x.max !== null) return [x.min, x.max];
    var rows = rowsOf(spec, f);
    if (!rows.length) return [null, null];
    var lo = rows[0].k, hi = rows[rows.length - 1].k, span = Math.max(hi - lo, 0.02);
    var raw = span / 5, mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10)), step = mag;
    [1, 2, 2.5, 5, 10].some(function (m) { step = m * mag; return step >= raw; });
    return [Math.floor((lo - 0.02 * span) / step) * step, Math.ceil((hi + 0.02 * span) / step) * step];
  }
  // the forward-delta axis's labels: N(−d₁) of 0.10 is the 10-delta put, of 0.90 the 10-delta call
  var DELTA_TICKS = [0.10, 0.25, 0.50, 0.75, 0.90];
  var CLIP_NAME = ' beyond the clip';
  function deltaLabel(v) {
    if (Math.abs(v - 0.5) < 1e-9) return '50\u0394';
    return v < 0.5 ? Math.round(v * 100) + '\u0394P' : Math.round((1 - v) * 100) + '\u0394C';
  }
  // the quote nearest to x (the pointer's axis value) among the shown group's quotes
  function nearestQuote(spec, f, x) {
    var rows = rowsOf(spec, f), best = null, bd = Infinity;
    rows.forEach(function (q) { var d = Math.abs((spec._xmode === 'delta' ? q.xd : q.k) - x); if (d < bd) { bd = d; best = q; } });
    return best;
  }
  // what the inspector says about one quote: the tooltip draws these lines and
  // the keyboard readout joins them with ' · ', so both say the same thing
  function quoteLines(spec, q, off) {
    var Q = spec.quotes, lines = [];
    var pct = function (v) { return num(v, { digits: 2, unit: '%' }); };
    lines.push({ head: true, text: 'K ' + num(q.strike, { digits: 0 }) + ' ' + q.right + ' \u00b7 ln(K/F) ' + num(q.k, { digits: 3, signed: true })
      + ' \u00b7 \u0394 ' + num(q.delta, { digits: 2, signed: true }) + ' \u00b7 ' + num(q.T, { digits: 0 }) + ' d' });
    lines.push({ text: 'Bid / mid / ask ' + num(q.bid, { digits: 2 }) + ' / ' + num(q.mid, { digits: 2 }) + ' / ' + pct(q.ask)
      + ' \u00b7 half-spread ' + num(q.hs, { digits: 2 }) + ' vol pts' });
    (Q.models || []).forEach(function (m) {
      var mm = q.models[m.key];
      if (!mm || off[m.label]) return;
      lines.push({ color: m.color, text: (m.short || m.label) + ' ' + (mm.iv === null ? 'no value' : pct(mm.iv) + ' \u00b7 z '
        + num(mm.z, { digits: 1, signed: true }) + ' \u00b7 ' + (mm.inside ? '\u25cf inside' : '\u25cb outside'))
        + (mm.fit && m.key !== Q.anchor ? ' \u00b7 fitted to it' : '') });     // the anchor model's: a flag below
    });
    var flags = [];
    if (Q.anchor && q.models[Q.anchor] && q.models[Q.anchor].fit) flags.push((Q.anchorLabel || 'fit') + ' anchor');
    if (q.exec) flags.push('in an executable arbitrage row');
    if (q.outside) flags.push('moved outside its bid\u2013ask by the arbitrage repair');
    else if (q.repaired) flags.push('moved inside its bid\u2013ask by the arbitrage repair');
    if (flags.length) lines.push({ flag: true, text: flags.join(' \u00b7 ') });
    if (Q.source) lines.push({ foot: true, text: Q.source });
    return lines;
  }
  function quoteHtml(lines, T) {
    return lines.map(function (l) {
      var st = l.head ? 'color:' + T.text3 + ';margin-bottom:2px' : (l.flag ? 'color:' + T.text2 : (l.foot ? 'color:' + T.text3 + ';font-size:10px;margin-top:2px' : ''));
      var sw = l.color ? '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:' + color(l.color, T) + ';margin-right:6px"></span>' : '';
      return '<div style="' + st + '">' + sw + l.text + '</div>';
    }).join('');
  }
  function quoteText(lines) { return lines.map(function (l) { return l.text; }).join(' \u00b7 '); }

  // -- 'S_T', 'g_min' as the tables set them (app.subs, the same pattern): as
  //    rich text on the canvas (rich), as <sub> in a tooltip ----
  var SUBS = /\b(S|g|VIX|RV)_(T|min|h)\b/g;
  function subs(t, rich) {
    return String(t).replace(SUBS, function (m, a, b) {
      return (a.length > 1 ? a : rich ? '{i|' + a + '}' : '<i>' + a + '</i>') + (rich ? '{s|' + b + '}' : '<sub>' + b + '</sub>');
    });
  }

  // -- spec -> ECharts option ----
  function axis(a, T, isX, spec) {
    a = a || {};
    var type = a.type || 'value';
    var ax = {
      type: type, name: subs(a.name || '', 1), nameLocation: 'middle', nameGap: isX ? 26 : 42,
      nameTextStyle: { color: T.text3, fontFamily: T.sans, fontSize: 11, rich: { i: { fontStyle: 'italic' }, s: { fontSize: 8, padding: [5, 0, 0, 0] } } },
      axisLine: { show: isX, lineStyle: { color: T.line2 } }, axisTick: { show: false },
      axisLabel: { color: T.text3, fontFamily: T.mono, fontSize: 10.5, hideOverlap: true },
      splitLine: { show: !isX || type === 'value', lineStyle: { color: T.grid } },
      scale: a.zero === undefined ? type === 'value' : !a.zero, inverse: !!a.inverse,
    };
    if (a.step) ax.interval = a.step;          // one tick step across small multiples
    if (a.min !== undefined && a.min !== null) ax.min = a.min;
    if (a.max !== undefined && a.max !== null) ax.max = a.max;
    if (type === 'category') {
      ax.data = a.categories || spec.xs || []; ax.axisLabel.interval = a.interval === undefined ? 'auto' : a.interval; ax.splitLine.show = false;
      if (a.rotate) { ax.axisLabel.rotate = a.rotate; ax.axisLabel.width = a.labelWidth || 120; ax.axisLabel.overflow = 'truncate'; ax.axisLabel.hideOverlap = false; }
    }
    else if (type === 'value') ax.axisLabel.formatter = axisFmt(a);
    else if (type === 'time') { ax.axisLabel.formatter = { year: '{yyyy}', month: '{MMM}', day: '{MM}-{dd}' }; ax.splitLine.show = false; }
    if (a.log) { ax.type = 'log'; }
    return ax;
  }

  function seriesData(s, spec) {
    if (s.xFrom && s.data) {          // y only, at the x values of another series (same quotes)
      var ref = (spec.series || []).filter(function (r) { return r.name === s.xFrom; })[0];
      var rx = ref ? ref.data : [];
      return s.data.map(function (y, i) { return [rx[i] ? rx[i][0] : null, y]; });
    }
    if (spec.xs && s.data && s.data.length && !Array.isArray(s.data[0])) {
      return s.data.map(function (y, i) { return [spec.xs[i], y]; });
    }
    return s.data || [];
  }

  function buildSeries(s, spec, T, i) {
    var c = color(s.color || ('cat' + (i % 6)), T);
    var base = { name: s.name, z: s.z || 2, yAxisIndex: s.y2 || s.strip ? 1 : 0, emphasis: { focus: 'none' } };
    // a quote smile's tooltip follows the axis, not a hovered item: its series
    // need no hover states (fewer elements to build on every redraw)
    if (spec.quotes) { base.silent = true; base.emphasis = { disabled: true }; }
    if (s.strip) base.xAxisIndex = 1;               // the residual strip under a quote smile
    var data = seriesData(s, spec);
    // a quote smile's bid–ask bars: one line series, each bar its caps and a break
    // (quoteOption); a custom series built an element per quote on every switch
    if (s.type === 'errorbar' && spec.quotes) {
      return Object.assign(base, { type: 'line', data: data, _bars: true, showSymbol: false, connectNulls: false,
        lineStyle: { color: c, width: 1.2 }, itemStyle: { color: c } });
    }
    if (s.type === 'errorbar') {
      return Object.assign(base, {
        type: 'custom', data: data, encode: { x: 0, y: [1, 2] }, tooltip: { show: true },
        renderItem: function (params, api) {
          var x = api.value(0), lo = api.coord([x, api.value(1)]), hi = api.coord([x, api.value(2)]);
          var w = 3;
          // one polyline per bar: lower cap, the bar, upper cap (a quarter of the elements of three lines)
          return { type: 'polyline', style: { stroke: c, lineWidth: 1.2, fill: null }, shape: { points: [
            [lo[0] - w, lo[1]], [lo[0] + w, lo[1]], [lo[0], lo[1]], [hi[0], hi[1]], [hi[0] - w, hi[1]], [hi[0] + w, hi[1]]] } };
        },
      });
    }
    if (s.type === 'scatter') {
      // a residual-strip point beyond ±clip: a triangle at the edge pointing the way
      // it is cut off (a series of its own on a model's strip, which keeps a gap there)
      var clipped = null;
      if (s.strip && spec.quotes) {
        var clip = spec.quotes.zClip || 10;
        var tri = function (p) { return { value: p, symbol: 'triangle', symbolRotate: p[2] < 0 ? 180 : 0, symbolSize: (s.size || 4) + 3 }; };
        if (s.quote === 'z') {
          clipped = data.filter(function (p) { return Math.abs(p[2]) > clip; }).map(tri);
          data = data.map(function (p) { return Math.abs(p[2]) > clip ? [p[0], null, p[2]] : p; });
        } else data = data.map(function (p) { return Math.abs(p[2]) > clip ? tri(p) : p; });
      }
      var sc = Object.assign(base, {
        type: 'scatter', data: data, symbol: s.symbol || 'circle', symbolSize: s.size || 6, symbolRotate: s.rotate || 0,
        itemStyle: s.hollow ? { color: 'transparent', borderColor: c, borderWidth: 1.4, opacity: s.opacity || 1 } : { color: c, opacity: s.opacity || 0.9 },
      });
      // a quote smile's mids and z strips: one path per series (large), not when empty
      // (it fails to paint) nor for the fit anchors (a series changing mode is rebuilt)
      if (spec.quotes) {
        sc.large = (s.quote === 'mid' || s.quote === 'z') && data.length > 0 && !data.some(function (p) { return p && !Array.isArray(p); });
        sc.largeThreshold = 0;
      }
      if (clipped) {
        sc._clip = Object.assign({}, sc, { name: s.name + CLIP_NAME, data: clipped, large: false });
        delete sc._clip._clip;
      }
      // a short label at each point (the VX leg's model − settlement gap)
      if (s.pointLabels) {
        sc.label = { show: true, position: s.labelPos || 'right', distance: 4, color: T.text3, fontFamily: T.mono, fontSize: 10,
          formatter: function (p) { var l = s.pointLabels[p.dataIndex]; return l === null || l === undefined ? '' : String(l); } };
        sc.labelLayout = { hideOverlap: true };
      }
      return sc;
    }
    if (s.type === 'bar') {
      var bar = Object.assign(base, { type: 'bar', data: data, barMaxWidth: s.barMaxWidth || 28, stack: s.stack || undefined });
      if (s.invisible) { bar.itemStyle = { color: 'transparent' }; bar.tooltip = { show: false }; bar.silent = true; return bar; }
      if (s.signed) {
        // a horizontal chart's points are [value, category] (buildOption swaps them)
        var hz = !!spec.horizontal;
        bar.itemStyle = { color: function (p) { var v = Array.isArray(p.value) ? p.value[hz ? 0 : 1] : p.value; return v < 0 ? T.colors.neg : T.colors.pos; } };
      } else { bar.itemStyle = { color: s.fill ? alpha(c, s.fill) : c, borderColor: c, borderWidth: s.fill ? 1 : 0 }; }
      return bar;
    }
    // line / area
    var line = Object.assign(base, {
      type: 'line', data: data, showSymbol: !!s.symbol, symbol: s.symbol || 'circle', symbolSize: s.size || 5,
      connectNulls: false, step: s.step || false, sampling: s.sampling || undefined,
      lineStyle: { color: c, width: s.width || 1.5, type: s.dash === 'dotted' ? 'dotted' : (s.dash ? 'dashed' : 'solid') },
      itemStyle: { color: c },
    });
    if (s.area) line.areaStyle = { color: alpha(c, 0.18), origin: s.areaOrigin || 'auto' };
    // a short name at the line's last shown point (the equity curves: 'S3', 'T-bill')
    if (s.endLabel) line.endLabel = { show: true, formatter: s.endLabel, color: c, fontFamily: T.mono, fontSize: 10, distance: 4 };
    // end labels that would touch are pushed apart vertically, never hidden (the cone's p5…p95)
    if (s.endLabel && s.endLabelDodge) line.labelLayout = { moveOverlap: 'shiftY' };
    // per-point marks (the σ frontier): a hollow marker (s.hollowPoints[j]), a
    // larger one (s.bigPoints[j]) and a short label beside each point (s.pointLabels)
    if (s.hollowPoints || s.bigPoints) {
      line.showSymbol = true;
      line.data = data.map(function (pt, j) {
        var o = { value: pt };
        if (s.hollowPoints && s.hollowPoints[j]) o.itemStyle = { color: T.surface, borderColor: c, borderWidth: 1.4 };
        if (s.bigPoints && s.bigPoints[j]) o.symbolSize = (s.size || 5) + 4;
        return o;
      });
    }
    // s.tipLabels: the labels go to the tooltip only (the σ frontier's other sessions)
    if (s.pointLabels && !s.tipLabels) {
      line.label = { show: true, position: 'right', distance: 3, color: T.text3, fontFamily: T.mono, fontSize: 10,
        formatter: function (p) { var l = s.pointLabels[p.dataIndex]; return l === null || l === undefined ? '' : String(l); } };
      line.labelLayout = { hideOverlap: true };
    }
    return line;
  }

  // The series that carries the mark lines: on a tab-filtered chart the first of
  // the group on show (never one the filter hides).
  function markTarget(spec, filter) {
    var t = 0;
    if (filter !== undefined) {
      (spec.series || []).some(function (s, i) {
        if (s.group !== undefined && String(s.group) === String(filter)) { t = i; return true; }
        return false;
      });
    }
    return t;
  }
  // the spec indices of the series a filtered chart draws: its group's and the ungrouped
  function groupIdx(spec, f) {
    var out = [];
    (spec.series || []).forEach(function (s, i) {
      if (f === undefined || s.group === undefined || String(s.group) === String(f)) out.push(i);
    });
    return out;
  }
  function marks(spec, T, opt, filter) {
    var m = spec.marks || {};
    if (!opt.series.length) return;
    var pos = opt._pos ? opt._pos[markTarget(spec, filter)] : markTarget(spec, filter);
    var target = opt.series[pos || 0];
    // on the forward-delta axis the forward has no one place (50Δ is drawn as a grid line)
    var mx = spec._xmode === 'delta' ? [] : (m.x || []);
    var dl = spec._xmode === 'delta' ? DELTA_TICKS : [];
    if (m.y || mx.length || dl.length) {
      var data = dl.map(function (v) { return { xAxis: v, name: '', lineStyle: { color: T.line2, type: 'dotted', width: 1 } }; });
      (m.y || []).forEach(function (l) { data.push({ yAxis: l.value, name: l.label || '', lineStyle: { color: color(l.color || 'muted', T), type: l.dash === false ? 'solid' : 'dashed', width: 1 } }); });
      mx.forEach(function (l) { data.push({ xAxis: l.value, name: l.label || '', lineStyle: { color: color(l.color || 'muted', T), type: 'dashed', width: 1 } }); });
      target.markLine = { symbol: 'none', silent: true, animation: false, data: data,
        label: { show: true, position: 'insideEndTop', color: T.text3, fontFamily: T.mono, fontSize: 10, formatter: '{b}' } };
    }
    if (m.areas) {
      target.markArea = { silent: true, data: m.areas.map(function (a) {
        return [{ xAxis: a.from, name: a.label || '', itemStyle: { color: alpha(color(a.color || 'breach', T), a.alpha || 0.08) } }, { xAxis: a.to }];
      }), label: { color: T.text3, fontFamily: T.mono, fontSize: 10, position: 'insideTop' } };
    }
  }

  // The tooltip grammar (P4.5): the head is the x value with its unit and x.tipNames'
  // name ('VIX3M \u00b7 93 d'); a row carries its axis's unit or its name's '(unit)'.
  var UNIT_WORDS = { '%': '%', 'vol pts': ' vol pts', 'index pts': ' index pts', 'index points': ' index points',
    'half-spreads': ' half-spreads', 'calendar days': ' d', 'days': ' d', '$000s': ' $000s', 'bp': ' bp' };
  function axisUnit(a) {
    if (a.unit) return a.unit;
    var m = /\(([^()]*)\)\s*$/.exec(a.name || '');
    return m && UNIT_WORDS[m[1].trim()] !== undefined ? UNIT_WORDS[m[1].trim()] : (/\bdays\b/i.test(a.name || '') ? ' d' : '');
  }
  Desk.axisUnit = axisUnit;
  function tooltipFor(spec, T) {
    var y = spec.y || {}, x = spec.x || {};
    var fy = function (v, s) {
      var a = (s && s.y2) ? (spec.y2 || y) : y;
      return num(v, { digits: a.tipDigits === undefined ? (a.digits === undefined ? 2 : a.digits + 1) : a.tipDigits, signed: a.signed, unit: axisUnit(a), prefix: a.prefix });
    };
    var xu = axisUnit(x);
    var fx = function (v) {
      if (x.type !== 'value') return String(v);
      // days are whole: '93 d', not '93.0'
      var dg = x.tipDigits !== undefined ? x.tipDigits : (xu === ' d' && Math.round(v) === Number(v) ? 0 : (x.digits === undefined ? 2 : x.digits + 1));
      var name = x.tipNames && x.tipNames[String(Math.round(v * 1e6) / 1e6)];
      return (name ? name + ' \u00b7 ' : '') + num(v, { digits: dg, signed: x.signed, unit: xu });
    };
    var specSeries = {};
    (spec.series || []).forEach(function (s) { specSeries[s.name] = s; });
    var hz = !!spec.horizontal;          // a horizontal chart's points are [value, category]
    var tipNum = function (v) { return num(v, { digits: y.tipDigits === undefined ? 2 : y.tipDigits, signed: true, unit: axisUnit(y) }); };
    return {
      trigger: spec.tooltip || 'axis', confine: true,
      axisPointer: { type: spec.tooltip === 'item' ? 'none' : (hz ? 'shadow' : 'line'), lineStyle: { color: T.line2 } },
      backgroundColor: T.tooltip, borderColor: T.line2, borderWidth: 1, padding: [6, 8],
      textStyle: { color: T.text, fontFamily: T.mono, fontSize: 11 },
      formatter: function (ps) {
        ps = Array.isArray(ps) ? ps : [ps];
        var head = '', rows = [], idx = null;
        ps.forEach(function (p) {
          var s = specSeries[p.seriesName] || {};
          if (s.invisible) return;
          var v = p.value;
          var xv = Array.isArray(v) ? v[hz ? 1 : 0] : p.name;
          if (!head) head = (x.label ? x.label + ' ' : '') + fx(xv);
          if (idx === null) idx = p.dataIndex;
          // an axis tooltip lists every series at the index, also those with no
          // value there (a stacked step bar is null where the other one draws)
          var yv = Array.isArray(v) ? (s.type === 'errorbar' ? v[1] : v[hz ? 0 : 1]) : v;
          if (yv === null || yv === undefined || yv === '-' || (typeof yv === 'number' && isNaN(yv))) return;
          var txt;
          if (s.tip) { if (s.tip[p.dataIndex] === null || s.tip[p.dataIndex] === undefined) return; txt = tipNum(s.tip[p.dataIndex]); }
          else if (s.type === 'errorbar') txt = fy(v[1], s) + ' \u2013 ' + fy(v[2], s);
          else txt = fy(yv, s);
          var sw = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:' + (typeof p.color === 'string' ? p.color : T.text2) + ';margin-right:6px"></span>';
          var pl = s.pointLabels && s.pointLabels[p.dataIndex] !== undefined && s.pointLabels[p.dataIndex] !== null
            ? ' \u00b7 ' + (s.pointPrefix || '') + s.pointLabels[p.dataIndex] : '';
          rows.push(sw + subs(p.seriesName) + pl + '&nbsp;&nbsp;<b>' + txt + '</b>');
        });
        // extra lines that belong to the index, not to a series (a waterfall's level after the step)
        (spec.tipRows || []).forEach(function (r) {
          var v = idx === null ? null : r.data[idx];
          if (v !== null && v !== undefined) rows.push(subs(r.name) + '&nbsp;&nbsp;<b>' + num(v, { digits: r.digits === undefined ? 1 : r.digits, signed: !!r.signed, unit: r.unit || '' }) + '</b>');
        });
        return '<div style="color:' + T.text3 + ';margin-bottom:2px">' + head + '</div>' + rows.join('<br>');
      },
    };
  }

  // -- a horizontal chart's long category names wrap at spaces into their band (a
  // column up to labelShare, labelMax px); one needing more lines takes spec.labelsShort,
  // then the column widens to 0.6 of the chart. No ellipsis --
  var LABEL_SIZE = 10.5, LABEL_LINE = 13, measureCtx = null;
  function textWidth(s, font) {
    if (!measureCtx) measureCtx = doc.createElement('canvas').getContext('2d');
    measureCtx.font = font;
    return measureCtx.measureText(s).width;
  }
  function wrapWords(s, w, font) {
    var lines = [], cur = '';
    String(s).split(' ').forEach(function (word) {
      var t = cur ? cur + ' ' + word : word;
      if (!cur || textWidth(t, font) <= w) cur = t; else { lines.push(cur); cur = word; }
    });
    if (cur) lines.push(cur);
    return lines;
  }
  function labelLayout(spec, T, width, height, grid) {
    var cats = ((spec.x && spec.x.categories) || spec.xs || []).map(String);
    var short = spec.labelsShort || [];
    var font = LABEL_SIZE + 'px ' + (spec.labelFont === 'sans' ? T.sans : T.mono);
    var band = (height - grid.top - grid.bottom) / Math.max(1, cats.length);
    var maxLines = Math.max(1, Math.floor(band / LABEL_LINE));
    var cap = Math.max(90, Math.min(spec.labelMax || 300, Math.round(width * spec.labelShare)));
    var widest = 0;
    cats.forEach(function (c) { widest = Math.max(widest, textWidth(c, font)); });
    function lay(w) {
      var used = [];
      var lines = cats.map(function (c, i) {
        var l = wrapWords(c, w, font);
        if (l.length > maxLines && short[i] && short[i] !== c) { l = wrapWords(short[i], w, font); used.push(i); }
        return l;
      });
      var most = Math.max.apply(null, lines.map(function (l) { return l.length; }));
      return { width: w, lines: lines, short: used, most: most };
    }
    var w = Math.min(cap, Math.ceil(widest) + 1), L = lay(w);
    var wMax = Math.max(w, Math.min(spec.labelMax || 300, Math.round(width * 0.6)));
    while (L.most > maxLines && w < wMax) { w = Math.min(wMax, w + 4); L = lay(w); }
    L.font = font; L.size = LABEL_SIZE; L.lineHeight = LABEL_LINE; L.band = band; L.maxLines = maxLines;
    L.text = L.lines.map(function (l) { return l.join('\n'); });
    L.widest = Math.max.apply(null, L.lines.map(function (l) {
      return Math.max.apply(null, l.map(function (x) { return textWidth(x, font); }));
    }));
    return L;
  }

  // -- period labels under vertical bars (x.fit) wrap at spaces; if the widest line
  // still does not fit, every step-th label is drawn back from the newest. None
  // overlaps or is cut --
  var LABEL_GAP = 6;
  function xLabelLayout(spec, T, width, grid) {
    var cats = ((spec.x && spec.x.categories) || spec.xs || []).map(String);
    var font = LABEL_SIZE + 'px ' + T.mono;
    var band = (width - grid.left - grid.right) / Math.max(1, cats.length);
    var lines = cats.map(function (c) {
      return textWidth(c, font) + LABEL_GAP <= band ? [c] : wrapWords(c, band - LABEL_GAP, font);
    });
    var widest = Math.max.apply(null, lines.map(function (l) {
      return Math.max.apply(null, l.map(function (x) { return textWidth(x, font); }));
    }));
    return { band: band, lines: lines, text: lines.map(function (l) { return l.join('\n'); }),
             step: Math.max(1, Math.ceil((widest + LABEL_GAP) / band)), widest: widest,
             most: Math.max.apply(null, lines.map(function (l) { return l.length; })) };
  }

  function buildOption(spec, T, filter, width, height, out, off) {
    var opt = {
      animation: false, backgroundColor: 'transparent',
      textStyle: { fontFamily: T.sans, color: T.text2 },
      grid: Object.assign({ left: 54, right: spec.y2 ? 54 : 18, top: 14, bottom: spec.zoom && spec.zoom !== 'inside' ? 58 : 36, containLabel: false }, spec.grid || {}),
      xAxis: axis(spec.x, T, true, spec),
      yAxis: [axis(spec.y, T, false, spec)],
      tooltip: tooltipFor(spec, T),
      legend: { show: false },
      series: [],
    };
    // on a tab-filtered chart only the shown group's series are drawn
    opt._idx = groupIdx(spec, filter);
    // A grouped series is named by its place in the group ('g0', …) and draw mode, so a
    // switch merges into the series drawn; a quote smile leaves out what is toggled off
    // and empty clip triangles. opt._pos: a spec index's place among the drawn
    var slot = 0, drop = spec.quotes && off ? off : {};
    opt._pos = {};
    opt._idx.forEach(function (i) {
      var s = spec.series[i];
      if (drop[legendKey(s)]) return;
      var b = buildSeries(s, spec, T, i), cl = b._clip;
      b.id = (filter !== undefined && s.group !== undefined ? 'g' + (slot++) : 's' + i) + (b.large ? 'L' : '');
      opt._pos[i] = opt.series.length;
      opt.series.push(b);
      delete b._clip;
      if (cl && cl.data.length) { cl.id = b.id + 'c'; opt.series.push(cl); }
    });
    if (spec.y2) { var a2 = axis(spec.y2, T, false, spec); a2.splitLine = { show: false }; opt.yAxis.push(a2); }
    if (!spec.horizontal && spec.x && spec.x.fit && width) {
      var X = xLabelLayout(spec, T, width, opt.grid), last = X.lines.length - 1, extra = (X.most - 1) * LABEL_LINE;
      Object.assign(opt.xAxis.axisLabel, {
        hideOverlap: false, fontSize: LABEL_SIZE, lineHeight: LABEL_LINE,
        interval: function (i) { return (last - i) % X.step === 0; },
        formatter: function (v, i) { return X.text[i] !== undefined ? X.text[i] : v; } });
      opt.grid.bottom += extra; opt.xAxis.nameGap += extra;
      if (out) out.xLabels = X;
    }
    if (spec.horizontal) {
      var t = opt.xAxis; opt.xAxis = opt.yAxis[0]; opt.yAxis = [t];
      opt.xAxis.axisLine = { show: true, lineStyle: { color: T.line2 } }; opt.yAxis[0].inverse = true;
      opt.yAxis[0].axisLabel.rotate = 0; opt.yAxis[0].nameGap = 0;
      opt.series.forEach(function (s) { s.data = (s.data || []).map(function (d) { return Array.isArray(d) ? [d[1], d[0]] : d; }); });
      // long category labels (a waterfall's step names): wrapped, never truncated
      if (spec.labelShare && width) {
        var L = labelLayout(spec, T, width, height || 0, opt.grid);
        opt.grid.left = L.width + 10;
        Object.assign(opt.yAxis[0].axisLabel, {
          width: L.width + 2, overflow: 'none', lineHeight: L.lineHeight, hideOverlap: false, interval: 0,
          fontSize: L.size, fontFamily: spec.labelFont === 'sans' ? T.sans : T.mono,
          formatter: function (v, i) { return L.text[i] !== undefined ? L.text[i] : v; } });
        if (out) out.labels = L;
      }
    }
    if (spec.zoom === 'inside') {          // wheel / drag only: a chart that zooms with a linked one
      opt.dataZoom = [{ type: 'inside', xAxisIndex: 0, filterMode: 'none' }];
    } else if (spec.zoom) {
      opt.dataZoom = [
        { type: 'inside', xAxisIndex: 0, filterMode: 'none' },
        { type: 'slider', xAxisIndex: 0, height: 16, bottom: 8, borderColor: T.line2, fillerColor: alpha(T.colors.info, 0.12),
          handleStyle: { color: T.text3, borderColor: T.line2 }, moveHandleStyle: { color: T.line2 },
          dataBackground: { lineStyle: { color: T.line2 }, areaStyle: { color: alpha(T.colors.muted, 0.15) } },
          textStyle: { color: T.text3, fontFamily: T.mono, fontSize: 10 }, labelFormatter: function (v, s) { return s; } },
      ];
    }
    if (spec.quotes) quoteOption(spec, T, opt, filter, width, height);
    var cur = cursorSeries(spec, T);
    if (cur) opt.series.push(cur);
    marks(spec, T, opt, filter);
    return opt;
  }

  // The selected expiry on a chart of expiries (spec.exps, #chart-by-expiry): a
  // line at its category, a series of its own (merged by id)
  function cursorSeries(spec, T) {
    if (!spec.exps || Desk.expCursor === undefined) return null;
    var i = spec.exps.indexOf(Desk.expCursor), cats = (spec.x && spec.x.categories) || spec.xs || [];
    return { id: 'cursor', name: '\u00a0selected expiry', type: 'line', data: [], silent: true, tooltip: { show: false }, z: 1,
      markLine: { silent: true, symbol: 'none', animation: false, label: { show: false },
        lineStyle: { color: T.text2, type: 'solid', width: 1.5 },
        data: i < 0 || cats[i] === undefined ? [] : [{ xAxis: cats[i] }] } };
  }

  // A quote smile: the inspector's tooltip, the expiry's x range, the delta axis and
  // the residual strip (each shown model's z on a second grid; ±1 shaded, ±2 dashed).
  function quoteOption(spec, T, opt, filter, width, height) {
    var Q = spec.quotes, id = spec.id, delta = spec._xmode === 'delta';
    var rng = quoteRange(spec, filter);
    var x0 = opt.xAxis;
    x0.min = rng[0]; x0.max = rng[1]; x0.scale = false;
    // the bid–ask bars' caps: 3 px either side, in x units; none under 8 px a quote
    // (they would touch, and cost a fifth of a switch at 4× CPU)
    var plotW = Math.max(40, (width || 600) - (opt.grid.left || 0) - (opt.grid.right || 0));
    var cap = rng[0] === null || plotW < 8 * rowsOf(spec, filter).length ? 0 : 3 * (rng[1] - rng[0]) / plotW;
    opt.series.forEach(function (s) {
      if (!s._bars) return;
      delete s._bars;
      var pts = [];
      s.data.forEach(function (p) {
        var x = p[0], lo = p[1], hi = p[2];
        if (x === null || lo === null || hi === null) return;
        if (cap) pts.push([x - cap, lo], [x + cap, lo]);
        pts.push([x, lo], [x, hi]);
        if (cap) pts.push([x - cap, hi], [x + cap, hi]);
        pts.push([x, null]);
      });
      s.data = pts;
    });
    if (delta) {
      x0.name = 'Forward delta';
      x0.axisLabel.formatter = deltaLabel; x0.axisLabel.customValues = DELTA_TICKS;
      x0.axisTick = { show: false, customValues: DELTA_TICKS };
      x0.splitLine = { show: false };              // grid lines at the labelled deltas: marks()
    }
    opt.tooltip = {
      trigger: 'axis', confine: true, axisPointer: { type: 'line', lineStyle: { color: T.line2 } },
      backgroundColor: T.tooltip, borderColor: T.line2, borderWidth: 1, padding: [6, 8],
      textStyle: { color: T.text, fontFamily: T.mono, fontSize: 11 },
      extraCssText: 'white-space:normal;max-width:calc(100% - 8px);',    // wraps inside a phone-width chart
      formatter: function (ps) {
        ps = Array.isArray(ps) ? ps : [ps];
        if (!ps.length) return '';
        var c = Desk.charts[id], sp = c ? c.spec : spec, f = Desk.filters[id];
        var q = nearestQuote(sp, f, ps[0].axisValue);
        if (!q) return '';
        var lines = quoteLines(sp, q, legendOff(id));
        if (c && c.qi !== q.i) { c.qi = q.i; showReadout(id, lines); }      // only when the quote changes
        return quoteHtml(lines, T);
      },
    };
    if (!Q.strip) return;
    // the pointer and the tooltip follow the x value on both grids
    opt.axisPointer = { link: [{ xAxisIndex: 'all' }] };
    // gap: the upper axis's bottom label and the strip's +clip label each overhang
    // their grid edge by half a line, so 10 px set them touching
    var g0 = opt.grid, sh = Math.max(44, Math.min(96, Math.round((height || 300) * 0.22))), gap = 20;
    // both grids are placed by their margins: ECharts 6's outer-bounds pass
    // (measuring every axis label to keep them in the box) cost a tenth of each update
    opt.grid = [Object.assign({}, g0, { bottom: g0.bottom + sh + gap, outerBoundsMode: 'none' }),
      { left: g0.left, right: g0.right, bottom: g0.bottom, height: sh, containLabel: false, outerBoundsMode: 'none' }];
    var x1 = JSON.parse(JSON.stringify(x0));
    if (delta) { x1.axisLabel.formatter = deltaLabel; }
    else x1.axisLabel.formatter = axisFmt(spec.x || {});
    x1.gridIndex = 1;
    x0.axisLabel = Object.assign({}, x0.axisLabel, { show: false }); x0.name = '';
    opt.xAxis = [x0, x1];
    var clip = Q.zClip || 10;
    var y1 = axis({ name: 'z', digits: 0, signed: true }, T, false, spec);
    Object.assign(y1, { gridIndex: 1, min: -clip, max: clip, interval: clip, scale: false, nameGap: 30 });
    opt.yAxis.push(y1);
    // the bands: a series with no data of its own carries them
    opt.series.push({ id: 'zband', name: '\u00a0z band', type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: [], silent: true,
      tooltip: { show: false }, z: 1,
      markArea: { silent: true, data: [[{ yAxis: -1, name: 'inside the spread', itemStyle: { color: alpha(T.colors.muted, 0.16) } }, { yAxis: 1 }]],
        label: { show: true, position: 'insideLeft', color: T.text3, fontFamily: T.mono, fontSize: 10,
          backgroundColor: alpha(T.surface, 0.85), padding: [0, 3] } },
      markLine: { silent: true, symbol: 'none', animation: false, label: { show: false },
        data: [{ yAxis: 2 }, { yAxis: -2 }].concat(delta ? DELTA_TICKS.map(function (v) {
          return { xAxis: v, lineStyle: { color: T.line2, type: 'dotted', width: 1 } }; }) : []),
        lineStyle: { color: T.line2, type: 'dashed', width: 1 } } });
  }

  // -- CSV export (from the same spec the chart draws) ----
  function csvCell(v) {
    if (v === null || v === undefined) return '';
    var s = String(v);
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }
  function specCsv(spec) {
    var lines = [];
    // a chart whose drawn encoding is not its data (a waterfall's stacked bars)
    // carries the rows to export
    if (spec.csv_rows) {
      lines.push(spec.csv_rows.columns.map(csvCell).join(','));
      spec.csv_rows.rows.forEach(function (r) { lines.push(r.map(csvCell).join(',')); });
      return lines.join('\n') + '\n';
    }
    if (spec._rows) return quotesCsv(spec);
    var xn = (spec.x && (spec.x.csv || spec.x.name)) || 'x';
    var series = (spec.series || []).filter(function (s) { return !s.invisible; });
    if (spec.xs) {
      lines.push([xn].concat(series.map(function (s) { return s.name; })).map(csvCell).join(','));
      spec.xs.forEach(function (x, i) {
        lines.push([x].concat(series.map(function (s) { var v = s.data[i]; return Array.isArray(v) ? v[1] : v; })).map(csvCell).join(','));
      });
    } else {
      // a chart whose points carry labels (the \u03c3 caps of the frontier) exports them too
      var labelled = series.some(function (s) { return s.pointLabels; });
      var head = ['series', xn, (spec.y && (spec.y.csv || spec.y.name)) || 'y', 'y_high'];
      lines.push((labelled ? head.concat([spec.pointCsv || 'point']) : head).map(csvCell).join(','));
      series.forEach(function (s) {
        seriesData(s, spec).forEach(function (p, j) {
          var row = Array.isArray(p) ? p : [null, p];
          var out = [s.name, row[0], row[1], row.length > 2 ? row[2] : ''];
          if (labelled) out.push(s.pointLabels ? s.pointLabels[j] : '');
          lines.push(out.map(csvCell).join(','));
        });
      });
    }
    return lines.join('\n') + '\n';
  }
  Desk.specCsv = specCsv;
  // a quote smile exports its quotes, one row each, every expiry and every model
  function quotesCsv(spec) {
    var Q = spec.quotes, ms = Q.models || [];
    var fx = function (v, d) { return v === null || v === undefined ? '' : Number(v).toFixed(d); };
    var head = ['expiry', 'T_days', 'strike', 'right', 'ln_K_F', 'forward_delta', 'bid_iv_pct', 'mid_iv_pct', 'ask_iv_pct', 'half_spread_vp'];
    ms.forEach(function (m) { head.push(m.label + ' IV (%)', m.label + ' z', m.label + ' fitted', m.label + ' inside'); });
    head.push('arb_exec', 'repaired', 'repaired_outside_spread');
    var lines = [head.map(csvCell).join(',')];
    (spec.quotes.order || Object.keys(spec.quotes.groups)).forEach(function (g) {
      rowsOf(spec, g).forEach(function (q) {
        var r = [g, fx(q.T, 4), q.strike, q.right, fx(q.k, 6), fx(q.delta, 6), fx(q.bid, 6), fx(q.mid, 5), fx(q.ask, 6), fx(q.hs, 6)];
        ms.forEach(function (m) {
          var mm = q.models[m.key] || {};
          r.push(fx(mm.iv, 4), fx(mm.z, 3), mm.fit ? 1 : 0, mm.inside === null || mm.inside === undefined ? '' : (mm.inside ? 1 : 0));
        });
        r.push(q.exec ? 1 : 0, q.repaired ? 1 : 0, q.outside ? 1 : 0);
        lines.push(r.map(csvCell).join(','));
      });
    });
    return lines.join('\n') + '\n';
  }
  function download(name, text) {
    var blob = new Blob([text], { type: 'text/csv;charset=utf-8' });
    var a = doc.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = name;
    doc.body.appendChild(a); a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 0);
  }

  // -- chart lifecycle ----
  // A compact spec (desk_charts.py): x values as {start, steps} day gaps, the x
  // values of another chart (xsFrom), a series derived from others (k\u00b7a \u2212 b).
  function expandDates(c) {
    var p = c.start.split('-'), t = Date.UTC(+p[0], +p[1] - 1, +p[2]), out = [c.start];
    c.steps.forEach(function (d) { t += d * 86400000; out.push(new Date(t).toISOString().slice(0, 10)); });
    return out;
  }
  function parseSpec(id) {
    var s = doc.getElementById(id);
    return s ? resolveSpec(JSON.parse(s.textContent)) : null;
  }
  function resolveSpec(spec) {
    if (spec.xsDates) { spec.xs = expandDates(spec.xsDates); delete spec.xsDates; }
    var ref = spec.xsFrom ? parseSpec(spec.xsFrom + '-spec') : null;
    if (ref) spec.xs = ref.xs;
    (spec.series || []).forEach(function (s) {
      var d = s.derive;
      if (!d) return;
      var src = d.from ? (ref && ref.id === d.from ? ref : parseSpec(d.from + '-spec')) : spec;
      var by = {};
      ((src && src.series) || []).forEach(function (x) { by[x.name] = x.data; });
      var a = by[d.a] || [], b = d.b ? by[d.b] : null, k = d.k === undefined ? 1 : d.k;
      var f = Math.pow(10, d.digits === undefined ? 2 : d.digits);
      s.data = a.map(function (v, i) {
        var w = b ? b[i] : 0;
        return v === null || v === undefined || w === null || w === undefined ? null : Math.round((k * v - w) * f) / f;
      });
    });
    // a quote smile: its expiries are decoded when drawn or exported (rowsOf)
    if (spec.quotes && spec.quotes.groups) { spec._rows = {}; spec._xmode = 'k'; }
    return spec;
  }
  Desk.resolveSpec = resolveSpec;
  function specOf(el) { return parseSpec(el.getAttribute('data-spec')); }
  // A legend button stands for every series with its key (s.legendKey, else the
  // name): on the smile slices one "Heston" button covers each expiry's curve.
  function legendKey(s) { return s.legendKey || s.name; }
  function legendOff(id) {
    var off = {};
    // a group's legend button names every member (data-legend-for="a b c")
    $$('[data-legend-for~="' + id + '"]').forEach(function (b) { if (b.getAttribute('aria-pressed') === 'false') off[b.getAttribute('data-series')] = true; });
    return off;
  }
  var fontsReady = !doc.fonts || doc.fonts.status === 'loaded';
  // shown = in the tab's group and not toggled off; every series is named (ECharts
  // merges the map: another group's must not keep an earlier 'shown')
  function selection(spec, id, f, off) {
    var sel = {}, inGroup = {};
    off = off || legendOff(id);
    groupIdx(spec, f).forEach(function (i) { inGroup[i] = true; });
    (spec.series || []).forEach(function (s, i) {
      sel[s.name] = !!inGroup[i] && !off[legendKey(s)];
      if (s.strip && s.quote === 'z') sel[s.name + CLIP_NAME] = sel[s.name];      // its clip triangles
    });
    return sel;
  }
  // A full draw: the spec is parsed once and kept (Desk.charts[id].spec); a tab
  // switch or a legend toggle then merges what changes, by series id.
  function render(el, print) {
    if (!window.echarts) return null;
    var prev = Desk.charts[el.id];
    var spec = prev && prev.spec ? prev.spec : specOf(el);
    if (!spec) return null;
    var mode = Desk.xmode[el.id] === 'delta' ? 'delta' : 'k';
    var f = Desk.filters[el.id], T = theme(print);
    if (spec.quotes) fillQuotes(spec, mode, f);
    var inst = window.echarts.getInstanceByDom(el) || window.echarts.init(el, null, { renderer: 'canvas' });
    var built = {};
    var off = legendOff(el.id);
    var opt = buildOption(spec, T, f, el.clientWidth, el.clientHeight, built, off);
    if (!spec.quotes) opt.legend.selected = selection(spec, el.id, f, off);
    delete opt._idx; delete opt._pos;
    // a book: fetched when the page is idle after the first draw ('finished' fires
    // inside setOption, so it is listened for first)
    var book = spec.quotes && spec.quotes.src;
    if (book && !prev) inst.on('finished', function pre() { inst.off('finished', pre); idle(function () { loadBook(spec).catch(noop); }); });
    inst.setOption(opt, true);
    // a chart group (data-chart-group, else spec.link): pointer and tooltip linked
    // by echarts.connect, the zoom kept equal by bindGroupZoom
    var grp = el.getAttribute('data-chart-group') || spec.link;
    if (grp && window.echarts.connect) { inst.group = grp; window.echarts.connect(grp); bindGroupZoom(inst, el.id, grp); }
    if (spec.exps) bindExpiryPick(inst, el.id);
    if (spec.zoomTo) bindZoomTo(inst, el.id);
    // drawn before the web fonts arrived: drawn again once they have (canvas text keeps its font)
    Desk.charts[el.id] = { el: el, chart: inst, spec: spec, T: T, early: !fontsReady,
      labels: built.labels, qi: prev ? prev.qi : null, cursorAt: Desk.expCursor };
    if (grp) joinGroupZoom(el.id, grp);
    if (book && !hasRows(spec, f)) needBook(Desk.charts[el.id]);    // an expiry not inline: at once
    return inst;
  }
  function noop() {}
  function idle(fn) { if (window.requestIdleCallback) window.requestIdleCallback(fn, { timeout: 2000 }); else setTimeout(fn, 50); }
  // an expiry not yet in hand: drawn empty under a loading mask until the book is
  function needBook(c) {
    var ch = c.chart;
    ch.showLoading({ text: 'Loading quotes\u2026', color: c.T.colors.info, textColor: c.T.text2, maskColor: alpha(c.T.surface, 0.6) });
    loadBook(c.spec).then(function () { bookFailed(c.el.id, null); ch.hideLoading(); queueRedraw(c.el.id); },
      function (err) { ch.hideLoading(); bookFailed(c.el.id, err); });
  }
  // A tab switch or a smile's model toggle: one merge of the series now drawn (by id),
  // the legend state or a smile's x range. The spec is not parsed again.
  function applyFilter(c) {
    var spec = c.spec, el = c.el, f = Desk.filters[el.id], off = legendOff(el.id);
    if (spec.quotes) {
      if (spec.quotes.src && !hasRows(spec, f)) needBook(c);     // drawn empty meanwhile
      fillQuotes(spec, spec._xmode, f);
    }
    // the drawn size, not the element's: reading that would lay the page out again
    // in the middle of a tab switch (the tables of the new group just shown)
    var opt = buildOption(spec, c.T, f, c.chart.getWidth(), c.chart.getHeight(), {}, off);
    var up = { series: opt.series };
    if (spec.quotes) up.xAxis = opt.xAxis; else up.legend = { selected: selection(spec, el.id, f, off) };
    c.chart.setOption(up, { replaceMerge: ['series'] });
  }
  // The tab (or toggle) shows its new state in the next frame, the chart is drawn
  // in the task after it: a run of switches (an arrow key held) draws the last only.
  var queued = {};
  function afterPaint(key, fn) {
    if (queued[key]) return;
    queued[key] = true;
    var run = function () { queued[key] = false; fn(); };
    if (window.requestAnimationFrame) window.requestAnimationFrame(function () { setTimeout(run, 0); });
    else setTimeout(run, 0);
  }
  function queueRedraw(id) { afterPaint(id, function () { if (Desk.charts[id]) applyFilter(Desk.charts[id]); }); }
  function renderAll(print) { $$('.chart[data-spec]').forEach(function (el) { render(el, print); }); }
  Desk.render = render;

  var pending = null;                 // charts the IntersectionObserver has not drawn yet
  function initCharts() {
    if (!$$('.chart').length) return;           // a page without charts loads no chart library
    if (!window.echarts) {
      root.classList.add('charts-failed');
      $$('.chart').forEach(function (w) {
        if (w.hidden || w.closest('[hidden]')) return;     // the unloaded 3D view: nothing is shown there
        var p = doc.createElement('p');
        p.className = 'chart-fallback';
        p.textContent = 'Chart unavailable: the charting library did not load. The values are listed below.';
        w.parentNode.insertBefore(p, w.nextSibling);
      });
      return;
    }
    var els = $$('.chart[data-spec]');
    if ('IntersectionObserver' in window) {
      pending = new Set(els);
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (!e.isIntersecting) return;
          io.unobserve(e.target); pending.delete(e.target);
          if (Desk.charts[e.target.id]) return;            // a tab switch drew it already
          try { render(e.target); } catch (err) { console.error('chart', e.target.id, err); }
        });
      }, { rootMargin: '300px 0px' });
      els.forEach(function (el) { io.observe(el); });
    } else { els.forEach(function (el) { render(el); }); }
    if (!fontsReady && doc.fonts && doc.fonts.ready) {
      doc.fonts.ready.then(function () {
        fontsReady = true;
        Object.keys(Desk.charts).forEach(function (k) { var c = Desk.charts[k]; if (c.early) render(c.el); });
      });
    }
    // a chart that grows to its row's height (.grow, a paired row) is resized
    // when the row settles (fonts, a neighbour drawn, a tab), not only when the
    // window is
    if ('ResizeObserver' in window) {
      var ro = new ResizeObserver(function (entries) {
        entries.forEach(function (e) { var c = Desk.charts[e.target.id]; if (c) c.chart.resize(); });
      });
      $$('.chart.grow[data-spec]').forEach(function (el) { ro.observe(el); });
    }
    var t;
    window.addEventListener('resize', function () {
      clearTimeout(t);
      t = setTimeout(function () {
        Object.keys(Desk.charts).forEach(function (k) {
          var c = Desk.charts[k];
          // label layout follows the width, a quote smile's strip the height
          if (c.spec && (c.spec.labelShare || (c.spec.x && c.spec.x.fit) || c.spec.quotes)) render(c.el); else c.chart.resize();
        });
      }, 120);
    });
  }
  Desk.isPending = function (el) { return !!(pending && pending.has(el)); };

  // -- print: light theme, every methodology drawer and tab panel open ----
  function initPrint() {
    var opened = [], shown = [];
    window.addEventListener('beforeprint', function () {
      $$('details:not([open])').forEach(function (d) { d.open = true; opened.push(d); });
      $$('[role="tabpanel"][hidden]').forEach(function (p) { p.hidden = false; shown.push(p); });
      if (window.echarts) { renderAll(true); Object.keys(Desk.charts).forEach(function (k) { Desk.charts[k].chart.resize(); }); }
    });
    window.addEventListener('afterprint', function () {
      opened.forEach(function (d) { d.open = false; }); opened = [];
      shown.forEach(function (p) { p.hidden = true; }); shown = [];
      if (window.echarts) renderAll(false);
    });
  }

  // -- chart toolbars: legend toggles, zoom presets, CSV, table ----
  function initTools() {
    $$('[data-legend-for]').forEach(function (b) {
      b.addEventListener('click', function () {
        var on = b.getAttribute('aria-pressed') !== 'true';
        b.setAttribute('aria-pressed', on ? 'true' : 'false');
        b.getAttribute('data-legend-for').split(/\s+/).forEach(function (id) { toggleLegend(id); });
      });
    });
    // one chart's series after a legend button changed (a group's button: each member's)
    function toggleLegend(id) {
      var c = Desk.charts[id], f = Desk.filters[id];
      if (!c) return;              // not drawn yet: render() reads the buttons when it is
      // one merge for every series of the key; a quote smile is drawn again (it leaves out what is off)
      if (c.spec.quotes) queueRedraw(id);
      else c.chart.setOption({ legend: { selected: selection(c.spec, id, f) } });
      if (c.spec.quotes && c.qi !== null && c.qi !== undefined) {       // the readout's model rows follow
        var q = rowsOf(c.spec, f)[c.qi];
        if (q) showReadout(id, quoteLines(c.spec, q, legendOff(id)));
      }
    }
    $$('[data-zoom-for]').forEach(function (b) {
      b.addEventListener('click', function () {
        var id = b.getAttribute('data-zoom-for'), c = Desk.charts[id];
        b.setAttribute('aria-pressed', 'true');
        if (!c) return;
        var years = parseFloat(b.getAttribute('data-years'));
        var xs = c.spec.xs || [];
        if (!xs.length) return;
        var from = xs[0];
        if (years) {
          var last = new Date(xs[xs.length - 1]), f = new Date(last);
          f.setFullYear(last.getFullYear() - years);
          from = f.toISOString().slice(0, 10);
        }
        // the same window of dates on every member of the chart's group
        zoomTo(id, from, xs[xs.length - 1], b);
      });
    });
    // a chart's CSV, or a group's (every member's rows under one header); a quote
    // smile's rows carry their expiry, every expiry's (its book first)
    $$('[data-csv-for],[data-csv-group]').forEach(function (b) {
      b.addEventListener('click', function () {
        var one = b.getAttribute('data-csv-for'), ids = (one || b.getAttribute('data-csv-group')).split(/\s+/);
        var specs = ids.map(function (id) { var el = doc.getElementById(id), c = Desk.charts[id]; return c ? c.spec : el && specOf(el); }).filter(Boolean);
        Promise.all(specs.map(function (sp) { return sp.quotes ? loadBook(sp) : sp; })).then(function () {
          var out = [];
          specs.forEach(function (spec) {
            var lines = specCsv(spec).replace(/\n$/, '').split('\n');
            out = out.length ? out.concat(lines.slice(1)) : lines;
          });
          if (out.length) download(b.getAttribute('data-filename') || (one ? one + '.csv' : 'charts.csv'), out.join('\n') + '\n');
        }, function (err) {
          flash(b, 'Unavailable');
          specs.forEach(function (sp) { if (sp.quotes) bookFailed(sp.id, err); });
        });
      });
    });
    $$('[data-zoom-year-for]').forEach(function (sel) {
      sel.addEventListener('change', function () {
        var v = sel.value.split('|');
        if (v.length === 2) zoomTo(sel.getAttribute('data-zoom-year-for'), v[0], v[1], sel);
      });
    });
    $$('[data-table-for]').forEach(function (b) {
      b.addEventListener('click', function () {
        var t = doc.getElementById(b.getAttribute('data-table-for'));
        if (!t) return;
        var show = !t.classList.contains('show');
        t.classList.toggle('show', show);
        b.setAttribute('aria-expanded', show ? 'true' : 'false');
      });
    });
    $$('[data-copy-link]').forEach(function (b) {
      b.addEventListener('click', function () {
        // the panel plus the page's tab state (#scorecard&exp=\u2026&model=\u2026)
        history.replaceState(null, '', state.link(b.getAttribute('data-copy-link')));
        var url = location.href;
        if (navigator.clipboard) navigator.clipboard.writeText(url).then(function () { flash(b, 'Copied'); }, function () { flash(b, 'Link set'); });
        else flash(b, 'Link set');
      });
    });
  }
  function flash(b, text) { var old = b.textContent; b.textContent = text; setTimeout(function () { b.textContent = old; }, 1200); }

  // -- URL-hash state: an optional panel anchor, then key=value pairs ----
  //    #scorecard            the panel (the browser scrolls to it)
  //    #exp=2026-12-18       tab state only
  //    #scorecard&exp=\u2026      both: the anchor is kept first, desk.js scrolls to it
  function tokens() { return location.hash.replace(/^#/, '').split('&').filter(Boolean); }
  var state = {
    anchor: function () {
      var a = tokens().filter(function (t) { return t.indexOf('=') < 0; })[0];
      return a ? decodeURIComponent(a) : null;
    },
    get: function (k) {
      var out = null;
      tokens().forEach(function (kv) {
        if (kv.indexOf('=') < 0) return;
        var p = kv.split('=');
        if (decodeURIComponent(p[0]) === k) out = decodeURIComponent(p[1] || '');
      });
      return out;
    },
    pairs: function () { return tokens().filter(function (t) { return t.indexOf('=') >= 0; }); },
    set: function (k, v) {
      var parts = state.pairs().filter(function (kv) { return decodeURIComponent(kv.split('=')[0]) !== k; });
      if (v !== null && v !== undefined) parts.push(encodeURIComponent(k) + '=' + encodeURIComponent(v));
      var a = state.anchor();
      if (a) parts.unshift(encodeURIComponent(a));
      history.replaceState(null, '', parts.length ? '#' + parts.join('&') : location.pathname + location.search);
    },
    // a link to one panel that keeps the page's tab state
    link: function (anchor) { return '#' + [encodeURIComponent(anchor)].concat(state.pairs()).join('&'); },
  };
  Desk.state = state;

  // a '#panel&k=v' fragment names no element: scroll to the anchor once the tabs are
  // set, and again at 'load' unless the reader has scrolled since
  function scrollToAnchor() {
    var a = state.anchor();
    if (!a || !state.pairs().length) return;
    var el = doc.getElementById(a);
    if (!el) return;
    el.scrollIntoView();
    var y = window.scrollY;
    window.addEventListener('load', function () { if (Math.abs(window.scrollY - y) < 2) el.scrollIntoView(); }, { once: true });
  }

  // -- the smile's one-line expiry strip: a selected tab out of view is centred in it
  // (scrollIntoView scrolls the page); the status line is the expiry's row of the
  // audited 'Quotes per expiry' table, as the server writes it
  function keepInView(list, tab) {
    requestAnimationFrame(function () {
      var l = list.getBoundingClientRect(), t = tab.getBoundingClientRect();
      if (t.left < l.left || t.right > l.right) list.scrollLeft += (t.left + t.right - l.left - l.right) / 2;
    });
  }
  function expiryStatus(out, tab) {
    var e = tab.getAttribute('data-value'), p = e.split('-'), s = tab.getAttribute('data-settle');
    var r = $$('#' + out.getAttribute('data-from') + ' > table > tbody > tr').filter(function (tr) { return tr.cells[0].textContent === e; })[0];
    if (!r) return;
    var c = r.cells, t = [e, 'SunMonTueWedThuFriSat'.substr(3 * new Date(Date.UTC(+p[0], p[1] - 1, +p[2])).getUTCDay(), 3)]
      .concat(s ? [s] : [], c[1].textContent + ' days', 'F ' + c[2].textContent, c[3].textContent + ' quotes',
        'median half-spread ' + c[4].textContent + ' vol pts').join(' \u00b7 ');
    if (out.textContent !== t) out.textContent = t;
  }

  // -- tabs (heatmap metric switch, smile expiry) ----
  function initTabs() {
    $$('[role="tablist"]').forEach(function (list) {
      var tabs = $$('[role="tab"]', list), key = list.getAttribute('data-state-key');
      // every tab drives one panel (the smile slices), or each its own (the session
      // comparison, which also filters charts: data-chart-filter lists their ids)
      var shared = tabs.every(function (t) { return t.getAttribute('aria-controls') === tabs[0].getAttribute('aria-controls'); });
      // the status line: the server wrote the one of the tab it selected
      var strip = list.classList.contains('exp-strip'), status = doc.getElementById(list.getAttribute('data-status') || '');
      var served = tabs.filter(function (t) { return t.getAttribute('aria-selected') === 'true'; })[0];
      // init: the first selection writes no state (a load keeps its URL and anchor) and
      // leaves undrawn charts to the IntersectionObserver. A switch writes the old and
      // new tab only; the status line after the click's frame
      function select(tab, focus, init) {
        tabs.forEach(function (t) {
          var on = t === tab, p = !shared && doc.getElementById(t.getAttribute('aria-controls'));
          if (t.getAttribute('aria-selected') !== String(on)) t.setAttribute('aria-selected', String(on));
          if (t.tabIndex !== (on ? 0 : -1)) t.tabIndex = on ? 0 : -1;
          if (p && p.hidden === on) p.hidden = !on;
        });
        if (focus) tab.focus();
        if (strip) keepInView(list, tab);
        if (status && init && tab !== served) expiryStatus(status, tab);
        else if (status && !init) afterPaint('status', function () { expiryStatus(status, $('[aria-selected="true"]', list)); });
        if (key && !init) state.set(key, tab.getAttribute('data-value'));
        if (key === 'exp' && Desk.onExpiry) Desk.onExpiry(tab.getAttribute('data-value'), !!init);
        var target = list.getAttribute('data-chart-filter');
        if (target) target.split(/\s+/).forEach(function (id) { filterChart(id, tab.getAttribute('data-value')); });
        // a chart drawn in a hidden panel is resized if its size changed (the tabs of
        // one shared panel, the smile's, never hide it)
        $$('.chart[data-spec]', doc.getElementById(tab.getAttribute('aria-controls')) || doc.body).forEach(function (el) {
          var c = Desk.charts[el.id];
          if (c) { if (!shared && (c.chart.getWidth() !== el.clientWidth || c.chart.getHeight() !== el.clientHeight)) c.chart.resize(); }
          else if (!Desk.isPending(el)) render(el);
        });
      }
      tabs.forEach(function (t, i) {
        t.addEventListener('click', function () { select(t, false); });
        t.addEventListener('keydown', function (e) {
          var j = e.key === 'ArrowRight' ? i + 1 : e.key === 'ArrowLeft' ? i - 1 : e.key === 'Home' ? 0 : e.key === 'End' ? tabs.length - 1 : null;
          if (j === null) return;
          e.preventDefault(); select(tabs[(j + tabs.length) % tabs.length], true);
        });
      });
      var want = key && state.get(key);
      var initial = tabs.filter(function (t) { return t.getAttribute('data-value') === want; })[0]
        || tabs.filter(function (t) { return t.getAttribute('aria-selected') === 'true'; })[0] || tabs[0];
      if (initial) select(initial, false, true);
      list.hidden = false;
    });
  }
  // A tab list can filter one chart's series by a "group" key (the smile slices).
  function filterChart(id, value) {
    Desk.filters[id] = value;
    // its legend buttons of one group (an earlier session's line) and its per-group tables
    // follow; only those that change are written
    $$('[data-legend-for="' + id + '"][data-group], #' + id + '-data [data-group]').forEach(function (b) {
      var h = String(b.getAttribute('data-group')) !== String(value);
      if (b.hidden !== h) b.hidden = h;
    });
    var c = Desk.charts[id];
    if (c) { c.qi = null; queueRedraw(id); afterPaint(id + '-r', function () { resetReadout(id); }); }    // not yet drawn: render() applies the filter
  }

  // -- quote smiles: the keyboard readout (focusable: arrow keys step through the
  // shown expiry's quotes, it says what the tooltip says and shows it) and the x axis ----
  var READOUT_KEYS = { ArrowRight: 1, ArrowLeft: -1, PageDown: 10, PageUp: -10 };
  function showReadout(id, lines) {
    var r = doc.getElementById(id + '-readout');
    if (r) r.textContent = quoteText(lines);
  }
  function resetReadout(id) {
    var r = doc.getElementById(id + '-readout'), c = Desk.charts[id], t = r && r.getAttribute('data-idle');
    if (c) c.qi = null;
    if (t && r.textContent !== t) r.textContent = t;
  }
  function stepQuote(id, key) {
    var c = Desk.charts[id];
    if (!c || !c.spec.quotes) return false;
    var spec = c.spec, f = Desk.filters[id], rows = rowsOf(spec, f);
    if (!rows.length) return false;
    var i;
    if (key === 'Home') i = 0;
    else if (key === 'End') i = rows.length - 1;
    else if (READOUT_KEYS[key] === undefined) return false;
    else if (c.qi === null || c.qi === undefined || c.qi >= rows.length) {
      // the first step starts at the quote nearest the forward
      i = rows.indexOf(nearestQuote(spec, f, spec._xmode === 'delta' ? 0.5 : 0));
    } else i = Math.max(0, Math.min(rows.length - 1, c.qi + READOUT_KEYS[key]));
    var q = rows[i];
    c.qi = i;
    showReadout(id, quoteLines(spec, q, legendOff(id)));
    var px = c.chart.convertToPixel({ xAxisIndex: 0, yAxisIndex: 0 }, [spec._xmode === 'delta' ? q.xd : q.k, q.mid]);
    if (px && isFinite(px[0]) && isFinite(px[1])) c.chart.dispatchAction({ type: 'showTip', x: px[0], y: px[1] });
    c.qi = i;                   // the tooltip's own update set it to the same quote
    return true;
  }
  Desk.stepQuote = stepQuote;
  function initQuotes() {
    $$('.quote-readout[data-readout-for]').forEach(function (r) {
      var id = r.getAttribute('data-readout-for');
      r.setAttribute('data-idle', r.textContent);
      r.addEventListener('keydown', function (e) { if (stepQuote(id, e.key)) e.preventDefault(); });
      r.addEventListener('blur', function () { var c = Desk.charts[id]; if (c) c.chart.dispatchAction({ type: 'hideTip' }); });
    });
    $$('[data-xmode-for]').forEach(function (b) {
      b.addEventListener('click', function () { setXmode(b.getAttribute('data-xmode-for'), b.getAttribute('data-xmode')); });
    });
    $$('[data-xmode-for]').forEach(function (b) {      // a link that names the axis (#\u2026&x=delta)
      if (b.getAttribute('data-xmode') === 'delta' && state.get('x') === 'delta') setXmode(b.getAttribute('data-xmode-for'), 'delta', true);
    });
  }
  // -- scatter strips (F4): the readout steps through the shown points by x (the
  // entry date), reads series, date and value and shows the tooltip there ----
  function stripPoints(id) {
    var c = Desk.charts[id];
    if (!c) return [];
    var off = legendOff(id), pts = [];
    (c.chart.getOption().series || []).forEach(function (s, si) {
      if (s.type !== 'scatter' || off[s.name]) return;
      (s.data || []).forEach(function (d, di) {
        var v = Array.isArray(d) ? d : (d && d.value);
        if (!v || v[1] === null || v[1] === undefined) return;
        var x = typeof v[0] === 'number' ? new Date(v[0]).toISOString().slice(0, 10) : String(v[0]).slice(0, 10);
        pts.push({ name: s.name, si: si, di: di, x: x, y: Number(v[1]) });
      });
    });
    pts.sort(function (a, b) { return a.x < b.x ? -1 : (a.x > b.x ? 1 : a.y - b.y); });
    return pts;
  }
  function stepPoint(id, key, r) {
    var c = Desk.charts[id], pts = stripPoints(id);
    if (!c || !pts.length) return false;
    var i, cur = c.pi;
    if (key === 'Home') i = 0;
    else if (key === 'End') i = pts.length - 1;
    else if (READOUT_KEYS[key] === undefined) return false;
    else i = (cur === null || cur === undefined) ? 0 : Math.max(0, Math.min(pts.length - 1, cur + READOUT_KEYS[key]));
    var p = pts[i], y = c.spec.y || {};
    c.pi = i;
    r.textContent = p.name + ', entered ' + p.x + ': ' + (y.name || 'value') + ' ' +
      num(p.y, { digits: y.tipDigits === undefined ? 1 : y.tipDigits, signed: true }) +
      ' \u00b7 ' + (i + 1) + ' of ' + pts.length + ' by entry date';
    c.chart.dispatchAction({ type: 'showTip', seriesIndex: p.si, dataIndex: p.di });
    return true;
  }
  function initPointReadouts() {
    $$('[data-points-for]').forEach(function (r) {
      var id = r.getAttribute('data-points-for');
      r.addEventListener('keydown', function (e) { if (stepPoint(id, e.key, r)) e.preventDefault(); });
      r.addEventListener('blur', function () { var c = Desk.charts[id]; if (c) c.chart.dispatchAction({ type: 'hideTip' }); });
    });
    // a sleeve toggled in the legend renumbers the points: start again from the first
    $$('[data-legend-for]').forEach(function (b) {
      b.addEventListener('click', function () {
        (b.getAttribute('data-legend-for') || '').split(' ').forEach(function (id) { var c = Desk.charts[id]; if (c) c.pi = null; });
      });
    });
  }
  function setXmode(id, mode, init) {
    Desk.xmode[id] = mode;
    $$('[data-xmode-for="' + id + '"]').forEach(function (o) { o.setAttribute('aria-pressed', o.getAttribute('data-xmode') === mode ? 'true' : 'false'); });
    if (!init) state.set('x', mode === 'delta' ? 'delta' : null);
    resetReadout(id);
    var el = doc.getElementById(id);
    if (el && Desk.charts[id]) render(el);
  }

  // -- heatmap cells: arrow-key navigation and a readout line. A Surface grid's
  // (data-src) numbers are in the session's data file with its template ({i:d}: number
  // i to d decimals, + signed); until it is in, a cell reads its row, column and text ----
  function initHeatmaps() {
    $$('table.heat').forEach(function (tbl) {
      var out = doc.getElementById(tbl.getAttribute('data-readout'));
      var src = tbl.getAttribute('data-src'), sel = src ? 'tbody td' : 'td[data-detail]', heat, last;
      var cells = $$(sel, tbl);
      if (!cells.length) return;
      cells.forEach(function (c, i) { c.tabIndex = i === 0 ? 0 : -1; });
      var withHeads = tbl.getAttribute('data-prefix') === 'headers';
      function heads(c) {
        var th = c.parentNode.querySelector('th'), col = tbl.tHead && tbl.tHead.rows[0].cells[c.cellIndex];
        return (th ? th.textContent.trim() + ' \u00b7 ' : '') + (col ? col.textContent.trim() + ' \u00b7 ' : '');
      }
      function show(c) {
        // the readout follows the units toggle of the region the table is in
        var u = c.closest('.units'), alt = u && u.getAttribute('data-show') === 'alt' && c.getAttribute('data-detail-alt'), v;
        last = c;
        if (src && heat === undefined) {
          heat = null;
          Desk.data(src, tbl.getAttribute('data-integrity'), tbl.getAttribute('data-session')).then(function (j) { heat = j.heat[tbl.id]; show(last); }, function () {});
        }
        v = heat && heat.cells[c.parentNode.sectionRowIndex][c.cellIndex - 1];
        if (out) out.textContent = (withHeads ? heads(c) : '') + (!src ? alt || c.getAttribute('data-detail') : c.className === 'empty' ? tbl.getAttribute('data-empty')
          : v ? heat.detail.replace(/\{(\d)(:(\+?)(\d))?\}/g, function (m, i, x, sg, dg) { return x ? num(v[i], { digits: +dg, signed: !!sg }) : v[i]; }) : c.textContent);
      }
      tbl.addEventListener('mouseover', function (e) { var c = e.target.closest(sel); if (c) show(c); });
      tbl.addEventListener('focusin', function (e) { var c = e.target.closest(sel); if (c) show(c); });
      tbl.addEventListener('keydown', function (e) {
        var c = e.target.closest('td'); if (!c) return;
        var row = c.parentNode, ri = Array.prototype.indexOf.call(row.parentNode.rows, row), ci = c.cellIndex, next = null;
        if (e.key === 'ArrowRight') next = row.cells[ci + 1];
        else if (e.key === 'ArrowLeft' && ci > 1) next = row.cells[ci - 1];
        else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
          var r2 = row.parentNode.rows[ri + (e.key === 'ArrowDown' ? 1 : -1)];
          next = r2 && r2.cells[ci];
        }
        if (next && next.tagName === 'TD') {
          e.preventDefault(); c.tabIndex = -1; next.tabIndex = 0; next.focus();
        }
      });
    });
  }

  // -- chart groups share one zoom (and their x values): a zoom on one is applied to
  // each other, a task each after the zoomed chart's frame; a member
  // already at the window (echarts.connect may carry a drag) is left alone ----
  function groupMembers(grp) {
    return $$('.chart[data-chart-group="' + grp + '"]').map(function (el) { return el.id; });
  }
  function zoomWindow(c) {
    var dz = c.chart.getModel().getComponent('dataZoom', 0);
    return dz ? dz.getValueRange() : null;
  }
  var zoomEcho = false, zoomByControl = false;
  function clearZoomControls(id) {
    $$('[data-zoom-for="' + id + '"]').forEach(function (o) { o.setAttribute('aria-pressed', 'false'); });
    $$('[data-zoom-year-for="' + id + '"]').forEach(function (o) { o.value = ''; });
  }
  // a window applied to one chart only (escapeConnect: not carried to the group)
  function setWindow(c, w) {
    var cur = zoomWindow(c);
    if (!cur || (Math.abs(cur[0] - w[0]) < 0.5 && Math.abs(cur[1] - w[1]) < 0.5)) return;
    zoomEcho = true;
    try { c.chart.dispatchAction({ type: 'dataZoom', batch: [{ startValue: w[0], endValue: w[1] }], escapeConnect: true }); }
    finally { zoomEcho = false; }
  }
  function syncGroup(grp, lead) {
    var c = Desk.charts[lead], w = c && zoomWindow(c);
    if (!w) return;
    var rest = groupMembers(grp).filter(function (id) { return id !== lead; });
    (function next(i) {
      if (i >= rest.length) return;
      var m = Desk.charts[rest[i]];
      if (m) setWindow(m, w);
      setTimeout(function () { next(i + 1); }, 0);
    })(0);
  }
  function bindGroupZoom(inst, id, grp) {
    if (inst.__groupZoom) return;
    inst.__groupZoom = true;
    inst.on('datazoom', function () {
      if (zoomEcho) return;
      // a drag or the wheel: no preset or year names the window any more
      if (!zoomByControl) groupMembers(grp).forEach(function (m) { clearZoomControls(m); });
      afterPaint('zoom:' + grp, function () { syncGroup(grp, id); });
    });
  }
  // a member drawn after the group was zoomed takes the group's window
  function joinGroupZoom(id, grp) {
    var c = Desk.charts[id];
    if (!c || !zoomWindow(c)) return;
    groupMembers(grp).some(function (o) {
      var m = o !== id && Desk.charts[o], w = m && zoomWindow(m);
      if (w) { setWindow(c, w); return true; }
      return false;
    });
  }
  // the zoom of a chart and of its group; its presets and year menu show none chosen
  function zoomTo(id, from, to, keep) {
    var el = doc.getElementById(id);
    if (!el) return;
    // the chart itself now; its group follows (bindGroupZoom)
    zoomByControl = true;
    try { if (Desk.charts[id]) Desk.charts[id].chart.dispatchAction({ type: 'dataZoom', batch: [{ startValue: from, endValue: to }], escapeConnect: true }); }
    finally { zoomByControl = false; }
    $$('[data-zoom-for="' + id + '"]').forEach(function (o) { if (o !== keep) o.setAttribute('aria-pressed', 'false'); });
    $$('[data-zoom-year-for="' + id + '"]').forEach(function (o) { if (o !== keep) o.value = o.querySelector('option[value="' + from + '|' + to + '"]') ? from + '|' + to : ''; });
  }
  // a click on a bar of a chart whose categories are windows of another (the
  // annual returns: each year of the equity group)
  function bindZoomTo(inst, id) {
    if (inst.__zoomTo) return;
    inst.__zoomTo = true;
    var zr = inst.getZr();
    var at = function (e) {
      var c = Desk.charts[id], p = [e.offsetX, e.offsetY];
      if (!c || !c.spec.zoomTo || !inst.containPixel({ gridIndex: 0 }, p)) return null;
      var i = Math.round(inst.convertFromPixel({ xAxisIndex: 0 }, p[0]));
      return c.spec.zoomTo.windows[i] || null;
    };
    zr.on('click', function (e) { var w = at(e); if (w) zoomTo(Desk.charts[id].spec.zoomTo.chart, w[0], w[1]); });
    zr.on('mousemove', function (e) { zr.setCursorStyle(at(e) ? 'pointer' : 'default'); });
  }

  // -- Surface's shared expiry cursor: a heatmap or delta-grid row head (a button added
  // here), a #chart-by-expiry category or '[' / ']' picks the smile tab; its rows go
  // aria-current, the by-expiry chart draws a line; the smile scrolls in if offscreen ----
  function expiryTabs() {
    var list = $('[role="tablist"][data-state-key="exp"]');
    return list ? $$('[role="tab"]', list) : [];
  }
  // A switch marks its rows after the click's frame; their tables' tab stops, each
  // capped grid's scroll to its row and the by-expiry line (on screen only) wait for idle
  var marked = null, cursorIO = 'IntersectionObserver' in window && new IntersectionObserver(function (es) {
    es.forEach(function (e) { if (e.isIntersecting) moveCursor(Desk.charts[e.target.id]); });
  });
  function expRows(e) { return e ? $$('tr[data-exp="' + e + '"]') : []; }
  function markRows() {
    var e = Desk.expCursor, rows = expRows(e);
    if (e !== marked) {
      expRows(marked).forEach(function (tr) { tr.removeAttribute('aria-current'); });
      rows.forEach(function (tr) { tr.setAttribute('aria-current', 'true'); });
      marked = e;
    }
    return rows;
  }
  function syncExpiry(exp, init) {
    Desk.expCursor = exp;
    if (!init) return afterPaint('exp', function () { var rows = markRows(); idle(function () { settle(rows); }); });
    markRows();
    // one tab stop per table: the selected row's head, else the first
    $$('table').forEach(function (t) {
      var b = t.querySelector('tr[aria-current="true"] .row-pick') || t.querySelector('.row-pick');
      if (b) b.tabIndex = 0;
    });
    idle(function () { settle(expRows(exp)); });
  }
  function settle(rows) {
    var moves = rows.map(function (tr) {
      var w = tr.closest('.heat-wrap.capped'), h = tr.closest('table').tHead;
      if (!w || w.scrollHeight <= w.clientHeight) return 0;
      var a = w.getBoundingClientRect(), r = tr.getBoundingClientRect(), top = a.top + (h ? h.offsetHeight : 0);
      return r.top < top || r.bottom > a.bottom ? r.top - top - (a.bottom - top - r.height) / 2 : 0;
    });
    rows.forEach(function (tr, i) {
      var b = tr.querySelector('.row-pick'), o = tr.closest('table').querySelector('.row-pick[tabindex="0"]');
      if (b && o !== b) { if (o) o.tabIndex = -1; b.tabIndex = 0; }
      if (moves[i]) tr.closest('.heat-wrap').scrollTop += moves[i];
    });
    for (var k in Desk.charts) {
      var c = Desk.charts[k], r = c.spec.exps && c.el.getBoundingClientRect();
      if (r && (!cursorIO || (r.bottom > 0 && r.top < innerHeight))) moveCursor(c);
    }
  }
  function moveCursor(c) {
    if (c && c.cursorAt !== Desk.expCursor) { c.cursorAt = Desk.expCursor; c.chart.setOption({ series: [cursorSeries(c.spec, c.T)] }); }
  }
  function pickExpiry(exp) {
    var tab = expiryTabs().filter(function (t) { return t.getAttribute('data-value') === exp; })[0];
    if (!tab) return false;
    tab.click();
    var sm = doc.getElementById(tab.parentNode.getAttribute('data-chart-filter'));
    if (sm) {
      var r = sm.getBoundingClientRect();
      if (r.bottom <= 0 || r.top >= (window.innerHeight || doc.documentElement.clientHeight)) sm.scrollIntoView({ block: 'nearest' });
    }
    return true;
  }
  function bindExpiryPick(inst, id) {
    if (inst.__expPick) return;
    inst.__expPick = true;
    if (cursorIO) cursorIO.observe(inst.getDom());
    var zr = inst.getZr();
    var at = function (e) {
      var c = Desk.charts[id], p = [e.offsetX, e.offsetY];
      if (!c || !c.spec.exps || !inst.containPixel({ gridIndex: 0 }, p)) return null;
      var i = Math.round(inst.convertFromPixel({ xAxisIndex: 0 }, p[0]));
      return i >= 0 && i < c.spec.exps.length ? c.spec.exps[i] : null;
    };
    zr.on('click', function (e) { var exp = at(e); if (exp) pickExpiry(exp); });
    zr.on('mousemove', function (e) { zr.setCursorStyle(at(e) ? 'pointer' : 'default'); });
  }
  function initExpiryCursor() {
    var tabs = expiryTabs();
    if (!tabs.length) return;
    $$('tr[data-exp]').forEach(function (tr) {
      var th = tr.querySelector('th[scope="row"]');
      if (!th || th.querySelector('.row-pick')) return;
      var b = doc.createElement('button');
      b.type = 'button'; b.className = 'row-pick'; b.tabIndex = -1;
      b.title = 'Select this expiry: the smile and the other tables follow';
      while (th.firstChild) b.appendChild(th.firstChild);
      th.appendChild(b);
      b.addEventListener('click', function () { pickExpiry(tr.getAttribute('data-exp')); });
      // up / down: the row heads of the same table
      b.addEventListener('keydown', function (e) {
        if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
        var heads = $$('.row-pick', tr.closest('table')), i = heads.indexOf(b) + (e.key === 'ArrowDown' ? 1 : -1);
        if (i < 0 || i >= heads.length) return;
        e.preventDefault(); b.tabIndex = -1; heads[i].tabIndex = 0; heads[i].focus();
      });
    });
    Desk.onExpiry = syncExpiry;
    Desk.stepExpiry = function (d) {
      var ts = expiryTabs(), i = -1;
      ts.forEach(function (t, j) { if (t.getAttribute('aria-selected') === 'true') i = j; });
      var j = Math.max(0, Math.min(ts.length - 1, i + d));
      if (!ts.length || j === i) return ts.length > 0;
      return pickExpiry(ts[j].getAttribute('data-value'));
    };
    var sel = tabs.filter(function (t) { return t.getAttribute('aria-selected') === 'true'; })[0] || tabs[0];
    syncExpiry(sel.getAttribute('data-value'), true);
  }

  // -- row filters: aria-pressed buttons that show a subset of a table's rows ----
  //    (without JavaScript the buttons are not shown and every row is)
  function initFilters() {
    var keep = {
      all: function () { return true; },
      quantified: function (tr) { return tr.getAttribute('data-status') === 'quantified'; },
      notest: function (tr) { return tr.getAttribute('data-tests') === '0'; },
    };
    $$('[data-filter-for]').forEach(function (b) {
      b.addEventListener('click', function () {
        var id = b.getAttribute('data-filter-for'), name = b.getAttribute('data-filter') || 'all';
        // 'status:breach': the rows of one status (the Model Risk summary strip)
        var f = name.indexOf('status:') === 0
          ? function (tr) { return tr.getAttribute('data-status') === name.slice(7); }
          : (keep[name] || keep.all);
        $$('#' + id + ' tbody tr').forEach(function (tr) { tr.hidden = !f(tr); });
        $$('[data-filter-for="' + id + '"]').forEach(function (o) { o.setAttribute('aria-pressed', o === b ? 'true' : 'false'); });
      });
    });
  }

  // -- a link to an element inside a closed <details> (a collapsed postmortem)
  //    opens it before the browser scrolls to the element --
  function openTarget(ev) {
    var id = state.anchor() || decodeURIComponent(location.hash.replace(/^#/, ''));
    var el = id && doc.getElementById(id);
    if (!el) return;
    var opened = false;
    for (var p = el; p; p = p.parentElement) {           // the target itself may be a disclosure
      if (p.tagName === 'DETAILS' && !p.open) { p.open = true; opened = true; }
    }
    // the browser scrolled before the disclosure opened (or could not scroll to a closed one)
    if (opened || (ev && ev.type === 'hashchange')) el.scrollIntoView();
  }

  // -- units toggle: a region shown in one of two units (aria-pressed buttons) ----
  function initUnits() {
    $$('[data-units-for]').forEach(function (b) {
      b.addEventListener('click', function () {
        var id = b.getAttribute('data-units-for'), region = doc.getElementById(id);
        if (!region) return;
        region.setAttribute('data-show', b.getAttribute('data-show'));
        $$('[data-units-for="' + id + '"]').forEach(function (o) { o.setAttribute('aria-pressed', o === b ? 'true' : 'false'); });
      });
    });
  }

  // -- lazy 3D surface (echarts-gl only on demand, SRI-checked) ----
  function init3d() {
    $$('[data-load-3d]').forEach(function (b) {
      b.addEventListener('click', function () {
        var target = doc.getElementById(b.getAttribute('data-load-3d'));
        var holder = target && doc.getElementById(target.getAttribute('data-spec3d'));
        var spec = holder && JSON.parse(holder.textContent);
        if (!spec || !window.echarts) return;
        var probe = doc.createElement('canvas');
        if (!(probe.getContext('webgl') || probe.getContext('experimental-webgl'))) {
          b.textContent = 'WebGL unavailable: the 2D heatmap is the full view'; b.disabled = true; return;
        }
        b.disabled = true; b.textContent = 'Loading 3D view\u2026';
        var fail = function () { b.textContent = '3D view failed to load: the 2D heatmap is the full view'; };
        // the points: the session's data file (Surface), or the spec's own (a model grid)
        var pts = spec.src ? Desk.data(spec.src, spec.integrity, spec.session).then(function (j) { return j.surface3d.points; }) : Promise.resolve(spec.points);
        var s = doc.createElement('script');
        s.src = b.getAttribute('data-src'); s.integrity = b.getAttribute('data-sri'); s.crossOrigin = 'anonymous';
        Promise.all([pts, new Promise(function (ok, no) { s.onload = ok; s.onerror = no; })]).then(function (r) {
          var points = r[0];
          target.hidden = false;
          var T = theme(false);
          var inst = window.echarts.init(target, null, { renderer: 'canvas' });
          var stops = spec.stops;
          var ax = function (n) { return { name: n, nameTextStyle: { color: T.text2, fontFamily: T.sans }, axisLabel: { color: T.text3, fontFamily: T.mono, fontSize: 10 }, axisLine: { lineStyle: { color: T.line2 } }, splitLine: { lineStyle: { color: T.line } } }; };
          inst.setOption({
            backgroundColor: 'transparent',
            tooltip: { formatter: function (p) { var v = p.value; return 'ln(K/F) ' + num(v[0], { digits: 3, signed: true }) + '<br>T ' + num(v[1], { digits: 0 }) + ' d<br>IV ' + num(v[2], { digits: 2, unit: '%' }); } },
            visualMap: { show: true, min: spec.zmin, max: spec.zmax, dimension: 2, right: 6, top: 10, itemHeight: 120, calculable: false, textStyle: { color: T.text2, fontFamily: T.mono, fontSize: 10 }, inRange: { color: stops } },
            grid3D: { viewControl: { projection: 'orthographic', alpha: 22, beta: 38, autoRotate: false, distance: 220 }, boxWidth: 120, boxDepth: 90, boxHeight: 60, environment: 'none', light: { main: { intensity: 1.1 }, ambient: { intensity: 0.45 } } },
            xAxis3D: Object.assign(ax('ln(K/F)'), { type: 'value' }), yAxis3D: Object.assign(ax('T (days)'), { type: 'value', min: 0 }), zAxis3D: Object.assign(ax('IV %'), { type: 'value', min: 0 }),
            series: [{ type: 'surface', data: points.map(function (q) { return [q[0], q[1], q[2] === null ? '-' : q[2]]; }), shading: 'color', wireframe: { show: true, lineStyle: { color: 'rgba(229,232,235,0.18)', width: 1 } }, itemStyle: { opacity: 0.94 } }],
          });
          Desk.charts[target.id] = { el: target, chart: inst, spec: spec };
          b.textContent = '3D view loaded (drag to rotate)';
        }, fail);
        doc.body.appendChild(s);
      });
    });
  }

  // -- navigation: menu button and g-shortcuts ----
  function initNav() {
    var btn = $('#navToggle'), links = $('#navLinks');
    if (btn && links) {
      var set = function (open) { links.classList.toggle('open', open); btn.setAttribute('aria-expanded', open ? 'true' : 'false'); };
      btn.addEventListener('click', function () { set(!links.classList.contains('open')); });
      doc.addEventListener('keydown', function (e) { if (e.key === 'Escape' && links.classList.contains('open')) { set(false); btn.focus(); } });
      doc.addEventListener('click', function (e) { if (links.classList.contains('open') && !btn.contains(e.target) && !links.contains(e.target)) set(false); });
    }
    // the phone strip's source list (Q20) closes like the menu; a tap on no
    // control fires no click in iOS Safari, so pointerdown too
    var sb = $('.sb-phone');
    if (sb) {
      doc.addEventListener('keydown', function (e) { if (e.key === 'Escape' && sb.open) { sb.open = false; sb.firstElementChild.focus(); } });
      ['click', 'pointerdown'].forEach(function (t) { doc.addEventListener(t, function (e) { if (sb.open && !sb.contains(e.target)) sb.open = false; }); });
    }
    var dlg = $('#keysDialog'), pending = false, timer;
    var go = {};
    $$('.nav-link[data-key]').forEach(function (a) { go[a.getAttribute('data-key')] = a.getAttribute('href'); });
    doc.addEventListener('keydown', function (e) {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      var t = e.target;
      if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
      if (e.key === '?' && dlg && dlg.showModal) { e.preventDefault(); dlg.showModal(); return; }
      if ((e.key === '[' || e.key === ']') && Desk.stepExpiry && Desk.stepExpiry(e.key === ']' ? 1 : -1)) { e.preventDefault(); return; }
      if (pending && go[e.key]) { pending = false; location.href = go[e.key]; return; }
      if (e.key === 'g') { pending = true; clearTimeout(timer); timer = setTimeout(function () { pending = false; }, 1200); }
    });
    $$('[data-close-dialog]').forEach(function (b) { b.addEventListener('click', function () { b.closest('dialog').close(); }); });
  }

  // -- view-time freshness (sessions behind the viewer's clock) ----
  function initFreshness() {
    var el = $('#freshness');
    if (!el) return;
    var dd = el.getAttribute('data-date');
    if (!dd || !/^\d{4}-\d{2}-\d{2}$/.test(dd)) return;
    var hol = {};
    try { JSON.parse(el.getAttribute('data-holidays') || '[]').forEach(function (h) { hol[h] = true; }); } catch (e) { /* no calendar: weekends only */ }
    var DAY = 864e5;
    var iso = function (t) { return new Date(t).toISOString().slice(0, 10); };
    var isSession = function (t) { var w = new Date(t).getUTCDay(); return w !== 0 && w !== 6 && !hol[iso(t)]; };
    var now = new Date();
    var t = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
    if (now.getUTCHours() < 22) t -= DAY;     // a session's close is published after ~22:00 UTC
    while (!isSession(t)) t -= DAY;
    var dataDay = Date.parse(dd + 'T00:00:00Z'), behind = 0;
    for (var x = t; x > dataDay && behind < 400; x -= DAY) { if (isSession(x)) behind++; }
    // short enough for the width the server's "as of <date>" reserved (no layout
    // shift after first paint); the explanation is in the title
    el.textContent = behind === 0 ? 'current' : (behind === 1 ? '1 session behind' : behind + ' sessions behind');
    el.title = 'Data as of ' + dd + (behind === 0 ? ': the newest NYSE session your clock has closed.'
      : ', ' + behind + ' NYSE session' + (behind === 1 ? '' : 's') + ' behind your clock' +
        (behind === 1 ? ' (the site rebuilds after each close, about 22:00 UTC).' : '.'));
    el.className = 'sb-v ' + (behind >= 5 ? 'lvl-error' : behind >= 2 ? 'lvl-warn' : 'lvl-ok');
    // the phone strip's source list carries the same reading (Q20)
    $$('[data-freshness-copy]').forEach(function (c) { c.textContent = el.textContent; c.title = el.title; c.className = el.className; });
  }

  // -- scroll regions: say so in text (not only the edge shadow) when cut off --
  function initScrollHints() {
    function mark() {
      $$('.tbl-wrap, .heat-wrap').forEach(function (el) {
        var over = el.scrollWidth > el.clientWidth + 1;
        el.setAttribute('data-overflow', over ? 'true' : 'false');
        var hint = el.previousElementSibling;
        if (!hint || !hint.classList.contains('scroll-hint')) {
          if (!over) return;
          hint = doc.createElement('p');
          hint.className = 'scroll-hint';
          hint.textContent = 'Scroll sideways for more columns';
          el.parentNode.insertBefore(hint, el);
        }
        hint.classList.toggle('show', over);
      });
    }
    mark();
    var t;
    window.addEventListener('resize', function () { clearTimeout(t); t = setTimeout(mark, 150); });
  }

  // -- card tables: explicit table roles, since display: block on a phone (desk.css
  //    .tbl.cards) drops the implicit ones in some browsers --
  function initTableRoles() {
    var roles = { TABLE: 'table', THEAD: 'rowgroup', TBODY: 'rowgroup', TR: 'row', TD: 'cell' };
    $$('table.cards').forEach(function (t) {
      [t].concat($$('thead, tbody, tr, th, td', t)).forEach(function (el) {
        var r = el.tagName === 'TH' ? (el.getAttribute('scope') === 'row' ? 'rowheader' : 'columnheader') : roles[el.tagName];
        if (r && !el.hasAttribute('role')) el.setAttribute('role', r);
      });
    });
  }

  // -- phone first screen (Q20): desk.css folds .fold-body at <= 600 px before the
  // first paint (<noscript> unfolds it); here its heading becomes a disclosure button.
  var NARROW = '(max-width: 600px)';
  function initNarrowDisclosure() {
    var heads = $$('[data-disclose]');
    if (!heads.length || !window.matchMedia) return;
    var mq = window.matchMedia(NARROW);
    var sync = function () {
      heads.forEach(function (h) {
        var b = h.querySelector('button.disclose'), body = doc.getElementById(h.getAttribute('data-disclose'));
        if (b && body) b.setAttribute('aria-expanded', (!mq.matches || body.classList.contains('unfolded')) ? 'true' : 'false');
      });
    };
    var enhance = function () {
      heads.forEach(function (h) {
        var body = doc.getElementById(h.getAttribute('data-disclose'));
        if (!body || h.querySelector('button.disclose')) return;
        var btn = doc.createElement('button');
        btn.type = 'button';
        btn.className = 'disclose';
        btn.setAttribute('aria-controls', body.id);
        while (h.firstChild) btn.appendChild(h.firstChild);
        h.appendChild(btn);
        btn.addEventListener('click', function () {
          var open = !body.classList.contains('unfolded');
          body.classList.toggle('unfolded', open);
          h.classList.toggle('unfolded', open);
          sync();
        });
      });
      sync();
    };
    if (mq.matches) enhance();
    var onChange = function () { if (mq.matches) enhance(); else sync(); };
    if (mq.addEventListener) mq.addEventListener('change', onChange); else if (mq.addListener) mq.addListener(onChange);
  }
  initNarrowDisclosure();

  // -- sortable, copyable tables (D12): a .tbl head sorts up, down, then back to the
  // published order (aria-sort); Copy = TSV of the shown rows, CSV = a download.
  var SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹', DASH = /^[—–-]+$/;
  // U+2212, $, %, $k / $M, 'm × 10⁻ⁿ', the first of 'a / b', 'a–b', 'a to b'; null for text
  // (and, whole === true, for anything but one number: not a map's index)
  function cellNumber(text, whole) {
    whole = whole === true;
    var s = String(text).replace(/−/g, '-').replace(/\s+/g, ' ').trim();
    if (!s || DASH.test(s)) return null;
    if (!whole) s = s.split(/ \/ |–| to /)[0].trim();
    var m = /^([+-]?)(\$?)(\d[\d,]*(?:\.\d+)?|\.\d+)(?: ?× ?10([⁻⁰¹²³⁴-⁹]+))?(%|k|M|bn)?( .*)?$/.exec(s);
    if (!m || (m[5] && m[5] !== '%' && !m[2]) || (whole && m[6])) return null;      // '1M' is a tenor
    var e = ({ k: 3, M: 6, bn: 9 }[m[5]] || 0) + (m[4] ? +m[4].replace(/./g, function (c) { return c === '⁻' ? '-' : SUP.indexOf(c); }) : 0);
    var v = parseFloat(m[3].replace(/,/g, '') + 'e' + e);    // 9.5e-9, not 9.500000000000001e-9
    return m[1] === '-' ? -v : v;
  }
  Desk.cellNumber = cellNumber;
  // a CSV cell: the number a cell is, else its text with an ASCII minus
  Desk.csvValue = function (x) { var v = cellNumber(x, true); return v === null ? x.replace(/−/g, '-') : v; };
  function cellText(c) {
    var k = c.cloneNode(true), src = $$('*', c);
    $$('*', k).forEach(function (x, i) {      // a block child (a tenor's expiry pair) is a word of its own
      if (!/^(inline|none|contents)$/.test(getComputedStyle(src[i]).display)) { x.before(' '); x.after(' '); }
    });
    $$('.sr-only, .th-sort-ind, [hidden], [aria-hidden="true"]', k).forEach(function (x) { x.remove(); });
    return k.textContent.replace(/\s+/g, ' ').trim();
  }
  function tableRows(t) {
    return [t.tHead.rows[0]].concat($$('tbody > tr', t).filter(function (r) { return !r.hidden; }))
      .map(function (r) { return Array.prototype.map.call(r.cells, cellText); });
  }
  Desk.tableTsv = function (t) { return tableRows(t).map(function (r) { return r.join('\t'); }).join('\n') + '\n'; };
  function sortable(t) {
    var h = t.tHead, rows = $$('tbody > tr', t);
    return !t.closest('.chart-data, .site-footer, [data-nosort]') && h && h.rows.length === 1 && rows.length > 1 &&
      !t.querySelector('[colspan], [rowspan]') && rows.every(function (r) { return r.cells.length === h.rows[0].cells.length; });
  }
  function initTableTools() {
    var tables = $$('table.tbl').filter(sortable), per = {};
    tables.forEach(function (t) { var p = t.closest('.panel'); if (p) per[p.id] = (per[p.id] || 0) + 1; });
    tables.forEach(function (t, ti) {
      var tbody = t.tBodies[0], rows = $$('tr', tbody), heads = $$('th', t.tHead.rows[0]), state = null;
      var initial = heads.map(function (h) { return h.getAttribute('aria-sort'); });
      var ord = function (r) { return +r.getAttribute('data-order'); };
      rows.forEach(function (r, i) { r.setAttribute('data-order', i); });
      heads.forEach(function (th, j) {
        if (th.querySelector('a, button, input, select, [tabindex]')) return;   // a head that is a link stays one
        var vals = rows.map(function (r) { return cellText(r.cells[j]); }).filter(function (x) { return x && !DASH.test(x); });
        var num = vals.length && vals.filter(function (x) { return cellNumber(x) !== null; }).length >= Math.max(1, vals.length / 2);
        var btn = doc.createElement('button'), lab = doc.createElement('span'), ind = doc.createElement('span');
        btn.type = 'button'; btn.className = 'th-sort'; lab.className = 'th-sort-label'; ind.className = 'th-sort-ind';
        ind.setAttribute('aria-hidden', 'true');
        while (th.firstChild) lab.appendChild(th.firstChild);
        btn.appendChild(lab); th.appendChild(btn); th.appendChild(ind); th.classList.add('sortable');
        btn.addEventListener('click', function () {
          var next = state && state.col === j ? (state.dir === 'ascending' ? 'descending' : null) : 'ascending';
          state = next && { col: j, dir: next };
          heads.forEach(function (h, i) { if (!next && initial[i]) h.setAttribute('aria-sort', initial[i]); else h.removeAttribute('aria-sort'); });
          var list = $$('tr', tbody), dir = next === 'ascending' ? 1 : -1, keys = new Map();
          if (next) th.setAttribute('aria-sort', next);
          list.forEach(function (r) { var x = cellText(r.cells[j]); keys.set(r, num ? cellNumber(x) : (x && !DASH.test(x) ? x : null)); });
          list.sort(function (a, b) {
            if (!next) return ord(a) - ord(b);
            var ka = keys.get(a), kb = keys.get(b);   // a dash or a word sorts last both ways
            if (ka === null || kb === null) return ka === kb ? ord(a) - ord(b) : (ka === null ? 1 : -1);
            return dir * (num ? ka - kb : String(ka).localeCompare(kb, undefined, { numeric: true })) || ord(a) - ord(b);
          }).forEach(function (r) { tbody.appendChild(r); });
        });
      });
      var cap = (t.caption && cellText(t.caption)) || 'table', p = t.closest('.panel');
      var mk = function (label, what, fn) {
        var b = doc.createElement('button');
        b.type = 'button'; b.className = 'btn btn-link tbl-act'; b.textContent = label;
        b.setAttribute('aria-label', what + ': ' + cap.slice(0, 80));
        b.addEventListener('click', function () { fn(b); });
        return b;
      };
      var copy = mk('Copy', 'Copy the table as tab-separated text', function (b) {
        var text = Desk.tableTsv(t), ok = function () { flash(b, 'Copied'); };
        if (navigator.clipboard) navigator.clipboard.writeText(text).then(ok, function () { flash(b, 'Blocked'); });
        else flash(b, 'Blocked');
      });
      var csv = mk('CSV', 'Download the table as CSV', function () {
        download((t.id || 'table-' + ti) + '.csv', tableRows(t).map(function (r) { return r.map(function (x) { return csvCell(Desk.csvValue(x)); }).join(','); }).join('\n') + '\n');
      });
      var acts = p && per[p.id] === 1 && p.querySelector('.panel-actions'), bar = acts;
      if (!acts) {
        bar = doc.createElement('div'); bar.className = 'tbl-tools';
        var at = t.closest('.tbl-wrap') || t;
        at.parentNode.insertBefore(bar, at);
      }
      bar.insertBefore(csv, bar.firstChild); bar.insertBefore(copy, csv);
    });
  }

  function boot() {
    root.classList.add('has-js');
    $$('.chart-groups').forEach(function (g) { g.removeAttribute('tabindex'); });   // it scrolls only without JS
    initNav(); initFreshness(); initTools(); initCharts(); initTabs(); initQuotes(); initPointReadouts(); initHeatmaps(); initExpiryCursor(); initUnits(); initFilters(); init3d(); initTableRoles(); initTableTools(); initScrollHints();
    initPrint(); openTarget(); scrollToAnchor();
    window.addEventListener('hashchange', openTarget);
  }
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', boot); else boot();
})();
