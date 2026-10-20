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
  var Desk = window.Desk = { charts: {}, filters: {} };
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
      },
    };
    if (print) {
      t.text = '#111111'; t.text2 = '#333333'; t.text3 = '#555555';
      t.line = '#bbbbbb'; t.line2 = '#999999'; t.grid = '#dddddd'; t.surface = '#ffffff'; t.tooltip = '#ffffff';
      t.colors.headline = '#111111'; t.colors.muted = '#666666';
      t.colors.cat3 = '#8a7a00';
    }
    return t;
  }
  function color(c, T) { return !c ? T.colors.cat0 : (c.charAt(0) === '#' ? c : (T.colors[c] || c)); }
  function alpha(hex, a) {
    var h = hex.replace('#', '');
    if (h.length !== 6) return hex;
    return 'rgba(' + parseInt(h.slice(0, 2), 16) + ',' + parseInt(h.slice(2, 4), 16) + ',' + parseInt(h.slice(4, 6), 16) + ',' + a + ')';
  }

  // -- spec -> ECharts option ----
  function axis(a, T, isX, spec) {
    a = a || {};
    var type = a.type || 'value';
    var ax = {
      type: type, name: a.name || '', nameLocation: 'middle', nameGap: isX ? 26 : 42,
      nameTextStyle: { color: T.text3, fontFamily: T.sans, fontSize: 11 },
      axisLine: { show: isX, lineStyle: { color: T.line2 } }, axisTick: { show: false },
      axisLabel: { color: T.text3, fontFamily: T.mono, fontSize: 10.5, hideOverlap: true },
      splitLine: { show: !isX || type === 'value', lineStyle: { color: T.grid } },
      scale: a.zero === undefined ? type === 'value' : !a.zero, inverse: !!a.inverse,
    };
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
    if (spec.xs && s.data && s.data.length && !Array.isArray(s.data[0])) {
      return s.data.map(function (y, i) { return [spec.xs[i], y]; });
    }
    return s.data || [];
  }

  function buildSeries(s, spec, T, i) {
    var c = color(s.color || ('cat' + (i % 6)), T);
    var base = { name: s.name, z: s.z || 2, yAxisIndex: s.y2 ? 1 : 0, emphasis: { focus: 'none' } };
    var data = seriesData(s, spec);
    if (s.type === 'errorbar') {
      return Object.assign(base, {
        type: 'custom', data: data, encode: { x: 0, y: [1, 2] }, tooltip: { show: true },
        renderItem: function (params, api) {
          var x = api.value(0), lo = api.coord([x, api.value(1)]), hi = api.coord([x, api.value(2)]);
          var w = 3;
          var st = { stroke: c, lineWidth: 1.2 };
          return { type: 'group', children: [
            { type: 'line', shape: { x1: lo[0], y1: lo[1], x2: hi[0], y2: hi[1] }, style: st },
            { type: 'line', shape: { x1: lo[0] - w, y1: lo[1], x2: lo[0] + w, y2: lo[1] }, style: st },
            { type: 'line', shape: { x1: hi[0] - w, y1: hi[1], x2: hi[0] + w, y2: hi[1] }, style: st },
          ] };
        },
      });
    }
    if (s.type === 'scatter') {
      return Object.assign(base, {
        type: 'scatter', data: data, symbol: s.symbol || 'circle', symbolSize: s.size || 6,
        itemStyle: s.hollow ? { color: 'transparent', borderColor: c, borderWidth: 1.4 } : { color: c, opacity: s.opacity || 0.9 },
      });
    }
    if (s.type === 'bar') {
      var bar = Object.assign(base, { type: 'bar', data: data, barMaxWidth: s.barMaxWidth || 28, stack: s.stack || undefined });
      if (s.invisible) { bar.itemStyle = { color: 'transparent' }; bar.tooltip = { show: false }; bar.silent = true; return bar; }
      if (s.signed) {
        bar.itemStyle = { color: function (p) { var v = Array.isArray(p.value) ? p.value[1] : p.value; return v < 0 ? T.colors.neg : T.colors.pos; } };
      } else { bar.itemStyle = { color: s.fill ? alpha(c, s.fill) : c, borderColor: c, borderWidth: s.fill ? 1 : 0 }; }
      return bar;
    }
    // line / area
    var line = Object.assign(base, {
      type: 'line', data: data, showSymbol: !!s.symbol, symbol: s.symbol || 'circle', symbolSize: s.size || 5,
      connectNulls: false, step: s.step || false, sampling: s.sampling || undefined,
      lineStyle: { color: c, width: s.width || 1.5, type: s.dash ? 'dashed' : 'solid' },
      itemStyle: { color: c },
    });
    if (s.area) line.areaStyle = { color: alpha(c, 0.18), origin: s.areaOrigin || 'auto' };
    return line;
  }

  function marks(spec, T, opt) {
    var m = spec.marks || {};
    if (!opt.series.length) return;
    var target = opt.series[0];
    if (m.y || m.x) {
      var data = [];
      (m.y || []).forEach(function (l) { data.push({ yAxis: l.value, name: l.label || '', lineStyle: { color: color(l.color || 'muted', T), type: l.dash === false ? 'solid' : 'dashed', width: 1 } }); });
      (m.x || []).forEach(function (l) { data.push({ xAxis: l.value, name: l.label || '', lineStyle: { color: color(l.color || 'muted', T), type: 'dashed', width: 1 } }); });
      target.markLine = { symbol: 'none', silent: true, animation: false, data: data,
        label: { show: true, position: 'insideEndTop', color: T.text3, fontFamily: T.mono, fontSize: 10, formatter: '{b}' } };
    }
    if (m.areas) {
      target.markArea = { silent: true, data: m.areas.map(function (a) {
        return [{ xAxis: a.from, name: a.label || '', itemStyle: { color: alpha(color(a.color || 'breach', T), a.alpha || 0.08) } }, { xAxis: a.to }];
      }), label: { color: T.text3, fontFamily: T.mono, fontSize: 10, position: 'insideTop' } };
    }
  }

  function tooltipFor(spec, T) {
    var y = spec.y || {}, x = spec.x || {};
    var fy = function (v, s) {
      var a = (s && s.y2) ? (spec.y2 || y) : y;
      return num(v, { digits: a.tipDigits === undefined ? (a.digits === undefined ? 2 : a.digits + 1) : a.tipDigits, signed: a.signed, unit: a.unit || '', prefix: a.prefix });
    };
    var fx = function (v) { return x.type === 'value' ? num(v, { digits: x.tipDigits === undefined ? (x.digits === undefined ? 2 : x.digits + 1) : x.tipDigits, signed: x.signed, unit: x.unit || '' }) : String(v); };
    var specSeries = {};
    (spec.series || []).forEach(function (s) { specSeries[s.name] = s; });
    return {
      trigger: spec.tooltip || 'axis', confine: true,
      axisPointer: { type: spec.tooltip === 'item' ? 'none' : 'line', lineStyle: { color: T.line2 } },
      backgroundColor: T.tooltip, borderColor: T.line2, borderWidth: 1, padding: [6, 8],
      textStyle: { color: T.text, fontFamily: T.mono, fontSize: 11 },
      formatter: function (ps) {
        ps = Array.isArray(ps) ? ps : [ps];
        var head = '', rows = [];
        ps.forEach(function (p) {
          var s = specSeries[p.seriesName] || {};
          if (s.invisible) return;
          var v = p.value;
          var xv = Array.isArray(v) ? v[0] : p.name;
          if (!head) head = (x.label ? x.label + ' ' : '') + fx(xv);
          if (v === null || v === undefined || v === '-') return;
          var txt;
          if (s.tip) txt = num(s.tip[p.dataIndex], { digits: (spec.y || {}).tipDigits === undefined ? 2 : spec.y.tipDigits, signed: true, unit: (spec.y || {}).unit || '' });
          else if (s.type === 'errorbar') txt = fy(v[1], s) + ' \u2013 ' + fy(v[2], s);
          else txt = fy(Array.isArray(v) ? v[1] : v, s);
          var sw = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:' + (typeof p.color === 'string' ? p.color : T.text2) + ';margin-right:6px"></span>';
          rows.push(sw + p.seriesName + '&nbsp;&nbsp;<b>' + txt + '</b>');
        });
        return '<div style="color:' + T.text3 + ';margin-bottom:2px">' + head + '</div>' + rows.join('<br>');
      },
    };
  }

  function buildOption(spec, T) {
    var opt = {
      animation: false, backgroundColor: 'transparent',
      textStyle: { fontFamily: T.sans, color: T.text2 },
      grid: Object.assign({ left: 54, right: spec.y2 ? 54 : 18, top: 14, bottom: spec.zoom ? 58 : 36, containLabel: false }, spec.grid || {}),
      xAxis: axis(spec.x, T, true, spec),
      yAxis: [axis(spec.y, T, false, spec)],
      tooltip: tooltipFor(spec, T),
      legend: { show: false },
      series: (spec.series || []).map(function (s, i) { return buildSeries(s, spec, T, i); }),
    };
    if (spec.y2) { var a2 = axis(spec.y2, T, false, spec); a2.splitLine = { show: false }; opt.yAxis.push(a2); }
    if (spec.horizontal) {
      var t = opt.xAxis; opt.xAxis = opt.yAxis[0]; opt.yAxis = [t];
      opt.xAxis.axisLine = { show: true, lineStyle: { color: T.line2 } }; opt.yAxis[0].inverse = true;
      opt.series.forEach(function (s) { s.data = (s.data || []).map(function (d) { return Array.isArray(d) ? [d[1], d[0]] : d; }); });
    }
    if (spec.zoom) {
      opt.dataZoom = [
        { type: 'inside', xAxisIndex: 0, filterMode: 'none' },
        { type: 'slider', xAxisIndex: 0, height: 16, bottom: 8, borderColor: T.line2, fillerColor: alpha(T.colors.info, 0.12),
          handleStyle: { color: T.text3, borderColor: T.line2 }, moveHandleStyle: { color: T.line2 },
          dataBackground: { lineStyle: { color: T.line2 }, areaStyle: { color: alpha(T.colors.muted, 0.15) } },
          textStyle: { color: T.text3, fontFamily: T.mono, fontSize: 10 }, labelFormatter: function (v, s) { return s; } },
      ];
    }
    marks(spec, T, opt);
    return opt;
  }

  // -- CSV export (from the same spec the chart draws) ----
  function csvCell(v) {
    if (v === null || v === undefined) return '';
    var s = String(v);
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }
  function specCsv(spec) {
    var lines = [];
    var xn = (spec.x && (spec.x.csv || spec.x.name)) || 'x';
    var series = (spec.series || []).filter(function (s) { return !s.invisible; });
    if (spec.xs) {
      lines.push([xn].concat(series.map(function (s) { return s.name; })).map(csvCell).join(','));
      spec.xs.forEach(function (x, i) {
        lines.push([x].concat(series.map(function (s) { var v = s.data[i]; return Array.isArray(v) ? v[1] : v; })).map(csvCell).join(','));
      });
    } else {
      lines.push(['series', xn, (spec.y && (spec.y.csv || spec.y.name)) || 'y', 'y_high'].map(csvCell).join(','));
      series.forEach(function (s) {
        (s.data || []).forEach(function (p) {
          var row = Array.isArray(p) ? p : [null, p];
          lines.push([s.name, row[0], row[1], row.length > 2 ? row[2] : ''].map(csvCell).join(','));
        });
      });
    }
    return lines.join('\n') + '\n';
  }
  Desk.specCsv = specCsv;
  function download(name, text) {
    var blob = new Blob([text], { type: 'text/csv;charset=utf-8' });
    var a = doc.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = name;
    doc.body.appendChild(a); a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 0);
  }

  // -- chart lifecycle ----
  function specOf(el) {
    var s = doc.getElementById(el.getAttribute('data-spec'));
    return s ? JSON.parse(s.textContent) : null;
  }
  function render(el, print) {
    if (!window.echarts) return null;
    var spec = specOf(el);
    if (!spec) return null;
    var inst = window.echarts.getInstanceByDom(el) || window.echarts.init(el, null, { renderer: 'canvas' });
    var opt = buildOption(spec, theme(print));
    var f = Desk.filters[el.id];
    if (f !== undefined) {           // a tab list shows one group of series (the smile expiry)
      opt.legend.selected = {};
      (spec.series || []).forEach(function (s) { if (s.group !== undefined) opt.legend.selected[s.name] = String(s.group) === String(f); });
    }
    inst.setOption(opt, true);
    // hidden series (legend state from the buttons, or the spec's defaults)
    $$('[data-legend-for="' + el.id + '"]').forEach(function (b) {
      if (b.getAttribute('aria-pressed') === 'false') inst.dispatchAction({ type: 'legendUnSelect', name: b.getAttribute('data-series') });
    });
    Desk.charts[el.id] = { el: el, chart: inst, spec: spec };
    return inst;
  }
  function renderAll(print) { $$('.chart[data-spec]').forEach(function (el) { render(el, print); }); }
  Desk.render = render;

  function initCharts() {
    if (!window.echarts) {
      root.classList.add('charts-failed');
      $$('.chart').forEach(function (w) {
        var p = doc.createElement('p');
        p.className = 'chart-fallback';
        p.textContent = 'Chart unavailable: the charting library did not load. The values are listed below.';
        w.parentNode.insertBefore(p, w.nextSibling);
      });
      return;
    }
    var els = $$('.chart[data-spec]');
    if ('IntersectionObserver' in window) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) { if (e.isIntersecting) { io.unobserve(e.target); try { render(e.target); } catch (err) { console.error('chart', e.target.id, err); } } });
      }, { rootMargin: '300px 0px' });
      els.forEach(function (el) { io.observe(el); });
    } else { els.forEach(function (el) { render(el); }); }
    var t;
    window.addEventListener('resize', function () { clearTimeout(t); t = setTimeout(function () { Object.keys(Desk.charts).forEach(function (k) { Desk.charts[k].chart.resize(); }); }, 120); });
    window.addEventListener('beforeprint', function () { renderAll(true); Object.keys(Desk.charts).forEach(function (k) { Desk.charts[k].chart.resize(); }); });
    window.addEventListener('afterprint', function () { renderAll(false); });
  }

  // -- chart toolbars: legend toggles, zoom presets, CSV, table ----
  function initTools() {
    $$('[data-legend-for]').forEach(function (b) {
      b.addEventListener('click', function () {
        var on = b.getAttribute('aria-pressed') !== 'true';
        b.setAttribute('aria-pressed', on ? 'true' : 'false');
        var c = Desk.charts[b.getAttribute('data-legend-for')];
        if (c) c.chart.dispatchAction({ type: on ? 'legendSelect' : 'legendUnSelect', name: b.getAttribute('data-series') });
      });
    });
    $$('[data-zoom-for]').forEach(function (b) {
      b.addEventListener('click', function () {
        var c = Desk.charts[b.getAttribute('data-zoom-for')];
        $$('[data-zoom-for="' + b.getAttribute('data-zoom-for') + '"]').forEach(function (o) { o.setAttribute('aria-pressed', o === b ? 'true' : 'false'); });
        if (!c) return;
        var years = parseFloat(b.getAttribute('data-years'));
        var xs = c.spec.xs || [];
        if (!years || !xs.length) { c.chart.dispatchAction({ type: 'dataZoom', start: 0, end: 100 }); return; }
        var last = new Date(xs[xs.length - 1]);
        var from = new Date(last); from.setFullYear(last.getFullYear() - years);
        c.chart.dispatchAction({ type: 'dataZoom', startValue: from.toISOString().slice(0, 10), endValue: xs[xs.length - 1] });
      });
    });
    $$('[data-csv-for]').forEach(function (b) {
      b.addEventListener('click', function () {
        var el = doc.getElementById(b.getAttribute('data-csv-for'));
        var spec = el && specOf(el);
        if (spec) download(b.getAttribute('data-filename') || (el.id + '.csv'), specCsv(spec));
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
        var id = b.getAttribute('data-copy-link');
        history.replaceState(null, '', '#' + id);
        var url = location.href;
        if (navigator.clipboard) navigator.clipboard.writeText(url).then(function () { flash(b, 'Copied'); }, function () { flash(b, 'Link set'); });
        else flash(b, 'Link set');
      });
    });
  }
  function flash(b, text) { var old = b.textContent; b.textContent = text; setTimeout(function () { b.textContent = old; }, 1200); }

  // -- URL-hash state: "#panel" anchors, or key=value pairs ----
  var state = {
    get: function (k) {
      var h = location.hash.replace(/^#/, '');
      if (h.indexOf('=') < 0) return null;
      var out = null;
      h.split('&').forEach(function (kv) { var p = kv.split('='); if (decodeURIComponent(p[0]) === k) out = decodeURIComponent(p[1] || ''); });
      return out;
    },
    set: function (k, v) {
      var h = location.hash.replace(/^#/, ''), parts = [];
      if (h.indexOf('=') >= 0) h.split('&').forEach(function (kv) { if (kv && decodeURIComponent(kv.split('=')[0]) !== k) parts.push(kv); });
      if (v !== null && v !== undefined) parts.push(encodeURIComponent(k) + '=' + encodeURIComponent(v));
      history.replaceState(null, '', parts.length ? '#' + parts.join('&') : location.pathname + location.search);
    },
  };
  Desk.state = state;

  // -- tabs (heatmap metric switch, smile expiry) ----
  function initTabs() {
    $$('[role="tablist"]').forEach(function (list) {
      var tabs = $$('[role="tab"]', list), key = list.getAttribute('data-state-key');
      var shared = !!list.getAttribute('data-chart-filter');   // every tab drives one panel
      function select(tab, focus) {
        tabs.forEach(function (t) {
          var on = t === tab;
          t.setAttribute('aria-selected', on ? 'true' : 'false');
          t.tabIndex = on ? 0 : -1;
          var p = doc.getElementById(t.getAttribute('aria-controls'));
          if (p && !shared) p.hidden = !on;
        });
        if (focus) tab.focus();
        if (key) state.set(key, tab.getAttribute('data-value'));
        var target = list.getAttribute('data-chart-filter');
        if (target) filterChart(target, tab.getAttribute('data-value'));
        $$('.chart[data-spec]', doc.getElementById(tab.getAttribute('aria-controls')) || doc.body).forEach(function (el) {
          if (Desk.charts[el.id]) Desk.charts[el.id].chart.resize(); else render(el);
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
      if (initial) select(initial, false);
      list.hidden = false;
    });
  }
  // A tab list can filter one chart's series by a "group" key (the smile slices).
  function filterChart(id, value) {
    Desk.filters[id] = value;
    var el = doc.getElementById(id);
    if (el && Desk.charts[id]) render(el);     // not yet drawn: render() applies the filter when it is
  }

  // -- heatmap cells: arrow-key navigation and a readout line ----
  function initHeatmaps() {
    $$('table.heat').forEach(function (tbl) {
      var out = doc.getElementById(tbl.getAttribute('data-readout'));
      var cells = $$('td[data-detail]', tbl);
      if (!cells.length) return;
      cells.forEach(function (c, i) { c.tabIndex = i === 0 ? 0 : -1; });
      var withHeads = tbl.getAttribute('data-prefix') === 'headers';
      function heads(c) {
        var th = c.parentNode.querySelector('th'), col = tbl.tHead && tbl.tHead.rows[0].cells[c.cellIndex];
        return (th ? th.textContent.trim() + ' \u00b7 ' : '') + (col ? col.textContent.trim() + ' \u00b7 ' : '');
      }
      function show(c) { if (out) out.textContent = (withHeads ? heads(c) : '') + c.getAttribute('data-detail'); }
      tbl.addEventListener('mouseover', function (e) { var c = e.target.closest('td[data-detail]'); if (c) show(c); });
      tbl.addEventListener('focusin', function (e) { var c = e.target.closest('td[data-detail]'); if (c) show(c); });
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
        var s = doc.createElement('script');
        s.src = b.getAttribute('data-src'); s.integrity = b.getAttribute('data-sri'); s.crossOrigin = 'anonymous';
        s.onload = function () {
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
            series: [{ type: 'surface', data: spec.points.map(function (q) { return [q[0], q[1], q[2] === null ? '-' : q[2]]; }), shading: 'color', wireframe: { show: true, lineStyle: { color: 'rgba(229,232,235,0.18)', width: 1 } }, itemStyle: { opacity: 0.94 } }],
          });
          Desk.charts[target.id] = { el: target, chart: inst, spec: spec };
          b.textContent = '3D view loaded (drag to rotate)';
        };
        s.onerror = function () { b.textContent = '3D view failed to load: the 2D heatmap is the full view'; };
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
      doc.addEventListener('click', function (e) { if (!btn.contains(e.target) && !links.contains(e.target)) set(false); });
    }
    var dlg = $('#keysDialog'), pending = false, timer;
    var go = {};
    $$('.nav-link[data-key]').forEach(function (a) { go[a.getAttribute('data-key')] = a.getAttribute('href'); });
    doc.addEventListener('keydown', function (e) {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      var t = e.target;
      if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
      if (e.key === '?' && dlg && dlg.showModal) { e.preventDefault(); dlg.showModal(); return; }
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
    el.textContent = behind === 0 ? 'current' : (behind === 1 ? '1 session behind (refreshes after the next close)' : behind + ' sessions behind');
    el.className = 'sb-v ' + (behind >= 5 ? 'lvl-error' : behind >= 2 ? 'lvl-warn' : 'lvl-ok');
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

  function boot() {
    root.classList.add('has-js');
    initNav(); initFreshness(); initTools(); initCharts(); initTabs(); initHeatmaps(); init3d(); initScrollHints();
  }
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', boot); else boot();
})();
