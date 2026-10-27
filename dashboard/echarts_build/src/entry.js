// Tree-shaken Apache ECharts core for the dashboard: only the chart types and
// components dashboard/static/desk.js uses. echarts-gl (loaded lazily on the
// surface page only) registers itself on this same `echarts` object, so the
// visual map and axis pointer it relies on are included here.
import * as echarts from 'echarts/core';
import { LineChart, BarChart, ScatterChart, CustomChart } from 'echarts/charts';
import {
  GridSimpleComponent, TooltipComponent, LegendPlainComponent, DataZoomInsideComponent, DataZoomSliderComponent,
  MarkLineComponent, MarkAreaComponent, VisualMapContinuousComponent, AxisPointerComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([
  LineChart, BarChart, ScatterChart, CustomChart,
  GridSimpleComponent, TooltipComponent, LegendPlainComponent, DataZoomInsideComponent, DataZoomSliderComponent,
  MarkLineComponent, MarkAreaComponent, VisualMapContinuousComponent, AxisPointerComponent,
  CanvasRenderer,
]);
window.echarts = echarts;
