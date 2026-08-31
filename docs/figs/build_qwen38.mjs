import { assembleVegaLite } from 'flint-chart';
import fs from 'node:fs';

const rows = [
  { tier: 'dense BF16', accuracy: 87.0, encoder: 'BF16 baseline' },
  { tier: 'IQ3_XXS MLP', accuracy: 86.625, encoder: 'IQ3 reference' },
];

const input = {
  data: { values: rows },
  semantic_types: {
    tier: 'Category',
    accuracy: 'Percentage',
    encoder: 'Category',
  },
  chart_spec: {
    chartType: 'Bar Chart',
    title: 'Reference IQ3 stays within the 2-point BF16 margin',
    subtitle: 'Qwen3.8-27B, 800 items, MLP only; dense−IQ3 CI upper = +1.75 points',
    encodings: {
      x: { field: 'tier' },
      y: { field: 'accuracy' },
      color: { field: 'encoder' },
    },
  },
  theme_spec: 'economist',
};

const spec = assembleVegaLite(input);
spec.width = spec._width;
const labelFormat = '.3~f';
const valueLabel = spec.layer.find((layer) => layer.__themeSynthetic);
valueLabel.encoding.text.format = labelFormat;
spec._theme.decisions.dataLabels.format = labelFormat;
for (const report of [spec._theme.report, spec._theme.decisions.report]) {
  const numberFormat = report.find((entry) => entry.path === 'annotation.numberFormat');
  numberFormat.message = 'printed values use `.3~f` to preserve the measured value 86.625';
}
fs.writeFileSync('qwen38_top1.json', JSON.stringify(spec, null, 1));
console.log('WROTE qwen38_top1.json', JSON.stringify(spec).length, 'bytes');
