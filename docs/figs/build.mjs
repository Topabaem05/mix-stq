import { assembleVegaLite } from 'flint-chart';
import fs from 'node:fs';

const rows = [
  { tier: 'IQ2_XXS', bpw: 2.0625, accuracy: 37.50, kind: 'uniform' },
  { tier: 'IQ2_XS', bpw: 2.3125, accuracy: 37.50, kind: 'uniform' },
  { tier: 'IQ3low+IQ2high', bpw: 2.4375, accuracy: 39.50, kind: 'mixed' },
  { tier: 'IQ2_S', bpw: 2.5625, accuracy: 36.50, kind: 'uniform' },
  { tier: 'IQ2low+IQ3high', bpw: 2.6875, accuracy: 34.00, kind: 'mixed' },
  { tier: 'IQ3_XXS', bpw: 3.0625, accuracy: 45.00, kind: 'uniform' },
  { tier: 'IQ3_S', bpw: 3.4375, accuracy: 47.00, kind: 'uniform' },
  { tier: 'fp16 dense', bpw: 16.0, accuracy: 41.50, kind: 'baseline' },
];

const input = {
  data: { values: rows },
  semantic_types: {
    tier: 'Category',
    bpw: 'Number',
    accuracy: 'Percentage',
    kind: 'Category',
  },
  chart_spec: {
    chartType: 'Bar Chart',
    title: 'Accuracy is flat below 2.7 bits, then steps up at 3.06',
    subtitle: 'Top-1 accuracy on MMLU 140 + ARC-Challenge 60, OLMoE-1B-7B expert tensors, percent',
    encodings: {
      x: { field: 'tier' },
      y: { field: 'accuracy' },
      color: { field: 'kind' },
    },
  },
  theme_spec: 'economist',
};

const spec = assembleVegaLite(input);
fs.writeFileSync('vega_top1.json', JSON.stringify(spec, null, 1));
console.log('MARK', JSON.stringify(spec.mark ?? Object.keys(spec)));
console.log('WROTE vega_top1.json', JSON.stringify(spec).length, 'bytes');

