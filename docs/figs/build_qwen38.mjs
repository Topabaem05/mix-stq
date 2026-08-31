import { assembleVegaLite } from 'flint-chart';
import fs from 'node:fs';

const rows = [
  { tier: 'dense FP16', accuracy: 80.5, encoder: 'baseline' },
  { tier: 'IQ2_XXS approx', accuracy: 73.0, encoder: 'not storable' },
  { tier: 'IQ3_XXS approx', accuracy: 79.0, encoder: 'not storable' },
  { tier: 'IQ3_XXS reference', accuracy: 77.0, encoder: 'reference' },
  { tier: 'IQ3_S approx', accuracy: 79.0, encoder: 'not storable' },
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
    title: 'Reference IQ3 is 3.5 points below dense FP16',
    subtitle: 'Qwen3.8-27B, MMLU 140 + ARC-Challenge 60; approximate IQ arms are not storable',
    encodings: {
      x: { field: 'tier' },
      y: { field: 'accuracy' },
      color: { field: 'encoder' },
    },
  },
  theme_spec: 'economist',
};

const spec = assembleVegaLite(input);
fs.writeFileSync('qwen38_top1.json', JSON.stringify(spec, null, 1));
console.log('WROTE qwen38_top1.json', JSON.stringify(spec).length, 'bytes');
