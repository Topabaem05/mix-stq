#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define QK_LTC 256
#define LTC_LANES 4
#define LTC_GROUPS (QK_LTC / LTC_LANES)
#define LTC_CODE_BITS 5
#define LTC_CODEBOOK_ENTRIES 32
#define LTC_CODE_BYTES (LTC_GROUPS * LTC_CODE_BITS / 8)

typedef struct {
  uint16_t d;
  uint8_t codes[LTC_CODE_BYTES];
} block_ltc1_0;

typedef struct {
  int8_t lanes[LTC_CODEBOOK_ENTRIES][LTC_LANES];
} ltc_codebook;

static float fp16_to_fp32(uint16_t value) {
  const uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
  const uint32_t exponent = (value >> 10) & 0x1Fu;
  const uint32_t mantissa = value & 0x3FFu;
  uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      uint32_t e = exponent;
      uint32_t m = mantissa;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        e -= 1;
      }
      m &= 0x3FFu;
      bits = sign | ((e + 127 - 15 + 1) << 23) | (m << 13);
    }
  } else if (exponent == 0x1Fu) {
    bits = sign | 0x7F800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  float out;
  memcpy(&out, &bits, sizeof(out));
  return out;
}

void ltc_codebook_from_indices(const uint8_t *indices, ltc_codebook *book) {
  for (int entry = 0; entry < LTC_CODEBOOK_ENTRIES; ++entry) {
    uint32_t remainder = indices[entry];
    for (int lane = 0; lane < LTC_LANES; ++lane) {
      book->lanes[entry][lane] = (int8_t)((int)(remainder % 3) - 1);
      remainder /= 3;
    }
  }
}

static inline int ltc_code_at(const uint8_t *codes, int group) {
  const int bit_offset = group * LTC_CODE_BITS;
  const int byte_index = bit_offset >> 3;
  const int shift = bit_offset & 7;
  uint32_t window = (uint32_t)codes[byte_index];
  window |= (uint32_t)codes[byte_index + 1] << 8;
  return (int)((window >> shift) & 0x1Fu);
}

void dequantize_row_ltc1_0(const block_ltc1_0 *blocks, const ltc_codebook *book,
                           float *out, int64_t count) {
  const int64_t block_count = count / QK_LTC;
  for (int64_t b = 0; b < block_count; ++b) {
    const float scale = fp16_to_fp32(blocks[b].d);
    float *dst = out + b * QK_LTC;
    for (int group = 0; group < LTC_GROUPS; ++group) {
      const int8_t *entry = book->lanes[ltc_code_at(blocks[b].codes, group)];
      float *lane_dst = dst + group * LTC_LANES;
      for (int lane = 0; lane < LTC_LANES; ++lane) {
        lane_dst[lane] = scale * (float)entry[lane];
      }
    }
  }
}

void ggml_vec_dot_ltc1_0(int64_t count, float *result, const block_ltc1_0 *blocks,
                         const ltc_codebook *book, const float *activations) {
  const int64_t block_count = count / QK_LTC;
  float total = 0.0f;
  for (int64_t b = 0; b < block_count; ++b) {
    const float scale = fp16_to_fp32(blocks[b].d);
    const float *act = activations + b * QK_LTC;
    float block_sum = 0.0f;
    for (int group = 0; group < LTC_GROUPS; ++group) {
      const int8_t *entry = book->lanes[ltc_code_at(blocks[b].codes, group)];
      const float *lane_act = act + group * LTC_LANES;
      block_sum += (float)entry[0] * lane_act[0]
                 + (float)entry[1] * lane_act[1]
                 + (float)entry[2] * lane_act[2]
                 + (float)entry[3] * lane_act[3];
    }
    total += scale * block_sum;
  }
  *result = total;
}

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s codebook.bin payload.bin activations.bin count\n", argv[0]);
    return 2;
  }
  const int64_t count = strtoll(argv[4], NULL, 10);
  const int64_t block_count = count / QK_LTC;
  if (block_count <= 0) {
    fprintf(stderr, "count must cover at least one block\n");
    return 2;
  }

  uint8_t indices[LTC_CODEBOOK_ENTRIES];
  FILE *cf = fopen(argv[1], "rb");
  if (!cf || fread(indices, 1, LTC_CODEBOOK_ENTRIES, cf) != LTC_CODEBOOK_ENTRIES) {
    fprintf(stderr, "cannot read codebook\n");
    if (cf) fclose(cf);
    return 3;
  }
  fclose(cf);

  block_ltc1_0 *blocks = malloc((size_t)block_count * sizeof(block_ltc1_0));
  float *acts = malloc((size_t)count * sizeof(float));
  float *deq = malloc((size_t)count * sizeof(float));
  if (!blocks || !acts || !deq) {
    fprintf(stderr, "oom\n");
    free(blocks); free(acts); free(deq);
    return 4;
  }

  FILE *pf = fopen(argv[2], "rb");
  size_t want = (size_t)block_count * sizeof(block_ltc1_0);
  if (!pf || fread(blocks, 1, want, pf) != want) {
    fprintf(stderr, "cannot read payload\n");
    if (pf) fclose(pf);
    free(blocks); free(acts); free(deq);
    return 3;
  }
  fclose(pf);

  FILE *af = fopen(argv[3], "rb");
  size_t awant = (size_t)count * sizeof(float);
  if (!af || fread(acts, 1, awant, af) != awant) {
    fprintf(stderr, "cannot read activations\n");
    if (af) fclose(af);
    free(blocks); free(acts); free(deq);
    return 3;
  }
  fclose(af);

  ltc_codebook book;
  ltc_codebook_from_indices(indices, &book);

  float fused = 0.0f;
  ggml_vec_dot_ltc1_0(count, &fused, blocks, &book, acts);

  dequantize_row_ltc1_0(blocks, &book, deq, count);
  float reference = 0.0f;
  for (int64_t b = 0; b < block_count; ++b) {
    float block_sum = 0.0f;
    for (int i = 0; i < QK_LTC; ++i) {
      const int64_t idx = b * QK_LTC + i;
      block_sum += deq[idx] * acts[idx];
    }
    reference += block_sum;
  }

  const int repeats = 200;
  double t0 = now_seconds();
  volatile float sink_fused = 0.0f;
  for (int r = 0; r < repeats; ++r) {
    float value = 0.0f;
    ggml_vec_dot_ltc1_0(count, &value, blocks, &book, acts);
    sink_fused += value;
  }
  double fused_time = now_seconds() - t0;

  t0 = now_seconds();
  volatile float sink_deq = 0.0f;
  for (int r = 0; r < repeats; ++r) {
    dequantize_row_ltc1_0(blocks, &book, deq, count);
    float value = 0.0f;
    for (int64_t i = 0; i < count; ++i) value += deq[i] * acts[i];
    sink_deq += value;
  }
  double deq_time = now_seconds() - t0;

  const double weights = (double)count * repeats;
  printf("fused_dot        %.8f\n", (double)fused);
  printf("dequant_then_dot %.8f\n", (double)reference);
  printf("abs_delta        %.3e\n", (double)(fused > reference ? fused - reference : reference - fused));
  printf("rel_delta        %.3e\n",
         reference != 0.0f ? (double)((fused - reference) / reference) : 0.0);
  printf("fused_throughput %.1f Mw/s\n", weights / fused_time / 1e6);
  printf("deq_throughput   %.1f Mw/s\n", weights / deq_time / 1e6);
  printf("fused_speedup    %.2fx\n", deq_time / fused_time);
  printf("sink %.3f %.3f\n", (double)sink_fused, (double)sink_deq);

  free(blocks); free(acts); free(deq);
  return 0;
}

