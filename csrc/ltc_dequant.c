#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

void dequantize_row_ltc1_0(const block_ltc1_0 *blocks, const ltc_codebook *book,
                           float *out, int64_t count) {
  const int64_t block_count = count / QK_LTC;
  for (int64_t b = 0; b < block_count; ++b) {
    const block_ltc1_0 *block = &blocks[b];
    const float scale = fp16_to_fp32(block->d);
    float *dst = out + b * QK_LTC;
    for (int group = 0; group < LTC_GROUPS; ++group) {
      const int bit_offset = group * LTC_CODE_BITS;
      const int byte_index = bit_offset >> 3;
      const int shift = bit_offset & 7;
      uint32_t window = (uint32_t)block->codes[byte_index];
      window |= (uint32_t)block->codes[byte_index + 1] << 8;
      const int code = (int)((window >> shift) & 0x1Fu);
      const int8_t *entry = book->lanes[code];
      float *lane_dst = dst + group * LTC_LANES;
      for (int lane = 0; lane < LTC_LANES; ++lane) {
        lane_dst[lane] = scale * (float)entry[lane];
      }
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s codebook.bin payload.bin count\n", argv[0]);
    return 2;
  }
  int64_t count = strtoll(argv[3], NULL, 10);
  int64_t block_count = count / QK_LTC;

  FILE *cf = fopen(argv[1], "rb");
  if (!cf) { fprintf(stderr, "cannot open codebook\n"); return 3; }
  uint8_t indices[LTC_CODEBOOK_ENTRIES];
  if (fread(indices, 1, LTC_CODEBOOK_ENTRIES, cf) != LTC_CODEBOOK_ENTRIES) {
    fprintf(stderr, "short codebook\n");
    fclose(cf);
    return 3;
  }
  fclose(cf);

  block_ltc1_0 *blocks = malloc((size_t)block_count * sizeof(block_ltc1_0));
  if (!blocks) { fprintf(stderr, "oom\n"); return 4; }
  FILE *pf = fopen(argv[2], "rb");
  if (!pf) { fprintf(stderr, "cannot open payload\n"); free(blocks); return 3; }
  size_t want = (size_t)block_count * sizeof(block_ltc1_0);
  if (fread(blocks, 1, want, pf) != want) {
    fprintf(stderr, "short payload: expected %zu bytes\n", want);
    fclose(pf);
    free(blocks);
    return 3;
  }
  fclose(pf);

  if (sizeof(block_ltc1_0) != 42) {
    fprintf(stderr, "block struct is %zu bytes, expected 42\n", sizeof(block_ltc1_0));
    free(blocks);
    return 5;
  }

  ltc_codebook book;
  ltc_codebook_from_indices(indices, &book);

  float *out = malloc((size_t)count * sizeof(float));
  if (!out) { fprintf(stderr, "oom\n"); free(blocks); return 4; }
  dequantize_row_ltc1_0(blocks, &book, out, count);

  fwrite(out, sizeof(float), (size_t)count, stdout);
  free(out);
  free(blocks);
  return 0;
}

