#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WEIGHTS (1 << 22)
#define REPEATS 24

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int8_t lut4[32 * 4];
static int8_t lut5[64 * 5];
static int8_t lut8[1024 * 8];
static uint8_t codes4[WEIGHTS / 4];
static uint8_t codes5[WEIGHTS / 5 + 1];
static uint16_t codes8[WEIGHTS / 8];
static float out[WEIGHTS];

static double run4(float scale, double *checksum) {
  double start = now_seconds();
  double acc = 0.0;
  for (int rep = 0; rep < REPEATS; ++rep) {
    for (size_t g = 0; g < WEIGHTS / 4; ++g) {
      const int8_t *e = &lut4[(size_t)codes4[g] * 4];
      float *d = &out[g * 4];
      d[0] = scale * (float)e[0];
      d[1] = scale * (float)e[1];
      d[2] = scale * (float)e[2];
      d[3] = scale * (float)e[3];
    }
    acc += (double)out[rep] + (double)out[WEIGHTS - 1 - rep];
  }
  *checksum += acc;
  return now_seconds() - start;
}

static double run5(float scale, double *checksum) {
  double start = now_seconds();
  double acc = 0.0;
  for (int rep = 0; rep < REPEATS; ++rep) {
    for (size_t g = 0; g < WEIGHTS / 5; ++g) {
      const int8_t *e = &lut5[(size_t)codes5[g] * 5];
      float *d = &out[g * 5];
      for (int i = 0; i < 5; ++i) d[i] = scale * (float)e[i];
    }
    acc += (double)out[rep] + (double)out[WEIGHTS - 1 - rep];
  }
  *checksum += acc;
  return now_seconds() - start;
}

static double run8(float scale, double *checksum) {
  double start = now_seconds();
  double acc = 0.0;
  for (int rep = 0; rep < REPEATS; ++rep) {
    for (size_t g = 0; g < WEIGHTS / 8; ++g) {
      const int8_t *e = &lut8[(size_t)codes8[g] * 8];
      float *d = &out[g * 8];
      for (int i = 0; i < 8; ++i) d[i] = scale * (float)e[i];
    }
    acc += (double)out[rep] + (double)out[WEIGHTS - 1 - rep];
  }
  *checksum += acc;
  return now_seconds() - start;
}

int main(void) {
  srandom(22);
  for (size_t i = 0; i < sizeof(lut4); ++i) lut4[i] = (int8_t)((random() % 3) - 1);
  for (size_t i = 0; i < sizeof(lut5); ++i) lut5[i] = (int8_t)((random() % 3) - 1);
  for (size_t i = 0; i < sizeof(lut8); ++i) lut8[i] = (int8_t)((random() % 3) - 1);
  for (size_t i = 0; i < WEIGHTS / 4; ++i) codes4[i] = (uint8_t)(random() % 32);
  for (size_t i = 0; i < WEIGHTS / 5; ++i) codes5[i] = (uint8_t)(random() % 64);
  for (size_t i = 0; i < WEIGHTS / 8; ++i) codes8[i] = (uint16_t)(random() % 1024);

  printf("int8 LUT sizes: 4/5=%zu B  5/6=%zu B  8/10=%zu B   (L1d=32768 B)\n",
         sizeof(lut4), sizeof(lut5), sizeof(lut8));
  printf("float32 LUT for 8/10 would be %zu B (exceeds L1d)\n\n", sizeof(lut8) * 4);

  double checksum = 0.0;
  double total = (double)WEIGHTS * REPEATS;
  double t4 = run4(0.5f, &checksum);
  double t5 = run5(0.5f, &checksum);
  double t8 = run8(0.5f, &checksum);
  double r4 = total / t4 / 1e6;
  double r5 = total / t5 / 1e6;
  double r8 = total / t8 / 1e6;
  printf("%-24s %7.3f s  %9.1f Mw/s  (baseline)\n", "4/5   32-entry 128 B", t4, r4);
  printf("%-24s %7.3f s  %9.1f Mw/s  %+6.1f%%\n", "5/6   64-entry 320 B", t5, r5,
         100.0 * (r5 / r4 - 1.0));
  printf("%-24s %7.3f s  %9.1f Mw/s  %+6.1f%%\n", "8/10 1024-entry 8 KB", t8, r8,
         100.0 * (r8 / r4 - 1.0));
  printf("\nchecksum %.6f (prevents dead-code elimination)\n", checksum);
  return 0;
}

