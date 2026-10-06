#include <stdint.h>
#include <stdio.h>

typedef void (*callback_t)(void *, uint32_t, uint32_t, uint32_t);
void mu_schedule(callback_t callback, void *arg, uint32_t occupancy) {
  uint32_t threads = occupancy * 16;
  for (uint32_t tid = 0; tid < threads; ++tid)
    callback(arg, tid, threads, 0);
}

extern int64_t entry(void);
int main(void) {
  int64_t actual = entry();
  if (actual != 6048) {
    fprintf(stderr, "STREAM copy sum: got %lld, expected 6048\n", (long long)actual);
    return 1;
  }
  puts("STREAM copy sum: 6048");
  return 0;
}
