#include <stdint.h>
#include <stdio.h>

typedef void (*callback_t)(void *, uint32_t, uint32_t, uint32_t);
void mu_schedule(callback_t callback, void *arg, uint32_t occupancy) {
  uint32_t threads = occupancy * 16;
  for (uint32_t block = 0; block < 2; ++block)
    for (uint32_t tid = 0; tid < threads; ++tid)
      callback(arg, tid, threads, block);
}
extern void entry(void);
extern int32_t get_count(int32_t i, int32_t j, int32_t k);
int main(void) {
  entry();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      for (int k = 0; k < 2; ++k)
        if (get_count(i, j, k) != 1) {
          fprintf(stderr, "count[%d,%d,%d]=%d\n", i, j, k, get_count(i, j, k));
          return 1;
        }
  puts("Muon 3D parallel distribution: 24 unique outputs passed");
  return 0;
}
