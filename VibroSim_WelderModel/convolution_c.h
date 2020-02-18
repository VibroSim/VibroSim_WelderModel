#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <omp.h>

#define ACCUMULATOR_PADFACTOR 16  // pad factor for accumulator entries; used because without padding the accumulator will cause horrible cache conflicts between cores. 8 should avoid conflicts between 64 byte L1 cache lines. We use 16 just in case. 

double rolled_inner_product_with_timereverse_c(double *history,long history_len,long rollshift_c,double *h,long h_len)
{

  long iter;
  size_t cnt;
  double *accumulator;
  double total=0.0;
  size_t max_threads=0;
  
  assert(history_len==h_len);
  assert(rollshift_c >= 0);

#pragma omp single
  max_threads=omp_get_max_threads();
  //printf("max_threads=%d\n",(int)max_threads);
  accumulator=(double *)malloc(max_threads*ACCUMULATOR_PADFACTOR*sizeof(*accumulator));
  for (cnt=0; cnt < max_threads; cnt++) {
    accumulator[cnt*ACCUMULATOR_PADFACTOR]=0.0;
  }
  
#pragma omp parallel default(none) private(iter) shared(accumulator,history,rollshift_c,history_len,h,max_threads)
#pragma omp for
  for (iter=0;iter < history_len;iter++) {
    accumulator[omp_get_thread_num()*ACCUMULATOR_PADFACTOR] += history[(rollshift_c + iter) % history_len]*h[history_len-1-iter];
  }

  #pragma omp single
  for (cnt=0; cnt < max_threads; cnt++) {
    total += accumulator[cnt*ACCUMULATOR_PADFACTOR];
  }

  return total;
}
