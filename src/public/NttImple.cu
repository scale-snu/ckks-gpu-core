/* Copyright (c) by CryptoLab Inc. and Seoul National University R&DB Foundation.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#include <cuda_runtime.h>

#include "Basic.cuh"
#include "Context.h"
#include "Define.h"
#include "DeviceVector.h"
#include "NttImple.cuh"

namespace ckks {

__device__ __inline__ void butt_ntt_local(uint64_t &a, uint64_t &b,
                                          const uint64_t &w, const uint64_t &w_,
                                          const uint64_t p) {
  uint64_t two_p = 2 * p;
  uint64_t U = mul_and_reduce_shoup(b, w, w_, p);
  if (a >= two_p) a -= two_p;
  b = a + (two_p - U);
  a += U;
}

__device__ void butt_intt_local(word64 &x, word64 &y, const word64 &w,
                                const word64 &w_, const word64 &p) {
  const word64 two_p = 2 * p;
  const word64 T = two_p - y + x;
  word64 new_x = x + y;
  if (new_x >= two_p) new_x -= two_p;
  if (T & 1) new_x += p;
  x = (new_x >> 1);
  y = mul_and_reduce_shoup(T, w, w_, p);
}

__global__ void Intt8PointPerThreadPhase2OoP(
    const word64 *in, const int m, const int num_prime, const int N,
    const int start_prime_idx, const int radix, const word64 *base_inv,
    const word64 *base_inv_, const word64 *primes, word64 *out) {
  extern __shared__ uint64_t temp[];
  int set = threadIdx.x / radix;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 8 * num_prime);
       i += blockDim.x * gridDim.x) {
    // size of a block
    uint64_t local[8];
    int t = N / 2 / m;
    // prime idx
    int np_idx = i / (N / 8) + start_prime_idx;
    // index in N/2 range
    int N_idx = i % (N / 8);
    // i'th block
    int m_idx = N_idx / (t / 4);
    int t_idx = N_idx % (t / 4);
    // base address
    const uint64_t *in_addr = in + np_idx * N;
    uint64_t *out_addr = out + np_idx * N;
    const uint64_t *prime_table = primes;
    uint64_t prime = prime_table[np_idx];
    int N_init = 2 * m_idx * t + t_idx;
    __syncthreads();
    for (int j = 0; j < 8; j++) {
      temp[set * 8 * radix + t_idx + t / 4 * j] =
          *(in_addr + N_init + t / 4 * j);
    }
    __syncthreads();
    for (int l = 0; l < 8; l++) {
      local[l] = temp[set * 8 * radix + 8 * t_idx + l];
    }
    int tw_idx = m + m_idx;
    int tw_idx2 = (t / 4) * tw_idx + t_idx;
    const uint64_t *WInv = base_inv + N * np_idx;
    const uint64_t *WInv_ = base_inv_ + N * np_idx;
    for (int j = 0; j < 4; j++) {
      butt_intt_local(local[2 * j], local[2 * j + 1], WInv[4 * tw_idx2 + j],
                      WInv_[4 * tw_idx2 + j], prime);
    }
    for (int j = 0; j < 2; j++) {
      butt_intt_local(local[4 * j], local[4 * j + 2], WInv[2 * tw_idx2 + j],
                      WInv_[2 * tw_idx2 + j], prime);
      butt_intt_local(local[4 * j + 1], local[4 * j + 3], WInv[2 * tw_idx2 + j],
                      WInv_[2 * tw_idx2 + j], prime);
    }
    for (int j = 0; j < 4; j++) {
      butt_intt_local(local[j], local[j + 4], WInv[tw_idx2], WInv_[tw_idx2],
                      prime);
    }
    int tail = 0;
    __syncthreads();
    for (int l = 0; l < 8; l++) {
      temp[set * 8 * radix + 8 * t_idx + l] = local[l];
    }
    __syncthreads();
#pragma unroll
    for (int j = t / 32, k = 32; j > 0; j >>= 3, k *= 8) {
      int m_idx2 = t_idx / (k / 4);
      int t_idx2 = t_idx % (k / 4);
      for (int l = 0; l < 8; l++) {
        local[l] =
            temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
      }
      tw_idx2 = j * tw_idx + m_idx2;
      for (int l = 0; l < 4; l++) {
        butt_intt_local(local[2 * l], local[2 * l + 1], WInv[4 * tw_idx2 + l],
                        WInv_[4 * tw_idx2 + l], prime);
      }
      for (int l = 0; l < 2; l++) {
        butt_intt_local(local[4 * l], local[4 * l + 2], WInv[2 * tw_idx2 + l],
                        WInv_[2 * tw_idx2 + l], prime);
        butt_intt_local(local[4 * l + 1], local[4 * l + 3],
                        WInv[2 * tw_idx2 + l], WInv_[2 * tw_idx2 + l], prime);
      }
      for (int l = 0; l < 4; l++) {
        butt_intt_local(local[l], local[l + 4], WInv[tw_idx2], WInv_[tw_idx2],
                        prime);
      }
      for (int l = 0; l < 8; l++) {
        temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
            local[l];
      }
      if (j == 2) tail = 1;
      if (j == 4) tail = 2;
      __syncthreads();
    }
    if (tail == 1) {
      for (int j = 0; j < 8; j++) {
        local[j] = temp[set * 8 * radix + t_idx + t / 4 * j];
      }
      butt_intt_local(local[0], local[4], WInv[tw_idx], WInv_[tw_idx], prime);
      butt_intt_local(local[1], local[5], WInv[tw_idx], WInv_[tw_idx], prime);
      butt_intt_local(local[2], local[6], WInv[tw_idx], WInv_[tw_idx], prime);
      butt_intt_local(local[3], local[7], WInv[tw_idx], WInv_[tw_idx], prime);
    } else if (tail == 2) {
      for (int j = 0; j < 8; j++) {
        local[j] = temp[set * 8 * radix + t_idx + t / 4 * j];
      }
      butt_intt_local(local[0], local[2], WInv[2 * tw_idx], WInv_[2 * tw_idx],
                      prime);
      butt_intt_local(local[1], local[3], WInv[2 * tw_idx], WInv_[2 * tw_idx],
                      prime);
      butt_intt_local(local[4], local[6], WInv[2 * tw_idx + 1],
                      WInv_[2 * tw_idx + 1], prime);
      butt_intt_local(local[5], local[7], WInv[2 * tw_idx + 1],
                      WInv_[2 * tw_idx + 1], prime);
      butt_intt_local(local[0], local[4], WInv[tw_idx], WInv_[tw_idx], prime);
      butt_intt_local(local[1], local[5], WInv[tw_idx], WInv_[tw_idx], prime);
      butt_intt_local(local[2], local[6], WInv[tw_idx], WInv_[tw_idx], prime);
      butt_intt_local(local[3], local[7], WInv[tw_idx], WInv_[tw_idx], prime);
    }
    for (int j = 0; j < 8; j++) {
      *(out_addr + N_init + t / 4 * j) = local[j];
    }
  }
}

__global__ void Intt8PointPerThreadPhase1OoP(
    const word64 *in, const int m, const int num_prime, const int N,
    const int start_prime_idx, int pad, int radix, const word64 *base_inv,
    const word64 *base_inv_, const word64 *primes, word64 *out) {
  extern __shared__ uint64_t temp[];
  int Warp_t = threadIdx.x % pad;
  int WarpID = threadIdx.x / pad;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 8 * num_prime);
       i += blockDim.x * gridDim.x) {
    // size of a block
    uint64_t local[8];
    int t = N / 2 / m;
    // prime idx
    int np_idx = i / (N / 8) + start_prime_idx;
    // index in N/2 range
    int N_idx = i % (N / 8);
    // i'th block
    int m_idx = N_idx / (t / 4);
    int t_idx = N_idx % (t / 4);
    // base address
    const uint64_t *in_addr = in + np_idx * N;
    uint64_t *out_addr = out + np_idx * N;
    const uint64_t *prime_table = primes;
    const uint64_t *WInv = base_inv + N * np_idx;
    const uint64_t *WInv_ = base_inv_ + N * np_idx;
    uint64_t prime = prime_table[np_idx];
    int N_init =
        2 * t / radix * WarpID + Warp_t + pad * (t_idx / (radix * pad));
    for (int j = 0; j < 8; j++) {
      local[j] = *(in_addr + N_init + t / 4 / radix * j);
    }
    int eradix = 8 * radix;
    int tw_idx = m + m_idx;
    int tw_idx2 = radix * tw_idx + WarpID;
    for (int j = 0; j < 4; j++) {
      butt_intt_local(local[2 * j], local[2 * j + 1], WInv[4 * tw_idx2 + j],
                      WInv_[4 * tw_idx2 + j], prime);
    }
    for (int j = 0; j < 2; j++) {
      butt_intt_local(local[4 * j], local[4 * j + 2], WInv[2 * tw_idx2 + j],
                      WInv_[2 * tw_idx2 + j], prime);
      butt_intt_local(local[4 * j + 1], local[4 * j + 3], WInv[2 * tw_idx2 + j],
                      WInv_[2 * tw_idx2 + j], prime);
    }
    for (int j = 0; j < 4; j++) {
      butt_intt_local(local[j], local[j + 4], WInv[tw_idx2], WInv_[tw_idx2],
                      prime);
    }
    for (int j = 0; j < 8; j++) {
      temp[Warp_t * (eradix + pad) + 8 * WarpID + j] = local[j];
    }
    int tail = 0;
    __syncthreads();
#pragma unroll
    for (int j = radix / 8, k = 32; j > 0; j >>= 3, k *= 8) {
      int m_idx2 = WarpID / (k / 4);
      int t_idx2 = WarpID % (k / 4);
      for (int l = 0; l < 8; l++) {
        local[l] = temp[(eradix + pad) * Warp_t + 2 * m_idx2 * k + t_idx2 +
                        (k / 4) * l];
      }
      int tw_idx2 = j * tw_idx + m_idx2;
      for (int l = 0; l < 4; l++) {
        butt_intt_local(local[2 * l], local[2 * l + 1], WInv[4 * tw_idx2 + l],
                        WInv_[4 * tw_idx2 + l], prime);
      }
      for (int l = 0; l < 2; l++) {
        butt_intt_local(local[4 * l], local[4 * l + 2], WInv[2 * tw_idx2 + l],
                        WInv_[2 * tw_idx2 + l], prime);
        butt_intt_local(local[4 * l + 1], local[4 * l + 3],
                        WInv[2 * tw_idx2 + l], WInv_[2 * tw_idx2 + l], prime);
      }
      for (int l = 0; l < 4; l++) {
        butt_intt_local(local[l], local[l + 4], WInv[tw_idx2], WInv_[tw_idx2],
                        prime);
      }
      for (int l = 0; l < 8; l++) {
        temp[(eradix + pad) * Warp_t + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
            local[l];
      }
      if (j == 2) tail = 1;
      if (j == 4) tail = 2;
      __syncthreads();
    }
    if (radix < 8) tail = (radix == 4) ? 2 : 1;
    for (int l = 0; l < 8; l++) {
      local[l] = temp[Warp_t * (eradix + pad) + WarpID + radix * l];
    }
    if (tail == 1) {
      butt_intt_local(local[0], local[4], WInv[tw_idx], WInv_[tw_idx], prime);
      butt_intt_local(local[1], local[5], WInv[tw_idx], WInv_[tw_idx], prime);
      butt_intt_local(local[2], local[6], WInv[tw_idx], WInv_[tw_idx], prime);
      butt_intt_local(local[3], local[7], WInv[tw_idx], WInv_[tw_idx], prime);
    } else if (tail == 2) {
      butt_intt_local(local[0], local[2], WInv[2 * tw_idx], WInv_[2 * tw_idx],
                      prime);
      butt_intt_local(local[1], local[3], WInv[2 * tw_idx], WInv_[2 * tw_idx],
                      prime);
      butt_intt_local(local[4], local[6], WInv[2 * tw_idx + 1],
                      WInv_[2 * tw_idx + 1], prime);
      butt_intt_local(local[5], local[7], WInv[2 * tw_idx + 1],
                      WInv_[2 * tw_idx + 1], prime);
      butt_intt_local(local[0], local[4], WInv[tw_idx], WInv_[tw_idx], prime);
      butt_intt_local(local[1], local[5], WInv[tw_idx], WInv_[tw_idx], prime);
      butt_intt_local(local[2], local[6], WInv[tw_idx], WInv_[tw_idx], prime);
      butt_intt_local(local[3], local[7], WInv[tw_idx], WInv_[tw_idx], prime);
    }
    for (int j = 0; j < 8; j++) {
      if (local[j] >= prime) local[j] -= prime;
    }
    N_init = t / 4 / radix * WarpID + Warp_t + pad * (t_idx / (radix * pad));
    for (int j = 0; j < 8; j++) {
      *(out_addr + N_init + t / 4 * j) = local[j];
    }
  }
}

__global__ void Intt8PointPerThreadPhase1OoPWithEpilogue(
    const word64 *in, const int m, const int num_prime, const int N,
    const int start_prime_idx, int pad, int radix, const word64 *base_inv,
    const word64 *base_inv_, const word64 *primes, word64 *out,
    const word64 *epilogue, const word64 *epilogue_) {
  extern __shared__ uint64_t temp[];
  int Warp_t = threadIdx.x % pad;
  int WarpID = threadIdx.x / pad;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 8 * num_prime);
       i += blockDim.x * gridDim.x) {
    // size of a block
    uint64_t local[8];
    int t = N / 2 / m;
    // prime idx
    int np_idx = i / (N / 8) + start_prime_idx;
    // index in N/2 range
    int N_idx = i % (N / 8);
    // i'th block
    int m_idx = N_idx / (t / 4);
    int t_idx = N_idx % (t / 4);
    // base address
    const uint64_t *in_addr = in + np_idx * N;
    uint64_t *out_addr = out + np_idx * N;
    const uint64_t *prime_table = primes;
    const uint64_t *WInv = base_inv + N * np_idx;
    const uint64_t *WInv_ = base_inv_ + N * np_idx;
    uint64_t prime = prime_table[np_idx];
    int N_init =
        2 * t / radix * WarpID + Warp_t + pad * (t_idx / (radix * pad));
    for (int j = 0; j < 8; j++) {
      local[j] = *(in_addr + N_init + t / 4 / radix * j);
    }
    int eradix = 8 * radix;
    int tw_idx = m + m_idx;
    int tw_idx2 = radix * tw_idx + WarpID;
    for (int j = 0; j < 4; j++) {
      butt_intt_local(local[2 * j], local[2 * j + 1], WInv[4 * tw_idx2 + j],
                      WInv_[4 * tw_idx2 + j], prime);
    }
    for (int j = 0; j < 2; j++) {
      butt_intt_local(local[4 * j], local[4 * j + 2], WInv[2 * tw_idx2 + j],
                      WInv_[2 * tw_idx2 + j], prime);
      butt_intt_local(local[4 * j + 1], local[4 * j + 3], WInv[2 * tw_idx2 + j],
                      WInv_[2 * tw_idx2 + j], prime);
    }
    for (int j = 0; j < 4; j++) {
      butt_intt_local(local[j], local[j + 4], WInv[tw_idx2], WInv_[tw_idx2],
                      prime);
    }
    for (int j = 0; j < 8; j++) {
      temp[Warp_t * (eradix + pad) + 8 * WarpID + j] = local[j];
    }
    int tail = 0;
    __syncthreads();
#pragma unroll
    for (int j = radix / 8, k = 32; j > 0; j >>= 3, k *= 8) {
      int m_idx2 = WarpID / (k / 4);
      int t_idx2 = WarpID % (k / 4);
      for (int l = 0; l < 8; l++) {
        local[l] = temp[(eradix + pad) * Warp_t + 2 * m_idx2 * k + t_idx2 +
                        (k / 4) * l];
      }
      int tw_idx2 = j * tw_idx + m_idx2;
      for (int l = 0; l < 4; l++) {
        butt_intt_local(local[2 * l], local[2 * l + 1], WInv[4 * tw_idx2 + l],
                        WInv_[4 * tw_idx2 + l], prime);
      }
      for (int l = 0; l < 2; l++) {
        butt_intt_local(local[4 * l], local[4 * l + 2], WInv[2 * tw_idx2 + l],
                        WInv_[2 * tw_idx2 + l], prime);
        butt_intt_local(local[4 * l + 1], local[4 * l + 3],
                        WInv[2 * tw_idx2 + l], WInv_[2 * tw_idx2 + l], prime);
      }
      for (int l = 0; l < 4; l++) {
        butt_intt_local(local[l], local[l + 4], WInv[tw_idx2], WInv_[tw_idx2],
                        prime);
      }
      for (int l = 0; l < 8; l++) {
        temp[(eradix + pad) * Warp_t + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
            local[l];
      }
      if (j == 2) tail = 1;
      if (j == 4) tail = 2;
      __syncthreads();
    }
    if (radix < 8) tail = (radix == 4) ? 2 : 1;
    for (int l = 0; l < 8; l++) {
      local[l] = temp[Warp_t * (eradix + pad) + WarpID + radix * l];
    }
    if (tail == 1) {
      butt_intt_local(local[0], local[4], WInv[tw_idx], WInv_[tw_idx], prime);
      butt_intt_local(local[1], local[5], WInv[tw_idx], WInv_[tw_idx], prime);
      butt_intt_local(local[2], local[6], WInv[tw_idx], WInv_[tw_idx], prime);
      butt_intt_local(local[3], local[7], WInv[tw_idx], WInv_[tw_idx], prime);
    } else if (tail == 2) {
      butt_intt_local(local[0], local[2], WInv[2 * tw_idx], WInv_[2 * tw_idx],
                      prime);
      butt_intt_local(local[1], local[3], WInv[2 * tw_idx], WInv_[2 * tw_idx],
                      prime);
      butt_intt_local(local[4], local[6], WInv[2 * tw_idx + 1],
                      WInv_[2 * tw_idx + 1], prime);
      butt_intt_local(local[5], local[7], WInv[2 * tw_idx + 1],
                      WInv_[2 * tw_idx + 1], prime);
      butt_intt_local(local[0], local[4], WInv[tw_idx], WInv_[tw_idx], prime);
      butt_intt_local(local[1], local[5], WInv[tw_idx], WInv_[tw_idx], prime);
      butt_intt_local(local[2], local[6], WInv[tw_idx], WInv_[tw_idx], prime);
      butt_intt_local(local[3], local[7], WInv[tw_idx], WInv_[tw_idx], prime);
    }
    for (int j = 0; j < 8; j++) {
      if (local[j] >= prime) local[j] -= prime;
    }
    N_init = t / 4 / radix * WarpID + Warp_t + pad * (t_idx / (radix * pad));
    for (int j = 0; j < 8; j++) {
      word64 after_epilogue = mul_and_reduce_shoup(local[j], epilogue[np_idx],
                                                   epilogue_[np_idx], prime);
      if (after_epilogue > prime) after_epilogue -= prime;
      *(out_addr + N_init + t / 4 * j) = after_epilogue;
    }
  }
}

// A special case where start_length is 1
__global__ void modUpStepTwoSimple(const word64 *ptr_after_intt,
                                   const word64 *ptr_before_intt,
                                   const int in_prime_idx, const int degree,
                                   const word64 *primes,
                                   const word64 *barrett_ratios,
                                   const word64 *barrett_Ks,
                                   const word64 end_length, word64 *to) {
  STRIDED_LOOP_START(degree * end_length, i);
  const int out_prime_idx = i / degree;
  const int degree_idx = i % degree;
  const auto barret_ratio = barrett_ratios[out_prime_idx];
  const auto barret_k = barrett_Ks[out_prime_idx];
  if (out_prime_idx != in_prime_idx) {
    const auto in = ptr_after_intt[degree_idx];
    if (primes[in_prime_idx] > primes[out_prime_idx]) {
      barret_reduction_64_64(in, to[i], primes[out_prime_idx], barret_ratio,
                             barret_k);
    } else {
      to[i] = in;
    }
  } else {
    to[i] = ptr_before_intt[degree_idx];
  }
  STRIDED_LOOP_END;
}

__global__ void Ntt8PointPerThreadPhase1ExcludeSomeRange(
    uint64_t *op, const int m, const int num_prime, const int N,
    const int start_prime_idx, const int excluded_range_start,
    const int excluded_range_end, const int pad, const int radix,
    const word64 *base_inv, const word64 *base_inv_, const word64 *primes) {
  extern __shared__ uint64_t temp[];
  int Warp_t = threadIdx.x % pad;
  int WarpID = threadIdx.x / pad;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 8 * num_prime);
       i += blockDim.x * gridDim.x) {
    // size of a block
    uint64_t local[8];
    int t = N / 2 / m;
    // prime idx
    int np_idx = i / (N / 8) + start_prime_idx;
    if (np_idx >= excluded_range_start && np_idx < excluded_range_end) continue;
    // index in N/2 range
    int N_idx = i % (N / 8);
    // i'th block
    int m_idx = N_idx / (t / 4);
    int t_idx = N_idx % (t / 4);
    // base address
    uint64_t *a_np = op + np_idx * N;
    const uint64_t *prime_table = primes;
    const uint64_t *W = base_inv + N * np_idx;
    const uint64_t *W_ = base_inv_ + N * np_idx;
    uint64_t prime = prime_table[np_idx];
    int N_init = 2 * m_idx * t + t / 4 / radix * WarpID + Warp_t +
                 pad * (t_idx / (radix * pad));
    for (int j = 0; j < 8; j++) {
      local[j] = *(a_np + N_init + t / 4 * j);
    }
    __syncthreads();
    int eradix = 8 * radix;
    int tw_idx = m + m_idx;
    for (int j = 0; j < 4; j++) {
      butt_ntt_local(local[j], local[j + 4], W[tw_idx], W_[tw_idx], prime);
    }
    for (int j = 0; j < 2; j++) {
      butt_ntt_local(local[4 * j], local[4 * j + 2], W[2 * tw_idx + j],
                     W_[2 * tw_idx + j], prime);
      butt_ntt_local(local[4 * j + 1], local[4 * j + 3], W[2 * tw_idx + j],
                     W_[2 * tw_idx + j], prime);
    }
    for (int j = 0; j < 4; j++) {
      butt_ntt_local(local[2 * j], local[2 * j + 1], W[4 * tw_idx + j],
                     W_[4 * tw_idx + j], prime);
    }
    for (int j = 0; j < 8; j++) {
      temp[Warp_t * (eradix + pad) + WarpID + radix * j] = local[j];
    }
    int tail = 0;
    __syncthreads();
#pragma unroll
    for (int j = 8, k = radix / 2; j < radix + 1; j *= 8, k >>= 3) {
      int m_idx2 = WarpID / (k / 4);
      int t_idx2 = WarpID % (k / 4);
      for (int l = 0; l < 8; l++) {
        local[l] = temp[(eradix + pad) * Warp_t + 2 * m_idx2 * k + t_idx2 +
                        (k / 4) * l];
      }
      int tw_idx2 = j * tw_idx + m_idx2;
      for (int j2 = 0; j2 < 4; j2++) {
        butt_ntt_local(local[j2], local[j2 + 4], W[tw_idx2], W_[tw_idx2],
                       prime);
      }
      for (int j2 = 0; j2 < 2; j2++) {
        butt_ntt_local(local[4 * j2], local[4 * j2 + 2], W[2 * tw_idx2 + j2],
                       W_[2 * tw_idx2 + j2], prime);
        butt_ntt_local(local[4 * j2 + 1], local[4 * j2 + 3],
                       W[2 * tw_idx2 + j2], W_[2 * tw_idx2 + j2], prime);
      }
      for (int j2 = 0; j2 < 4; j2++) {
        butt_ntt_local(local[2 * j2], local[2 * j2 + 1], W[4 * tw_idx2 + j2],
                       W_[4 * tw_idx2 + j2], prime);
      }

      for (int l = 0; l < 8; l++) {
        temp[(eradix + pad) * Warp_t + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
            local[l];
      }
      if (j == radix / 2) tail = 1;
      if (j == radix / 4) tail = 2;
      __syncthreads();
    }
    if (radix < 8) tail = (radix == 4) ? 2 : 1;
    if (tail == 1) {
      for (int l = 0; l < 8; l++) {
        local[l] = temp[(eradix + pad) * Warp_t + 8 * WarpID + l];
      }
      int tw_idx2 = (4 * radix) * tw_idx + 4 * WarpID;
      butt_ntt_local(local[0], local[1], W[tw_idx2], W_[tw_idx2], prime);
      butt_ntt_local(local[2], local[3], W[tw_idx2 + 1], W_[tw_idx2 + 1],
                     prime);
      butt_ntt_local(local[4], local[5], W[tw_idx2 + 2], W_[tw_idx2 + 2],
                     prime);
      butt_ntt_local(local[6], local[7], W[tw_idx2 + 3], W_[tw_idx2 + 3],
                     prime);
      for (int l = 0; l < 8; l++) {
        temp[(eradix + pad) * Warp_t + 8 * WarpID + l] = local[l];
      }
    } else if (tail == 2) {
      for (int l = 0; l < 8; l++) {
        local[l] = temp[(eradix + pad) * Warp_t + 8 * WarpID + l];
      }
      int tw_idx2 = 2 * radix * tw_idx + 2 * WarpID;
      butt_ntt_local(local[0], local[2], W[tw_idx2], W_[tw_idx2], prime);
      butt_ntt_local(local[1], local[3], W[tw_idx2], W_[tw_idx2], prime);
      butt_ntt_local(local[4], local[6], W[tw_idx2 + 1], W_[tw_idx2 + 1],
                     prime);
      butt_ntt_local(local[5], local[7], W[tw_idx2 + 1], W_[tw_idx2 + 1],
                     prime);
      butt_ntt_local(local[0], local[1], W[2 * tw_idx2], W_[2 * tw_idx2],
                     prime);
      butt_ntt_local(local[2], local[3], W[2 * tw_idx2 + 1],
                     W_[2 * tw_idx2 + 1], prime);
      butt_ntt_local(local[4], local[5], W[2 * tw_idx2 + 2],
                     W_[2 * tw_idx2 + 2], prime);
      butt_ntt_local(local[6], local[7], W[2 * tw_idx2 + 3],
                     W_[2 * tw_idx2 + 3], prime);
      for (int l = 0; l < 8; l++) {
        temp[(eradix + pad) * Warp_t + 8 * WarpID + l] = local[l];
      }
    }
    __syncthreads();
    for (int j = 0; j < 8; j++) {
      local[j] = temp[Warp_t * (eradix + pad) + WarpID + radix * j];
    }
    for (int j = 0; j < 8; j++) {
      *(a_np + N_init + t / 4 * j) = local[j];
    }
  }
}

__global__ void Ntt8PointPerThreadPhase2ExcludeSomeRange(
    uint64_t *op, const int m, const int num_prime, const int N,
    const int start_prime_idx, const int excluded_range_start,
    const int excluded_range_end, const int radix, const word64 *base_inv,
    const word64 *base_inv_, const word64 *primes) {
  extern __shared__ uint64_t temp[];
  int set = threadIdx.x / radix;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 8 * num_prime);
       i += blockDim.x * gridDim.x) {
    // size of a block
    uint64_t local[8];
    int t = N / 2 / m;
    // prime idx
    int np_idx = num_prime - 1 - (i / (N / 8)) + start_prime_idx;
    if (np_idx >= excluded_range_start && np_idx < excluded_range_end) continue;
    // index in N/2 range
    int N_idx = i % (N / 8);
    // i'th block
    int m_idx = N_idx / (t / 4);
    int t_idx = N_idx % (t / 4);
    // base address
    uint64_t *a_np = op + np_idx * N;
    const uint64_t *prime_table = primes;
    uint64_t prime = prime_table[np_idx];
    int N_init = 2 * m_idx * t + t_idx;
    for (int j = 0; j < 8; j++) {
      local[j] = *(a_np + N_init + t / 4 * j);
    }
    int tw_idx = m + m_idx;
    const uint64_t *W = base_inv + N * np_idx;
    const uint64_t *W_ = base_inv_ + N * np_idx;
    for (int j = 0; j < 4; j++) {
      butt_ntt_local(local[j], local[j + 4], W[tw_idx], W_[tw_idx], prime);
    }
    for (int j = 0; j < 2; j++) {
      butt_ntt_local(local[4 * j], local[4 * j + 2], W[2 * tw_idx + j],
                     W_[2 * tw_idx + j], prime);
      butt_ntt_local(local[4 * j + 1], local[4 * j + 3], W[2 * tw_idx + j],
                     W_[2 * tw_idx + j], prime);
    }
    for (int j = 0; j < 4; j++) {
      butt_ntt_local(local[2 * j], local[2 * j + 1], W[4 * tw_idx + j],
                     W_[4 * tw_idx + j], prime);
    }
    for (int j = 0; j < 8; j++) {
      temp[set * 8 * radix + t_idx + t / 4 * j] = local[j];
    }
    int tail = 0;
    __syncthreads();
#pragma unroll
    for (int j = 8, k = t / 8; j < t / 4 + 1; j *= 8, k >>= 3) {
      int m_idx2 = t_idx / (k / 4);
      int t_idx2 = t_idx % (k / 4);
      for (int l = 0; l < 8; l++) {
        local[l] =
            temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
      }
      int tw_idx2 = j * tw_idx + m_idx2;
      for (int j2 = 0; j2 < 4; j2++) {
        butt_ntt_local(local[j2], local[j2 + 4], W[tw_idx2], W_[tw_idx2],
                       prime);
      }
      for (int j2 = 0; j2 < 2; j2++) {
        butt_ntt_local(local[4 * j2], local[4 * j2 + 2], W[2 * tw_idx2 + j2],
                       W_[2 * tw_idx2 + j2], prime);
        butt_ntt_local(local[4 * j2 + 1], local[4 * j2 + 3],
                       W[2 * tw_idx2 + j2], W_[2 * tw_idx2 + j2], prime);
      }
      for (int j2 = 0; j2 < 4; j2++) {
        butt_ntt_local(local[2 * j2], local[2 * j2 + 1], W[4 * tw_idx2 + j2],
                       W_[4 * tw_idx2 + j2], prime);
      }

      for (int l = 0; l < 8; l++) {
        temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
            local[l];
      }
      if (j == t / 8) tail = 1;
      if (j == t / 16) tail = 2;
      __syncthreads();
    }
    if (tail == 1) {
      for (int l = 0; l < 8; l++) {
        local[l] = temp[set * 8 * radix + 8 * t_idx + l];
      }
      int tw_idx2 = t * tw_idx + 4 * t_idx;
      butt_ntt_local(local[0], local[1], W[tw_idx2], W_[tw_idx2], prime);
      butt_ntt_local(local[2], local[3], W[tw_idx2 + 1], W_[tw_idx2 + 1],
                     prime);
      butt_ntt_local(local[4], local[5], W[tw_idx2 + 2], W_[tw_idx2 + 2],
                     prime);
      butt_ntt_local(local[6], local[7], W[tw_idx2 + 3], W_[tw_idx2 + 3],
                     prime);
      for (int l = 0; l < 8; l++) {
        temp[set * 8 * radix + 8 * t_idx + l] = local[l];
      }
    } else if (tail == 2) {
      for (int l = 0; l < 8; l++) {
        local[l] = temp[set * 8 * radix + 8 * t_idx + l];
      }
      int tw_idx2 = (t / 2) * tw_idx + 2 * t_idx;
      butt_ntt_local(local[0], local[2], W[tw_idx2], W_[tw_idx2], prime);
      butt_ntt_local(local[1], local[3], W[tw_idx2], W_[tw_idx2], prime);
      butt_ntt_local(local[4], local[6], W[tw_idx2 + 1], W_[tw_idx2 + 1],
                     prime);
      butt_ntt_local(local[5], local[7], W[tw_idx2 + 1], W_[tw_idx2 + 1],
                     prime);
      butt_ntt_local(local[0], local[1], W[2 * tw_idx2], W_[2 * tw_idx2],
                     prime);
      butt_ntt_local(local[2], local[3], W[2 * tw_idx2 + 1],
                     W_[2 * tw_idx2 + 1], prime);
      butt_ntt_local(local[4], local[5], W[2 * tw_idx2 + 2],
                     W_[2 * tw_idx2 + 2], prime);
      butt_ntt_local(local[6], local[7], W[2 * tw_idx2 + 3],
                     W_[2 * tw_idx2 + 3], prime);
      for (int l = 0; l < 8; l++) {
        temp[set * 8 * radix + 8 * t_idx + l] = local[l];
      }
    }
    __syncthreads();
    for (int j = 0; j < 8; j++) {
      local[j] = temp[set * 8 * radix + t_idx + t / 4 * j];
      for (int k = 0; k < 3; k++) {
        if (local[j] >= prime) local[j] -= prime;
      }
    }
    for (int j = 0; j < 8; j++) {
      *(a_np + N_init + t / 4 * j) = local[j];
    }
  }
}

__global__ void Ntt8PointPerThreadPhase1(
    uint64_t *op, const int m, const int num_prime, const int N,
    const int start_prime_idx, const int pad, const int radix,
    const word64 *base_inv, const word64 *base_inv_, const word64 *primes) {
  extern __shared__ uint64_t temp[];
  int Warp_t = threadIdx.x % pad;
  int WarpID = threadIdx.x / pad;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 8 * num_prime);
       i += blockDim.x * gridDim.x) {
    // size of a block
    uint64_t local[8];
    int t = N / 2 / m;
    // prime idx
    int np_idx = i / (N / 8) + start_prime_idx;
    // index in N/2 range
    int N_idx = i % (N / 8);
    // i'th block
    int m_idx = N_idx / (t / 4);
    int t_idx = N_idx % (t / 4);
    // base address
    uint64_t *a_np = op + np_idx * N;
    const uint64_t *prime_table = primes;
    const uint64_t *W = base_inv + N * np_idx;
    const uint64_t *W_ = base_inv_ + N * np_idx;
    uint64_t prime = prime_table[np_idx];
    int N_init = 2 * m_idx * t + t / 4 / radix * WarpID + Warp_t +
                 pad * (t_idx / (radix * pad));
    for (int j = 0; j < 8; j++) {
      local[j] = *(a_np + N_init + t / 4 * j);
    }
    __syncthreads();
    int eradix = 8 * radix;
    int tw_idx = m + m_idx;
    for (int j = 0; j < 4; j++) {
      butt_ntt_local(local[j], local[j + 4], W[tw_idx], W_[tw_idx], prime);
    }
    for (int j = 0; j < 2; j++) {
      butt_ntt_local(local[4 * j], local[4 * j + 2], W[2 * tw_idx + j],
                     W_[2 * tw_idx + j], prime);
      butt_ntt_local(local[4 * j + 1], local[4 * j + 3], W[2 * tw_idx + j],
                     W_[2 * tw_idx + j], prime);
    }
    for (int j = 0; j < 4; j++) {
      butt_ntt_local(local[2 * j], local[2 * j + 1], W[4 * tw_idx + j],
                     W_[4 * tw_idx + j], prime);
    }
    for (int j = 0; j < 8; j++) {
      temp[Warp_t * (eradix + pad) + WarpID + radix * j] = local[j];
    }
    int tail = 0;
    __syncthreads();
#pragma unroll
    for (int j = 8, k = radix / 2; j < radix + 1; j *= 8, k >>= 3) {
      int m_idx2 = WarpID / (k / 4);
      int t_idx2 = WarpID % (k / 4);
      for (int l = 0; l < 8; l++) {
        local[l] = temp[(eradix + pad) * Warp_t + 2 * m_idx2 * k + t_idx2 +
                        (k / 4) * l];
      }
      int tw_idx2 = j * tw_idx + m_idx2;
      for (int j2 = 0; j2 < 4; j2++) {
        butt_ntt_local(local[j2], local[j2 + 4], W[tw_idx2], W_[tw_idx2],
                       prime);
      }
      for (int j2 = 0; j2 < 2; j2++) {
        butt_ntt_local(local[4 * j2], local[4 * j2 + 2], W[2 * tw_idx2 + j2],
                       W_[2 * tw_idx2 + j2], prime);
        butt_ntt_local(local[4 * j2 + 1], local[4 * j2 + 3],
                       W[2 * tw_idx2 + j2], W_[2 * tw_idx2 + j2], prime);
      }
      for (int j2 = 0; j2 < 4; j2++) {
        butt_ntt_local(local[2 * j2], local[2 * j2 + 1], W[4 * tw_idx2 + j2],
                       W_[4 * tw_idx2 + j2], prime);
      }

      for (int l = 0; l < 8; l++) {
        temp[(eradix + pad) * Warp_t + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
            local[l];
      }
      if (j == radix / 2) tail = 1;
      if (j == radix / 4) tail = 2;
      __syncthreads();
    }
    if (radix < 8) tail = (radix == 4) ? 2 : 1;
    if (tail == 1) {
      for (int l = 0; l < 8; l++) {
        local[l] = temp[(eradix + pad) * Warp_t + 8 * WarpID + l];
      }
      int tw_idx2 = (4 * radix) * tw_idx + 4 * WarpID;
      butt_ntt_local(local[0], local[1], W[tw_idx2], W_[tw_idx2], prime);
      butt_ntt_local(local[2], local[3], W[tw_idx2 + 1], W_[tw_idx2 + 1],
                     prime);
      butt_ntt_local(local[4], local[5], W[tw_idx2 + 2], W_[tw_idx2 + 2],
                     prime);
      butt_ntt_local(local[6], local[7], W[tw_idx2 + 3], W_[tw_idx2 + 3],
                     prime);
      for (int l = 0; l < 8; l++) {
        temp[(eradix + pad) * Warp_t + 8 * WarpID + l] = local[l];
      }
    } else if (tail == 2) {
      for (int l = 0; l < 8; l++) {
        local[l] = temp[(eradix + pad) * Warp_t + 8 * WarpID + l];
      }
      int tw_idx2 = 2 * radix * tw_idx + 2 * WarpID;
      butt_ntt_local(local[0], local[2], W[tw_idx2], W_[tw_idx2], prime);
      butt_ntt_local(local[1], local[3], W[tw_idx2], W_[tw_idx2], prime);
      butt_ntt_local(local[4], local[6], W[tw_idx2 + 1], W_[tw_idx2 + 1],
                     prime);
      butt_ntt_local(local[5], local[7], W[tw_idx2 + 1], W_[tw_idx2 + 1],
                     prime);
      butt_ntt_local(local[0], local[1], W[2 * tw_idx2], W_[2 * tw_idx2],
                     prime);
      butt_ntt_local(local[2], local[3], W[2 * tw_idx2 + 1],
                     W_[2 * tw_idx2 + 1], prime);
      butt_ntt_local(local[4], local[5], W[2 * tw_idx2 + 2],
                     W_[2 * tw_idx2 + 2], prime);
      butt_ntt_local(local[6], local[7], W[2 * tw_idx2 + 3],
                     W_[2 * tw_idx2 + 3], prime);
      for (int l = 0; l < 8; l++) {
        temp[(eradix + pad) * Warp_t + 8 * WarpID + l] = local[l];
      }
    }
    __syncthreads();
    for (int j = 0; j < 8; j++) {
      local[j] = temp[Warp_t * (eradix + pad) + WarpID + radix * j];
    }
    for (int j = 0; j < 8; j++) {
      *(a_np + N_init + t / 4 * j) = local[j];
    }
  }
}

__global__ void Ntt8PointPerThreadPhase2FusedWithSubNegateConstMult(
    uint64_t *op, const int m, const int num_prime, const int N,
    const int start_prime_idx, const int radix, const word64 *base_inv,
    const word64 *base_inv_, const word64 *primes, const word64 *op2,
    const word64 *epilogue, const word64 *epilogue_) {
  extern __shared__ uint64_t temp[];
  int set = threadIdx.x / radix;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 8 * num_prime);
       i += blockDim.x * gridDim.x) {
    // size of a block
    uint64_t local[8];
    int t = N / 2 / m;
    // prime idx
    int np_idx = num_prime - 1 - (i / (N / 8)) + start_prime_idx;
    // index in N/2 range
    int N_idx = i % (N / 8);
    // i'th block
    int m_idx = N_idx / (t / 4);
    int t_idx = N_idx % (t / 4);
    // base address
    uint64_t *a_np = op + np_idx * N;
    const uint64_t *prime_table = primes;
    uint64_t prime = prime_table[np_idx];
    int N_init = 2 * m_idx * t + t_idx;
    for (int j = 0; j < 8; j++) {
      local[j] = *(a_np + N_init + t / 4 * j);
    }
    int tw_idx = m + m_idx;
    const uint64_t *W = base_inv + N * np_idx;
    const uint64_t *W_ = base_inv_ + N * np_idx;
    for (int j = 0; j < 4; j++) {
      butt_ntt_local(local[j], local[j + 4], W[tw_idx], W_[tw_idx], prime);
    }
    for (int j = 0; j < 2; j++) {
      butt_ntt_local(local[4 * j], local[4 * j + 2], W[2 * tw_idx + j],
                     W_[2 * tw_idx + j], prime);
      butt_ntt_local(local[4 * j + 1], local[4 * j + 3], W[2 * tw_idx + j],
                     W_[2 * tw_idx + j], prime);
    }
    for (int j = 0; j < 4; j++) {
      butt_ntt_local(local[2 * j], local[2 * j + 1], W[4 * tw_idx + j],
                     W_[4 * tw_idx + j], prime);
    }
    for (int j = 0; j < 8; j++) {
      temp[set * 8 * radix + t_idx + t / 4 * j] = local[j];
    }
    int tail = 0;
    __syncthreads();
#pragma unroll
    for (int j = 8, k = t / 8; j < t / 4 + 1; j *= 8, k >>= 3) {
      int m_idx2 = t_idx / (k / 4);
      int t_idx2 = t_idx % (k / 4);
      for (int l = 0; l < 8; l++) {
        local[l] =
            temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
      }
      int tw_idx2 = j * tw_idx + m_idx2;
      for (int j2 = 0; j2 < 4; j2++) {
        butt_ntt_local(local[j2], local[j2 + 4], W[tw_idx2], W_[tw_idx2],
                       prime);
      }
      for (int j2 = 0; j2 < 2; j2++) {
        butt_ntt_local(local[4 * j2], local[4 * j2 + 2], W[2 * tw_idx2 + j2],
                       W_[2 * tw_idx2 + j2], prime);
        butt_ntt_local(local[4 * j2 + 1], local[4 * j2 + 3],
                       W[2 * tw_idx2 + j2], W_[2 * tw_idx2 + j2], prime);
      }
      for (int j2 = 0; j2 < 4; j2++) {
        butt_ntt_local(local[2 * j2], local[2 * j2 + 1], W[4 * tw_idx2 + j2],
                       W_[4 * tw_idx2 + j2], prime);
      }

      for (int l = 0; l < 8; l++) {
        temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
            local[l];
      }
      if (j == t / 8) tail = 1;
      if (j == t / 16) tail = 2;
      __syncthreads();
    }
    if (tail == 1) {
      for (int l = 0; l < 8; l++) {
        local[l] = temp[set * 8 * radix + 8 * t_idx + l];
      }
      int tw_idx2 = t * tw_idx + 4 * t_idx;
      butt_ntt_local(local[0], local[1], W[tw_idx2], W_[tw_idx2], prime);
      butt_ntt_local(local[2], local[3], W[tw_idx2 + 1], W_[tw_idx2 + 1],
                     prime);
      butt_ntt_local(local[4], local[5], W[tw_idx2 + 2], W_[tw_idx2 + 2],
                     prime);
      butt_ntt_local(local[6], local[7], W[tw_idx2 + 3], W_[tw_idx2 + 3],
                     prime);
      for (int l = 0; l < 8; l++) {
        temp[set * 8 * radix + 8 * t_idx + l] = local[l];
      }
    } else if (tail == 2) {
      for (int l = 0; l < 8; l++) {
        local[l] = temp[set * 8 * radix + 8 * t_idx + l];
      }
      int tw_idx2 = (t / 2) * tw_idx + 2 * t_idx;
      butt_ntt_local(local[0], local[2], W[tw_idx2], W_[tw_idx2], prime);
      butt_ntt_local(local[1], local[3], W[tw_idx2], W_[tw_idx2], prime);
      butt_ntt_local(local[4], local[6], W[tw_idx2 + 1], W_[tw_idx2 + 1],
                     prime);
      butt_ntt_local(local[5], local[7], W[tw_idx2 + 1], W_[tw_idx2 + 1],
                     prime);
      butt_ntt_local(local[0], local[1], W[2 * tw_idx2], W_[2 * tw_idx2],
                     prime);
      butt_ntt_local(local[2], local[3], W[2 * tw_idx2 + 1],
                     W_[2 * tw_idx2 + 1], prime);
      butt_ntt_local(local[4], local[5], W[2 * tw_idx2 + 2],
                     W_[2 * tw_idx2 + 2], prime);
      butt_ntt_local(local[6], local[7], W[2 * tw_idx2 + 3],
                     W_[2 * tw_idx2 + 3], prime);
      for (int l = 0; l < 8; l++) {
        temp[set * 8 * radix + 8 * t_idx + l] = local[l];
      }
    }
    __syncthreads();
    for (int j = 0; j < 8; j++) {
      local[j] = temp[set * 8 * radix + t_idx + t / 4 * j];
      for (int k = 0; k < 3; k++) {
        if (local[j] >= prime) local[j] -= prime;
      }
    }
    for (int j = 0; j < 8; j++) {
      // sub-negate-constmult
      const auto after_epilogue =
          sub_negate_const_mult(local[j], op2[np_idx * N + N_init + t / 4 * j],
                                epilogue[np_idx], epilogue_[np_idx], prime);
      *(a_np + N_init + t / 4 * j) = after_epilogue;
    }
  }
}

__global__ void Ntt8PointPerThreadPhase2(
    uint64_t *op, const int m, const int num_prime, const int N,
    const int start_prime_idx, const int radix, const word64 *base_inv,
    const word64 *base_inv_, const word64 *primes) {
  extern __shared__ uint64_t temp[];
  int set = threadIdx.x / radix;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 8 * num_prime);
       i += blockDim.x * gridDim.x) {
    // size of a block
    uint64_t local[8];
    int t = N / 2 / m;
    // prime idx
    int np_idx = num_prime - 1 - (i / (N / 8)) + start_prime_idx;
    // index in N/2 range
    int N_idx = i % (N / 8);
    // i'th block
    int m_idx = N_idx / (t / 4);
    int t_idx = N_idx % (t / 4);
    // base address
    uint64_t *a_np = op + np_idx * N;
    const uint64_t *prime_table = primes;
    uint64_t prime = prime_table[np_idx];
    int N_init = 2 * m_idx * t + t_idx;
    for (int j = 0; j < 8; j++) {
      local[j] = *(a_np + N_init + t / 4 * j);
    }
    int tw_idx = m + m_idx;
    const uint64_t *W = base_inv + N * np_idx;
    const uint64_t *W_ = base_inv_ + N * np_idx;
    for (int j = 0; j < 4; j++) {
      butt_ntt_local(local[j], local[j + 4], W[tw_idx], W_[tw_idx], prime);
    }
    for (int j = 0; j < 2; j++) {
      butt_ntt_local(local[4 * j], local[4 * j + 2], W[2 * tw_idx + j],
                     W_[2 * tw_idx + j], prime);
      butt_ntt_local(local[4 * j + 1], local[4 * j + 3], W[2 * tw_idx + j],
                     W_[2 * tw_idx + j], prime);
    }
    for (int j = 0; j < 4; j++) {
      butt_ntt_local(local[2 * j], local[2 * j + 1], W[4 * tw_idx + j],
                     W_[4 * tw_idx + j], prime);
    }
    for (int j = 0; j < 8; j++) {
      temp[set * 8 * radix + t_idx + t / 4 * j] = local[j];
    }
    int tail = 0;
    __syncthreads();
#pragma unroll
    for (int j = 8, k = t / 8; j < t / 4 + 1; j *= 8, k >>= 3) {
      int m_idx2 = t_idx / (k / 4);
      int t_idx2 = t_idx % (k / 4);
      for (int l = 0; l < 8; l++) {
        local[l] =
            temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
      }
      int tw_idx2 = j * tw_idx + m_idx2;
      for (int j2 = 0; j2 < 4; j2++) {
        butt_ntt_local(local[j2], local[j2 + 4], W[tw_idx2], W_[tw_idx2],
                       prime);
      }
      for (int j2 = 0; j2 < 2; j2++) {
        butt_ntt_local(local[4 * j2], local[4 * j2 + 2], W[2 * tw_idx2 + j2],
                       W_[2 * tw_idx2 + j2], prime);
        butt_ntt_local(local[4 * j2 + 1], local[4 * j2 + 3],
                       W[2 * tw_idx2 + j2], W_[2 * tw_idx2 + j2], prime);
      }
      for (int j2 = 0; j2 < 4; j2++) {
        butt_ntt_local(local[2 * j2], local[2 * j2 + 1], W[4 * tw_idx2 + j2],
                       W_[4 * tw_idx2 + j2], prime);
      }

      for (int l = 0; l < 8; l++) {
        temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
            local[l];
      }
      if (j == t / 8) tail = 1;
      if (j == t / 16) tail = 2;
      __syncthreads();
    }
    if (tail == 1) {
      for (int l = 0; l < 8; l++) {
        local[l] = temp[set * 8 * radix + 8 * t_idx + l];
      }
      int tw_idx2 = t * tw_idx + 4 * t_idx;
      butt_ntt_local(local[0], local[1], W[tw_idx2], W_[tw_idx2], prime);
      butt_ntt_local(local[2], local[3], W[tw_idx2 + 1], W_[tw_idx2 + 1],
                     prime);
      butt_ntt_local(local[4], local[5], W[tw_idx2 + 2], W_[tw_idx2 + 2],
                     prime);
      butt_ntt_local(local[6], local[7], W[tw_idx2 + 3], W_[tw_idx2 + 3],
                     prime);
      for (int l = 0; l < 8; l++) {
        temp[set * 8 * radix + 8 * t_idx + l] = local[l];
      }
    } else if (tail == 2) {
      for (int l = 0; l < 8; l++) {
        local[l] = temp[set * 8 * radix + 8 * t_idx + l];
      }
      int tw_idx2 = (t / 2) * tw_idx + 2 * t_idx;
      butt_ntt_local(local[0], local[2], W[tw_idx2], W_[tw_idx2], prime);
      butt_ntt_local(local[1], local[3], W[tw_idx2], W_[tw_idx2], prime);
      butt_ntt_local(local[4], local[6], W[tw_idx2 + 1], W_[tw_idx2 + 1],
                     prime);
      butt_ntt_local(local[5], local[7], W[tw_idx2 + 1], W_[tw_idx2 + 1],
                     prime);
      butt_ntt_local(local[0], local[1], W[2 * tw_idx2], W_[2 * tw_idx2],
                     prime);
      butt_ntt_local(local[2], local[3], W[2 * tw_idx2 + 1],
                     W_[2 * tw_idx2 + 1], prime);
      butt_ntt_local(local[4], local[5], W[2 * tw_idx2 + 2],
                     W_[2 * tw_idx2 + 2], prime);
      butt_ntt_local(local[6], local[7], W[2 * tw_idx2 + 3],
                     W_[2 * tw_idx2 + 3], prime);
      for (int l = 0; l < 8; l++) {
        temp[set * 8 * radix + 8 * t_idx + l] = local[l];
      }
    }
    __syncthreads();
    for (int j = 0; j < 8; j++) {
      local[j] = temp[set * 8 * radix + t_idx + t / 4 * j];
      for (int k = 0; k < 3; k++) {
        if (local[j] >= prime) local[j] -= prime;
      }
    }
    for (int j = 0; j < 8; j++) {
      *(a_np + N_init + t / 4 * j) = local[j];
    }
  }
}

}  // namespace ckks