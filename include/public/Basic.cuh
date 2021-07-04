/* Copyright (c) by CryptoLab Inc.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#pragma once
#include <stdint.h>

#include "Define.h"

#define STRIDED_LOOP_START(N, i)                             \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; \
       i += blockDim.x * gridDim.x) {
#define STRIDED_LOOP_END }
#define TO_PTR(x) (x.data())

// https://gist.github.com/jefflarkin/5390993
#define CudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(0);                                                 \
    }                                                          \
  }

#ifdef __CUDACC__
#define CUDA_CALLABLE __device__
#define CUDA_CALLABLE_INLINE __inline__ __device__
#else
#define CUDA_CALLABLE
#define CUDA_CALLABLE_INLINE
#endif

namespace ckks {
struct uint128_t {
  uint64_t hi;
  uint64_t lo;
  __device__ uint128_t& operator+=(const uint128_t& op);
};

struct uint128_t4 {
  uint128_t x;
  uint128_t y;
  uint128_t z;
  uint128_t w;
};

// https://forums.developer.nvidia.com/t/long-integer-multiplication-mul-wide-u64-and-mul-wide-u128/51520
__inline__ __device__ uint128_t mult_64_64_128(const uint64_t op1,
                                               const uint64_t op2) {
  uint128_t temp;
  asm("{\n\t"
      ".reg .u32 p0l, p0h, p1l, p1h, p2l, p2h, p3l, p3h, r0, r1, r2, r3, "
      "alo, "
      "ahi, blo, bhi;\n\t"
      ".reg .u64 p0, p1, p2, p3;\n\t"
      "mov.b64         {alo,ahi}, %2;\n\t"
      "mov.b64         {blo,bhi}, %3;\n\t"
      "mul.wide.u32    p0, alo, blo;\n\t"
      "mul.wide.u32    p1, alo, bhi;\n\t"
      "mul.wide.u32    p2, ahi, blo;\n\t"
      "mul.wide.u32    p3, ahi, bhi;\n\t"
      "mov.b64         {p0l,p0h}, p0;\n\t"
      "mov.b64         {p1l,p1h}, p1;\n\t"
      "mov.b64         {p2l,p2h}, p2;\n\t"
      "mov.b64         {p3l,p3h}, p3;\n\t"
      "mov.b32         r0, p0l;\n\t"
      "add.cc.u32      r1, p0h, p1l;\n\t"
      "addc.cc.u32     r2, p1h, p2h;\n\t"
      "addc.u32        r3, p3h, 0;\n\t"
      "add.cc.u32      r1, r1, p2l;\n\t"
      "addc.cc.u32     r2, r2, p3l;\n\t"
      "addc.u32        r3, r3, 0;\n\t"
      "mov.b64         %0, {r0,r1};\n\t"
      "mov.b64         %1, {r2,r3};\n\t"
      "}"
      : "=l"(temp.lo), "=l"(temp.hi)
      : "l"(op1), "l"(op2));
  return temp;
}

__inline__ __device__ void inplace_add_128_128(const uint128_t op1,
                                               uint128_t& res) {
  asm("add.cc.u64 %1, %3, %1;\n\t"
      "addc.cc.u64 %0, %2, %0;\n\t"
      : "+l"(res.hi), "+l"(res.lo)
      : "l"(op1.hi), "l"(op1.lo));
}

__inline__ __device__ uint64_t
barret_reduction_128_64(const uint128_t in, const uint64_t prime,
                        const uint64_t barret_ratio, const uint64_t barret_k) {
  uint128_t temp1 = mult_64_64_128(in.lo, barret_ratio);
  uint128_t temp2 = mult_64_64_128(in.hi, barret_ratio);
  // carry = add_64_64_carry(temp1.hi, temp2.lo, temp1.hi);
  asm("add.cc.u64 %0, %0, %1;" : "+l"(temp1.hi) : "l"(temp2.lo));
  // carry = add_64_64_carry(temp2.hi, 0, temp2.hi, carry);
  asm("{addc.cc.u64 %0, %0, %1;}" : "+l"(temp2.hi) : "l"((unsigned long)0));
  temp1.hi >>= barret_k - 64;
  temp2.hi <<= 128 - barret_k;
  temp1.hi = temp1.hi + temp2.hi;
  temp1.hi = temp1.hi * prime;
  uint64_t res = in.lo - temp1.hi;
  if (res >= prime) res -= prime;
  return res;
}

__inline__ __device__ uint32_t
barret_reduction_64_32(const uint64_t in, const uint32_t prime,
                       const uint32_t barret_ratio, const uint64_t barret_k) {
  uint32_t temp1_hi = __umulhi(static_cast<uint32_t>(in), barret_ratio);
  const uint64_t temp2 = static_cast<uint64_t>(in >> 32) * barret_ratio;
  uint32_t temp2_hi = (temp2 >> 32);
  uint32_t temp2_lo = static_cast<uint32_t>(temp2);
  // carry = add_64_64_carry(temp1.hi, temp2.lo, temp1.hi);
  asm("add.cc.u32 %0, %0, %1;" : "+r"(temp1_hi) : "r"(temp2_lo));
  // carry = add_64_64_carry(temp2.hi, 0, temp2.hi, carry);
  asm("{addc.cc.u32 %0, %0, 0;}" : "+r"(temp2_hi));
  temp1_hi >>= barret_k - 32;
  temp2_hi <<= 64 - barret_k;
  temp1_hi = temp1_hi + temp2_hi;
  temp1_hi = temp1_hi * prime;
  uint32_t res = static_cast<uint32_t>(in) - temp1_hi;
  if (res >= prime) res -= prime;
  return res;
}

__inline__ __device__ void barret_reduction_64_64(const word64 in, word64& res,
                                                  const word64 prime,
                                                  const word64 ratio,
                                                  const word64 k) {
  word64 hi = __umul64hi(in, ratio);
  hi >>= (k - 64);
  res = in - (hi * prime);
  if (res >= prime) res -= prime;
}

__inline__ __device__ uint64_t mul_and_reduce_shoup(const uint64_t op1,
                                                    const uint64_t op2,
                                                    const uint64_t scaled_op2,
                                                    const uint64_t prime) {
  uint64_t hi = __umul64hi(scaled_op2, op1);
  return (uint64_t)op1 * op2 - hi * prime;
};

__inline__ __device__ uint64_t sub_negate_const_mult(const uint64_t op1,
                                                     const word64 op2,
                                                     const uint64_t op3,
                                                     const uint64_t scaled_op3,
                                                     const uint64_t prime) {
  word64 temp;
  if (op1 >= op2)
    temp = prime - op1 + op2;
  else {
    temp = op2 - op1;
  }
  word64 out = mul_and_reduce_shoup(temp, op3, scaled_op3, prime);
  if (out >= prime) out -= prime;
  return out;
};

__inline__ __device__ uint128_t& uint128_t::operator+=(const uint128_t& op) {
  inplace_add_128_128(op, *this);
  return *this;
}

}  // namespace ckks