/* Copyright (c) by CryptoLab Inc. and Seoul National University R&DB Foundation.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#pragma once

#include "Define.h"
#include "DeviceVector.h"

namespace ckks {

__global__ void Intt8PointPerThreadPhase2OoP(
    const word64 *in, const int m, const int num_prime, const int N,
    const int start_prime_idx, const int radix, const word64 *base_inv,
    const word64 *base_inv_, const word64 *primes, word64 *out);

__global__ void Intt8PointPerThreadPhase1OoP(const word64 *in, const int m,
                                             const int num_prime, const int N,
                                             const int start_prime_idx, int pad,
                                             int radix, const word64 *base_inv,
                                             const word64 *base_inv_,
                                             const word64 *primes, word64 *out);

__global__ void Intt8PointPerThreadPhase1OoPWithEpilogue(
    const word64 *in, const int m, const int num_prime, const int N,
    const int start_prime_idx, int pad, int radix, const word64 *base_inv,
    const word64 *base_inv_, const word64 *primes, word64 *out,
    const word64 *epilogue, const word64 *epilogue_);

// A special case where start_length is 1
__global__ void modUpStepTwoSimple(const word64 *ptr_after_intt,
                                   const word64 *ptr_before_intt,
                                   const int in_prime_idx, const int degree,
                                   const word64 *primes,
                                   const word64 *barrett_ratios,
                                   const word64 *barrett_Ks,
                                   const word64 end_length, word64 *to);

__global__ void Ntt8PointPerThreadPhase1ExcludeSomeRange(
    uint64_t *op, const int m, const int num_prime, const int N,
    const int start_prime_idx, const int excluded_range_start,
    const int excluded_range_end, const int pad, const int radix,
    const word64 *base_inv, const word64 *base_inv_, const word64 *primes);

__global__ void Ntt8PointPerThreadPhase2ExcludeSomeRange(
    uint64_t *op, const int m, const int num_prime, const int N,
    const int start_prime_idx, const int excluded_range_start,
    const int excluded_range_end, const int radix, const word64 *base_inv,
    const word64 *base_inv_, const word64 *primes);

__global__ void Ntt8PointPerThreadPhase1(
    uint64_t *op, const int m, const int num_prime, const int N,
    const int start_prime_idx, const int pad, const int radix,
    const word64 *base_inv, const word64 *base_inv_, const word64 *primes);

__global__ void Ntt8PointPerThreadPhase2FusedWithSubNegateConstMult(
    uint64_t *op, const int m, const int num_prime, const int N,
    const int start_prime_idx, const int radix, const word64 *base_inv,
    const word64 *base_inv_, const word64 *primes, const word64 *op2,
    const word64 *epilogue, const word64 *epilogue_);

__global__ void Ntt8PointPerThreadPhase2(
    uint64_t *op, const int m, const int num_prime, const int N,
    const int start_prime_idx, const int radix, const word64 *base_inv,
    const word64 *base_inv_, const word64 *primes);

}  // namespace ckks