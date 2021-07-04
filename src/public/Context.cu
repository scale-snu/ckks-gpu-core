/* Copyright (c) by CryptoLab Inc.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <tuple>
#include <memory>

#include "Basic.cuh"
#include "Ciphertext.h"
#include "Context.h"
#include "Define.h"
#include "EvaluationKey.h"
#include "MemoryPool.h"
#include "NttImple.cuh"

using namespace ckks;

namespace {

// from https://github.com/snucrypto/HEAAN, 131d275
void mulMod(uint64_t &r, uint64_t a, uint64_t b, uint64_t m) {
  unsigned __int128 mul = static_cast<unsigned __int128>(a) * b;
  mul %= static_cast<unsigned __int128>(m);
  r = static_cast<uint64_t>(mul);
}

auto MulMod(const word64 a, const word64 b, const word64 p) {
  word64 r;
  mulMod(r, a, b, p);
  return r;
}

// from https://github.com/snucrypto/HEAAN, 131d275
uint64_t powMod(uint64_t x, uint64_t y, uint64_t modulus) {
  uint64_t res = 1;
  while (y > 0) {
    if (y & 1) {
      mulMod(res, res, x, modulus);
    }
    y = y >> 1;
    mulMod(x, x, x, modulus);
  }
  return res;
}

// from https://github.com/snucrypto/HEAAN, 131d275
void findPrimeFactors(std::vector<uint64_t> &s, uint64_t number) {
  while (number % 2 == 0) {
    s.push_back(2);
    number /= 2;
  }
  for (uint64_t i = 3; i < sqrt(number); i++) {
    while (number % i == 0) {
      s.push_back(i);
      number /= i;
    }
  }
  if (number > 2) {
    s.push_back(number);
  }
}

// from https://github.com/snucrypto/HEAAN, 131d275
uint64_t findPrimitiveRoot(uint64_t modulus) {
  std::vector<uint64_t> s;
  uint64_t phi = modulus - 1;
  findPrimeFactors(s, phi);
  for (uint64_t r = 2; r <= phi; r++) {
    bool flag = false;
    for (auto it = s.begin(); it != s.end(); it++) {
      if (powMod(r, phi / (*it), modulus) == 1) {
        flag = true;
        break;
      }
    }
    if (flag == false) {
      return r;
    }
  }
  throw "Cannot find the primitive root of unity";
}

// from https://github.com/snucrypto/HEAAN, 131d275
auto BitReverse(std::vector<word64> &vals) {
  const int n = vals.size();
  for (long i = 1, j = 0; i < n; ++i) {
    long bit = n >> 1;
    for (; j >= bit; bit >>= 1) {
      j -= bit;
    }
    j += bit;
    if (i < j) {
      std::swap(vals[i], vals[j]);
    }
  }
}

word64 Inverse(const word64 op, const word64 prime) {
  word64 tmp = op > prime ? (op % prime) : op;
  return powMod(tmp, prime - 2, prime);
}

auto genBitReversedTwiddleFacotrs(const word64 root, const word64 p,
                                  const int degree) {
  std::vector<word64> pow_roots(degree), inv_pow_roots(degree);
  const auto root_inverse = Inverse(root, p);
  pow_roots[0] = 1;
  inv_pow_roots[0] = 1;
  for (int i = 1; i < degree; i++) {
    pow_roots[i] = MulMod(pow_roots[i - 1], root, p);
    inv_pow_roots[i] = MulMod(inv_pow_roots[i - 1], root_inverse, p);
  }
  BitReverse(pow_roots);
  BitReverse(inv_pow_roots);
  return std::pair{pow_roots, inv_pow_roots};
}

auto Shoup(const word64 in, const word64 prime) {
  word128 temp = static_cast<word128>(in) << 64;
  return static_cast<word64>(temp / prime);
}

auto ShoupEach(const std::vector<word64> &in, const word64 prime) {
  std::vector<word64> shoup;
  shoup.reserve(in.size());
  for (size_t i = 0; i < in.size(); i++) {
    shoup.push_back(Shoup(in[i], prime));
  }
  return shoup;
}

auto DivTwo(const std::vector<word64> &in, const word64 prime) {
  const auto two_inv = Inverse(2, prime);
  std::vector<word64> out(in.size());
  for (size_t i = 0; i < in.size(); i++) {
    out[i] = MulMod(in[i], two_inv, prime);
  }
  return out;
}

auto Flatten = [](const auto &device_vec) {
  size_t flat_size = 0;
  for (size_t i = 0; i < device_vec.size(); i++) {
    flat_size += device_vec[i].size();
  }
  DeviceVector flat;
  flat.resize(flat_size);
  word64 *out = flat.data();
  for (size_t i = 0; i < device_vec.size(); i++) {
    const auto &src = device_vec[i];
    cudaMemcpyAsync(out, src.data(), src.size() * sizeof(word64),
                    cudaMemcpyDefault);
    out += device_vec[i].size();
  }
  return flat;
};

auto Append = [](auto &vec, const auto &vec_to_append) {
  vec.insert(vec.end(), vec_to_append.begin(), vec_to_append.end());
};

auto ProductExcept = [](auto beg, auto end, word64 except, word64 modulus) {
  return std::accumulate(
      beg, end, (word64)1, [modulus, except](word64 accum, word64 prime) {
        return prime == except ? accum
                               : MulMod(accum, prime % modulus, modulus);
      });
};

auto ComputeQiHats(const std::vector<word64> &primes) {
  HostVector q_i_hats;
  std::transform(primes.begin(), primes.end(), back_inserter(q_i_hats),
                 [&primes](const auto modulus) {
                   return ProductExcept(primes.begin(), primes.end(), modulus,
                                        modulus);
                 });
  return q_i_hats;
}

auto ComputeQiModQj(const std::vector<word64> &primes) {
  std::vector<word64> hat_inv_vec;
  std::vector<word64> hat_inv_shoup_vec;
  const auto q_i_hats = ComputeQiHats(primes);
  std::transform(q_i_hats.begin(), q_i_hats.end(), primes.begin(),
                 back_inserter(hat_inv_vec), Inverse);
  std::transform(hat_inv_vec.begin(), hat_inv_vec.end(), primes.begin(),
                 back_inserter(hat_inv_shoup_vec), Shoup);
  return std::pair{hat_inv_vec, hat_inv_shoup_vec};
}

auto SetDifference = [](auto begin, auto end, auto remove_begin,
                        auto remove_end) {
  std::vector<word64> slimed;
  std::remove_copy_if(begin, end, std::back_inserter(slimed), [&](auto p) {
    return std::find(remove_begin, remove_end, p) != remove_end;
  });
  return slimed;
};

auto ComputeProdQiModQj = [](const auto &end_primes, auto start_begin,
                             auto start_end) {
  HostVector prod_q_i_mod_q_j;
  for (auto modulus : end_primes) {
    std::for_each(start_begin, start_end, [&](auto p) {
      prod_q_i_mod_q_j.push_back(
          ProductExcept(start_begin, start_end, p, modulus));
    });
  }
  return prod_q_i_mod_q_j;
};

}  // namespace

Context::Context(const Parameter &param)
    : param__{param},
      degree__{param.degree_},
      num_moduli_after_modup__{param.max_num_moduli_},
      alpha__{param.alpha_},
      primes__{param.primes_} {
  HostVector barret_k, barret_ratio, power_of_roots_vec,
      power_of_roots_shoup_vec, inv_power_of_roots_vec,
      inv_power_of_roots_shoup_vec;
  for (auto p : param.primes_) {
    long barret = floor(log2(p)) + 63;
    barret_k.push_back(barret);
    word128 temp = ((word64)1 << (barret - 64));
    temp <<= 64;
    barret_ratio.push_back((word64)(temp / p));
    auto root = findPrimitiveRoot(p);
    root = powMod(root, (p - 1) / (2 * degree__), p);
    auto [power_of_roots, inverse_power_of_roots] =
        genBitReversedTwiddleFacotrs(root, p, degree__);
    auto power_of_roots_shoup = ShoupEach(power_of_roots, p);
    auto inv_power_of_roots_div_two = DivTwo(inverse_power_of_roots, p);
    auto inv_power_of_roots_shoup = ShoupEach(inv_power_of_roots_div_two, p);
    Append(power_of_roots_vec, power_of_roots);
    Append(power_of_roots_shoup_vec, power_of_roots_shoup);
    Append(inv_power_of_roots_vec, inv_power_of_roots_div_two);
    Append(inv_power_of_roots_shoup_vec, inv_power_of_roots_shoup);
  }
  barret_ratio__ = DeviceVector(barret_ratio);
  barret_k__ = DeviceVector(barret_k);
  power_of_roots__ = DeviceVector(power_of_roots_vec);
  power_of_roots_shoup__ = DeviceVector(power_of_roots_shoup_vec);
  inverse_power_of_roots_div_two__ = DeviceVector(inv_power_of_roots_vec);
  inverse_scaled_power_of_roots_div_two__ =
      DeviceVector(inv_power_of_roots_shoup_vec);
  // make base-conversion-related parameters
  GenModUpParams();
  GenModDownParams();
}

void Context::GenModUpParams() {
  for (int dnum_idx = 0; dnum_idx < param__.dnum_; dnum_idx++) {
    auto prime_begin = param__.primes_.begin();
    auto prime_end = param__.primes_.end();
    auto start_begin = prime_begin + dnum_idx * alpha__;
    auto start_end = start_begin + alpha__;
    auto [hat_inv, hat_inv_shoup] =
        ComputeQiModQj(std::vector<word64>(start_begin, start_end));
    hat_inverse_vec__.push_back(DeviceVector(hat_inv));
    hat_inverse_vec_shoup__.push_back(DeviceVector(hat_inv_shoup));
    auto end_primes =
        SetDifference(prime_begin, prime_end, start_begin, start_end);
    prod_q_i_mod_q_j__.push_back(
        DeviceVector(ComputeProdQiModQj(end_primes, start_begin, start_end)));
  }
  hat_inverse_vec_batched__ = Flatten(hat_inverse_vec__);
  hat_inverse_vec_shoup_batched__ = Flatten(hat_inverse_vec_shoup__);
}

void Context::GenModDownParams() {
  auto prime_begin = param__.primes_.begin();
  auto prime_end = param__.primes_.end();
  for (int gap = 0; gap < param__.chain_length_; gap++) {
    int start_length = param__.num_special_moduli_ + gap;
    int end_length = param__.chain_length_ - gap;
    auto start_begin = param__.primes_.begin() + end_length;
    auto start_end = start_begin + start_length;
    auto [hat_inv, hat_inv_shoup] =
        ComputeQiModQj(std::vector<word64>(start_begin, start_end));
    hat_inverse_vec_moddown__.push_back(DeviceVector(hat_inv));
    hat_inverse_vec_shoup_moddown__.push_back(DeviceVector(hat_inv_shoup));
    auto end_primes =
        SetDifference(prime_begin, prime_end, start_begin, start_end);
    prod_q_i_mod_q_j_moddown__.push_back(
        DeviceVector(ComputeProdQiModQj(end_primes, start_begin, start_end)));
    std::vector<word64> prod_inv, prod_shoup;
    for (auto p : end_primes) {
      auto prod = ProductExcept(start_begin, start_end, 0, p);
      auto inv = Inverse(prod, p);
      prod_inv.push_back(inv);
      prod_shoup.push_back(Shoup(inv, p));
    }
    prod_inv_moddown__.push_back(DeviceVector(prod_inv));
    prod_inv_shoup_moddown__.push_back(DeviceVector(prod_shoup));
  }
}

// NTT is made up of two NTT stages. Here the 'first' means the first NTT
// stage in NTT (so it becomes second in view of iNTT).
DeviceVector Context::FromNTT(const DeviceVector &in) const {
  DeviceVector out;
  out.resize(in.size());
  const int batch = in.size() / degree__;
  const int start_prime_idx = 0;
  dim3 grid(2048);
  dim3 block(256);
  const int per_thread_ntt_size = 8;
  const int first_stage_radix_size = 256;
  const int second_radix_size = degree__ / first_stage_radix_size;
  const int pad = 4;
  const int per_thread_storage = block.x * per_thread_ntt_size * sizeof(word64);
  Intt8PointPerThreadPhase2OoP<<<grid, block, per_thread_storage>>>(
      in.data(), first_stage_radix_size, batch, degree__, start_prime_idx,
      second_radix_size / per_thread_ntt_size,
      inverse_power_of_roots_div_two__.data(),
      inverse_scaled_power_of_roots_div_two__.data(), primes__.data(),
      out.data());
  Intt8PointPerThreadPhase1OoP<<<grid, (first_stage_radix_size / 8) * pad,
                                 (first_stage_radix_size + pad + 1) * pad *
                                     sizeof(uint64_t)>>>(
      out.data(), 1, batch, degree__, start_prime_idx, pad,
      first_stage_radix_size / 8, inverse_power_of_roots_div_two__.data(),
      inverse_scaled_power_of_roots_div_two__.data(), primes__.data(),
      out.data());
  CudaCheckError();
  return out;
}

void Context::FromNTTInplace(word64 *op1, int start_prime_idx,
                             int batch) const {
  dim3 gridDim(2048);
  dim3 blockDim(256);
  // NTT is made up of two NTT stages. Here the 'first' means the first NTT
  // stage in NTT (so it becomes second in view of iNTT).
  const int per_thread_ntt_size = 8;
  const int first_stage_radix_size = 256;
  const int second_radix_size = degree__ / first_stage_radix_size;
  const int pad = 4;
  const int per_thread_storage =
      blockDim.x * per_thread_ntt_size * sizeof(word64);
  Intt8PointPerThreadPhase2OoP<<<gridDim, blockDim, per_thread_storage>>>(
      op1, first_stage_radix_size, batch, degree__, start_prime_idx,
      second_radix_size / per_thread_ntt_size,
      inverse_power_of_roots_div_two__.data(),
      inverse_scaled_power_of_roots_div_two__.data(), primes__.data(), op1);
  Intt8PointPerThreadPhase1OoP<<<gridDim, (first_stage_radix_size / 8) * pad,
                                 (first_stage_radix_size + pad + 1) * pad *
                                     sizeof(uint64_t)>>>(
      op1, 1, batch, degree__, start_prime_idx, pad, first_stage_radix_size / 8,
      inverse_power_of_roots_div_two__.data(),
      inverse_scaled_power_of_roots_div_two__.data(), primes__.data(), op1);
  CudaCheckError();
}

DeviceVector Context::FromNTT(const DeviceVector &in,
                              const DeviceVector &scale_constants,
                              const DeviceVector &scale_constants_shoup) const {
  DeviceVector out;
  out.resize(in.size());
  const int batch = in.size() / degree__;
  const int start_prime_idx = 0;
  dim3 grid(2048);
  dim3 block(256);
  const int per_thread_ntt_size = 8;
  const int first_stage_radix_size = 256;
  const int second_radix_size = degree__ / first_stage_radix_size;
  const int pad = 4;
  const int per_thread_storage = block.x * per_thread_ntt_size * sizeof(word64);
  Intt8PointPerThreadPhase2OoP<<<grid, block, per_thread_storage>>>(
      in.data(), first_stage_radix_size, batch, degree__, start_prime_idx,
      second_radix_size / per_thread_ntt_size,
      inverse_power_of_roots_div_two__.data(),
      inverse_scaled_power_of_roots_div_two__.data(), primes__.data(),
      out.data());
  Intt8PointPerThreadPhase1OoPWithEpilogue<<<
      grid, (first_stage_radix_size / 8) * pad,
      (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_t)>>>(
      out.data(), 1, batch, degree__, start_prime_idx, pad,
      first_stage_radix_size / 8, inverse_power_of_roots_div_two__.data(),
      inverse_scaled_power_of_roots_div_two__.data(), primes__.data(),
      out.data(), scale_constants.data(), scale_constants_shoup.data());
  CudaCheckError();
  return out;
}

DeviceVector Context::ModUp(const DeviceVector &in) const {
  const int num_moduli = in.size() / degree__;
  const int beta = num_moduli / alpha__;
  if (num_moduli % alpha__ != 0)
    throw std::logic_error("Size does not match; apply RNS-decompose first.");
  DeviceVector raised;
  raised.resize(num_moduli_after_modup__ * degree__ * beta);
  if (is_modup_batched)
    ModUpBatchImpl(in, raised, beta);
  else {
    for (int i = 0; i < beta; ++i) {
      ModUpImpl(in.data() + (alpha__ * degree__ * i),
                raised.data() + (num_moduli_after_modup__ * degree__) * i, i);
    }
  }
  return raised;
}

void Context::ModUpImpl(const word64 *from, word64 *to, int idx) const {
  if (alpha__ == 1) {
    DeviceVector temp(param__.max_num_moduli_ * degree__);
    cudaMemcpyAsync(temp.data() + idx * degree__, from, 8 * degree__,
                    cudaMemcpyDeviceToDevice, cudaStreamLegacy);
    FromNTTInplace(temp.data(), idx, alpha__);
    word64 *target = to;
    ModUpLengthIsOne(temp.data() + idx * degree__, from, idx,
                     num_moduli_after_modup__, target);
    ToNTTInplaceExceptSomeRange(target, 0, num_moduli_after_modup__, idx,
                                alpha__);
  } else {
    size_t begin_idx = idx * alpha__;
    cudaMemcpyAsync(to + (degree__ * begin_idx), from, 8 * alpha__ * degree__,
                    cudaMemcpyDeviceToDevice, cudaStreamLegacy);
    FromNTTInplace(to, begin_idx, alpha__);
    const DeviceVector &hat_inverse_vec = hat_inverse_vec__.at(idx);
    const DeviceVector &hat_inverse_vec_psinv = hat_inverse_vec_shoup__.at(idx);
    ConstMultBatch(to, hat_inverse_vec, hat_inverse_vec_psinv, begin_idx,
                   alpha__, to);
    ModUpMatMul(to + degree__ * begin_idx, idx, to);
    ToNTTInplaceExceptSomeRange(to, 0, num_moduli_after_modup__, begin_idx,
                                alpha__);
    cudaMemcpyAsync(to + (degree__ * begin_idx), from, 8 * alpha__ * degree__,
                    cudaMemcpyDeviceToDevice, cudaStreamLegacy);
  }
}
__global__ void constMultBatch_(size_t degree, const word64 *primes,
                                const word64 *op1, const word64 *op2,
                                const word64 *op2_psinv,
                                const int start_prime_idx, const int batch,
                                word64 *to) {
  STRIDED_LOOP_START(degree * batch, i);
  const int op2_idx = i / degree;
  const int prime_idx = op2_idx + start_prime_idx;
  const auto prime = primes[prime_idx];
  word64 out = mul_and_reduce_shoup(op1[start_prime_idx * degree + i],
                                    op2[op2_idx], op2_psinv[op2_idx], prime);
  if (out >= prime) out -= prime;
  to[start_prime_idx * degree + i] = out;
  STRIDED_LOOP_END;
}

void Context::ConstMultBatch(const word64 *op1, const DeviceVector &op2,
                             const DeviceVector &op2_psinv, int start_prime_idx,
                             int batch, word64 *res) const {
  assert(op2.size() == op2_psinv.size());
  assert(op2.size() == batch);
  const int block_dim = 256;
  const int grid_dim = degree__ * batch / block_dim;
  constMultBatch_<<<grid_dim, block_dim>>>(degree__, primes__.data(), op1,
                                           op2.data(), op2_psinv.data(),
                                           start_prime_idx, batch, res);
}

__device__ uint128_t4 AccumulateInModUp(const word64 *ptr, const int degree,
                                        const word64 *hat_mod_end,
                                        const int start_length,
                                        const int degree_idx,
                                        const int hat_mod_end_idx) {
  uint128_t4 accum{0};
  for (int i = 0; i < start_length; i++) {
    const uint64_t op2 = hat_mod_end[hat_mod_end_idx * start_length + i];
    uint128_t4 out;
    // cache streaming?
    // or, texture?
    uint64_t op1_x, op1_y, op1_z, op1_w;
    asm("{\n\t"
        "ld.global.v2.u64 {%0, %1}, [%2];\n\t"
        "}"
        : "=l"(op1_x), "=l"(op1_y)
        : "l"(ptr + i * degree + degree_idx));

    out.x = mult_64_64_128(op1_x, op2);
    inplace_add_128_128(out.x, accum.x);
    out.y = mult_64_64_128(op1_y, op2);
    inplace_add_128_128(out.y, accum.y);
    asm("{\n\t"
        "ld.global.v2.u64 {%0, %1}, [%2];\n\t"
        "}"
        : "=l"(op1_z), "=l"(op1_w)
        : "l"(ptr + i * degree + degree_idx + 2));
    out.z = mult_64_64_128(op1_z, op2);
    inplace_add_128_128(out.z, accum.z);
    out.w = mult_64_64_128(op1_w, op2);
    inplace_add_128_128(out.w, accum.w);
  }
  return accum;
}

// Applied loop unroll.
// Mod-up `ptr` to `to`.
// hat_mod_end[:end_length], ptr[:start_length][:degree],
// to[:start_length+end_length][:degree] ptr` is entirely overlapped with `to`.
__global__ void modUpStepTwoKernel(const word64 *ptr, const int begin_idx,
                                   const int degree, const word64 *primes,
                                   const word64 *barrett_ratios,
                                   const word64 *barrett_Ks,
                                   const word64 *hat_mod_end,
                                   const int hat_mod_end_size,
                                   const word64 start_length,
                                   const word64 end_length, word64 *to) {
  constexpr const int unroll_number = 4;
  extern __shared__ word64 s_hat_mod_end[];
  for (int i = threadIdx.x; i < hat_mod_end_size; i += blockDim.x) {
    s_hat_mod_end[i] = hat_mod_end[i];
  }
  __syncthreads();
  STRIDED_LOOP_START((degree * end_length + unroll_number - 1) / unroll_number,
                     i);
  const int degree_idx = unroll_number * (i / end_length);
  const int hat_mod_end_idx = i % end_length;
  // Leap over the overlapped region.
  const int out_prime_idx =
      hat_mod_end_idx + ((hat_mod_end_idx >= begin_idx) ? start_length : 0);
  assert(degree_idx < degree);
  uint128_t4 accum = AccumulateInModUp(ptr, degree, s_hat_mod_end, start_length,
                                       degree_idx, hat_mod_end_idx);
  const auto prime = primes[out_prime_idx];
  const auto barret_ratio = barrett_ratios[out_prime_idx];
  const auto barret_k = barrett_Ks[out_prime_idx];
  {
    uint64_t out =
        barret_reduction_128_64(accum.x, prime, barret_ratio, barret_k);
    uint64_t out2 =
        barret_reduction_128_64(accum.y, prime, barret_ratio, barret_k);
    asm("st.cs.global.v2.u64 [%0],{%1, %2};" ::"l"(to + out_prime_idx * degree +
                                                   degree_idx),
        "l"(out), "l"(out2));
  }
  {
    uint64_t out =
        barret_reduction_128_64(accum.z, prime, barret_ratio, barret_k);
    uint64_t out2 =
        barret_reduction_128_64(accum.w, prime, barret_ratio, barret_k);
    asm("st.cs.global.v2.u64 [%0],{%1, %2};" ::"l"(to + out_prime_idx * degree +
                                                   degree_idx + 2),
        "l"(out), "l"(out2));
  }
  STRIDED_LOOP_END;
}

__global__ void ModDownLengthOne(const uint64_t *poly, const int current_level,
                                 const KernelParams ring, uint64_t *out) {
  const int degree = ring.degree;
  STRIDED_LOOP_START(degree * current_level, i);
  const int prime_idx = i / degree;
  const uint64_t start_prime = ring.primes[current_level];
  const uint64_t end_prime = ring.primes[prime_idx];
  const int degree_idx = i % degree;
  if (end_prime < start_prime) {
    barret_reduction_64_64(poly[degree_idx], out[i], ring.primes[prime_idx],
                           ring.barret_ratio[prime_idx],
                           ring.barret_k[prime_idx]);
  } else {
    out[i] = poly[degree_idx];
  }
  STRIDED_LOOP_END;
}

__global__ void ModDownKernel(KernelParams ring, const word64 *ptr,
                              const word64 *hat_mod_end,
                              const int hat_mod_end_size,
                              const word64 start_length,
                              const word64 end_length, word64 *to) {
  constexpr const int unroll_number = 4;
  extern __shared__ word64 s_hat_mod_end[];
  for (int i = threadIdx.x; i < hat_mod_end_size; i += blockDim.x) {
    s_hat_mod_end[i] = hat_mod_end[i];
  }
  __syncthreads();
  STRIDED_LOOP_START(
      (ring.degree * end_length + unroll_number - 1) / unroll_number, i);
  const int degree_idx = unroll_number * (i / end_length);
  const int out_prime_idx = i % end_length;
  uint128_t4 accum = AccumulateInModUp(ptr, ring.degree, hat_mod_end,
                                       start_length, degree_idx, out_prime_idx);
  const auto prime = ring.primes[out_prime_idx];
  const auto barret_ratio = ring.barret_ratio[out_prime_idx];
  const auto barret_k = ring.barret_k[out_prime_idx];
  {
    uint64_t out =
        barret_reduction_128_64(accum.x, prime, barret_ratio, barret_k);
    uint64_t out2 =
        barret_reduction_128_64(accum.y, prime, barret_ratio, barret_k);
    asm("st.cs.global.v2.u64 [%0],{%1, %2};" ::"l"(
            to + out_prime_idx * ring.degree + degree_idx),
        "l"(out), "l"(out2));
  }
  {
    uint64_t out =
        barret_reduction_128_64(accum.z, prime, barret_ratio, barret_k);
    uint64_t out2 =
        barret_reduction_128_64(accum.w, prime, barret_ratio, barret_k);
    asm("st.cs.global.v2.u64 [%0],{%1, %2};" ::"l"(
            to + out_prime_idx * ring.degree + degree_idx + 2),
        "l"(out), "l"(out2));
  }
  STRIDED_LOOP_END;
}

__global__ void negateInplace_(size_t degree, size_t log_degree, size_t batch,
                               const uint64_t *primes, uint64_t *op) {
  STRIDED_LOOP_START(batch * degree, i);
  const int prime_idx = i >> log_degree;
  const uint64_t prime = primes[prime_idx];
  if (op[i] != 0) op[i] = prime - op[i];
  STRIDED_LOOP_END;
}

void Context::NegateInplace(word64 *op1, const int batch) const {
  const int block_dim = 256;
  const int grid_dim = degree__ * batch / block_dim;
  negateInplace_<<<grid_dim, block_dim>>>(degree__, log2(degree__), batch,
                                          primes__.data(), op1);
}

__global__ void subInplace_(size_t degree, size_t log_degree, size_t batch,
                            const uint64_t *primes, uint64_t *op1,
                            const uint64_t *op2) {
  STRIDED_LOOP_START(batch * degree, i);
  const int prime_idx = i >> log_degree;
  const uint64_t prime = primes[prime_idx];
  if (op1[i] >= op2[i]) {
    op1[i] -= op2[i];
  } else {
    op1[i] = prime - (op2[i] - op1[i]);
  }
  STRIDED_LOOP_END;
}

void Context::SubInplace(word64 *op1, const word64 *op2,
                         const int batch) const {
  const int block_dim = 256;
  const int grid_dim = degree__ * batch / block_dim;
  subInplace_<<<grid_dim, block_dim>>>(degree__, log2(degree__), batch,
                                       primes__.data(), op1, op2);
}

void Context::ModDown(DeviceVector &from_v, DeviceVector &to_v,
                      long target_chain_idx) const {
  if (target_chain_idx > param__.chain_length_)
    throw std::logic_error("Target chain index is too big.");
  to_v.resize(target_chain_idx * degree__);
  const int gap = param__.chain_length_ - target_chain_idx;
  const int start_length = param__.max_num_moduli_ - target_chain_idx;
  const int end_length = target_chain_idx;
  const int block_dim = 256;
  const int grid_dim = degree__ * end_length / block_dim;
  if (start_length == 1) {
    FromNTTInplace(from_v, end_length, start_length);
    ModDownLengthOne<<<grid_dim, block_dim>>>(
        from_v.data() + end_length * degree__, end_length, GetKernelParams(),
        to_v.data());
  } else {
    FromNTTInplace(from_v.data(), end_length, start_length);
    const int gap = param__.chain_length_ - target_chain_idx;
    const DeviceVector &hat_inverse_vec = hat_inverse_vec_moddown__.at(gap);
    const DeviceVector &hat_inverse_vec_psinv =
        hat_inverse_vec_shoup_moddown__.at(gap);
    ConstMultBatch(from_v.data(), hat_inverse_vec, hat_inverse_vec_psinv,
                   end_length, start_length, from_v.data());
    auto ptr = from_v.data() + degree__ * end_length;
    const auto &prod_q_i_mod_q_j = prod_q_i_mod_q_j_moddown__[gap];
    ModDownKernel<<<grid_dim, block_dim,
                    prod_q_i_mod_q_j.size() * sizeof(word64)>>>(
        GetKernelParams(), ptr, prod_q_i_mod_q_j.data(),
        start_length * end_length, start_length, end_length, to_v.data());
  }
  const auto &prod_inv = prod_inv_moddown__.at(gap);
  const auto &prod_inv_psinv = prod_inv_shoup_moddown__.at(gap);
  // fuse four functions: NTT, sub, negate, and const-mult.
  if (is_moddown_fused)
    ToNTTInplaceFused(to_v, from_v, prod_inv, prod_inv_psinv);
  else {
    ToNTTInplace(to_v.data(), 0, end_length);
    SubInplace(to_v.data(), from_v.data(), end_length);
    NegateInplace(to_v.data(), end_length);
    ConstMultBatch(to_v.data(), prod_inv, prod_inv_psinv, 0, end_length,
                   to_v.data());
  }
}

void Context::ToNTTInplace(word64 *op, int start_prime_idx, int batch) const {
  dim3 gridDim(2048);
  dim3 blockDim(256);
  const int per_thread_ntt_size = 8;
  const int first_stage_radix_size = 256;
  const int second_radix_size = degree__ / first_stage_radix_size;
  const int pad = 4;
  const int per_thread_storage =
      blockDim.x * per_thread_ntt_size * sizeof(word64);
  Ntt8PointPerThreadPhase1<<<gridDim, (first_stage_radix_size / 8) * pad,
                             (first_stage_radix_size + pad + 1) * pad *
                                 sizeof(uint64_t)>>>(
      op, 1, batch, degree__, start_prime_idx, pad,
      first_stage_radix_size / per_thread_ntt_size, power_of_roots__.data(),
      power_of_roots_shoup__.data(), primes__.data());
  Ntt8PointPerThreadPhase2<<<gridDim, blockDim.x, per_thread_storage>>>(
      op, first_stage_radix_size, batch, degree__, start_prime_idx,
      second_radix_size / per_thread_ntt_size, power_of_roots__.data(),
      power_of_roots_shoup__.data(), primes__.data());
  CudaCheckError();
}

void Context::ModUpMatMul(const word64 *ptr, int beta_idx, word64 *to) const {
  const int unroll_factor = 4;
  const int start_length = alpha__;
  const int begin_idx = beta_idx * alpha__;
  const int end_length = num_moduli_after_modup__ - alpha__;
  long grid_dim{degree__ * end_length / 256 / unroll_factor};
  int block_dim{256};
  const auto &prod_q_i_mod_q_j = prod_q_i_mod_q_j__.at(beta_idx);
  modUpStepTwoKernel<<<grid_dim, block_dim,
                       prod_q_i_mod_q_j.size() * sizeof(word64)>>>(
      ptr, begin_idx, degree__, primes__.data(), barret_ratio__.data(),
      barret_k__.data(), prod_q_i_mod_q_j.data(), prod_q_i_mod_q_j.size(),
      start_length, end_length, to);
}

void Context::ModUpBatchImpl(const DeviceVector &from, DeviceVector &to,
                             int beta) const {
  if (alpha__ == 1) {
    DeviceVector from_after_intt = FromNTT(from);  // no scaling when alpha = 1
    for (int idx = 0; idx < beta; idx++) {
      const int begin_idx = idx * alpha__;
      word64 *target = to.data() + idx * num_moduli_after_modup__ * degree__;
      int end_length = num_moduli_after_modup__ - alpha__;
      ModUpLengthIsOne(from_after_intt.data() + degree__ * idx,
                       from.data() + degree__ * idx, begin_idx, end_length + 1,
                       target);
      ToNTTInplaceExceptSomeRange(target, 0, num_moduli_after_modup__,
                                  begin_idx, alpha__);
    }
  } else {
    // Apply iNTT and multiplies \hat{q}_i (fast base conversion)
    DeviceVector temp = FromNTT(from, hat_inverse_vec_batched__,
                                hat_inverse_vec_shoup_batched__);
    for (int idx = 0; idx < beta; idx++) {
      const int begin_idx = idx * alpha__;
      word64 *target = to.data() + idx * num_moduli_after_modup__ * degree__;
      ModUpMatMul(temp.data() + degree__ * begin_idx, idx, target);
      ToNTTInplaceExceptSomeRange(target, 0, num_moduli_after_modup__,
                                  begin_idx, alpha__);
      cudaMemcpyAsync(
          target + begin_idx * degree__, from.data() + idx * alpha__ * degree__,
          8 * alpha__ * degree__, cudaMemcpyDeviceToDevice, cudaStreamLegacy);
    }
  }
}

void Context::ModUpLengthIsOne(const word64 *ptr_after_intt,
                               const word64 *ptr_before_intt, int begin_idx,
                               int end_length, word64 *to) const {
  int block_dim{256};
  long grid_dim{degree__ * end_length / block_dim};
  modUpStepTwoSimple<<<grid_dim, block_dim>>>(
      ptr_after_intt, ptr_before_intt, begin_idx, degree__, primes__.data(),
      barret_ratio__.data(), barret_k__.data(), end_length, to);
}

void Context::ToNTTInplaceExceptSomeRange(word64 *op, int start_prime_idx,
                                          int batch, int excluded_range_start,
                                          int excluded_range_size) const {
  const int excluded_range_end = excluded_range_start + excluded_range_size;
  if (excluded_range_start < start_prime_idx ||
      excluded_range_end > (start_prime_idx + batch)) {
    throw "Excluded range in NTT is invalid.";
  }
  dim3 grid(2048);
  dim3 block(256);
  const int per_thread_ntt_size = 8;
  const int first_stage_radix_size = 256;
  const int second_radix_size = degree__ / first_stage_radix_size;
  const int pad = 4;
  const int per_thread_storage = block.x * per_thread_ntt_size * sizeof(word64);
  Ntt8PointPerThreadPhase1ExcludeSomeRange<<<
      grid, (first_stage_radix_size / 8) * pad,
      (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_t)>>>(
      op, 1, batch, degree__, start_prime_idx, excluded_range_start,
      excluded_range_end, pad, first_stage_radix_size / per_thread_ntt_size,
      power_of_roots__.data(), power_of_roots_shoup__.data(), primes__.data());
  Ntt8PointPerThreadPhase2ExcludeSomeRange<<<grid, block.x,
                                             per_thread_storage>>>(
      op, first_stage_radix_size, batch, degree__, start_prime_idx,
      excluded_range_start, excluded_range_end,
      second_radix_size / per_thread_ntt_size, power_of_roots__.data(),
      power_of_roots_shoup__.data(), primes__.data());
  CudaCheckError();
}

void Context::ToNTTInplaceFused(DeviceVector &op1, const DeviceVector &op2,
                                const DeviceVector &epilogue,
                                const DeviceVector &epilogue_) const {
  dim3 gridDim(2048);
  dim3 blockDim(256);
  const int per_thread_ntt_size = 8;
  const int first_stage_radix_size = 256;
  const int second_radix_size = degree__ / first_stage_radix_size;
  const int pad = 4;
  const int per_thread_storage =
      blockDim.x * per_thread_ntt_size * sizeof(word64);
  const int start_prime_idx = 0;
  const int batch = op1.size() / degree__;
  Ntt8PointPerThreadPhase1<<<gridDim, (first_stage_radix_size / 8) * pad,
                             (first_stage_radix_size + pad + 1) * pad *
                                 sizeof(uint64_t)>>>(
      op1.data(), 1, batch, degree__, start_prime_idx, pad,
      first_stage_radix_size / per_thread_ntt_size, power_of_roots__.data(),
      power_of_roots_shoup__.data(), primes__.data());
  Ntt8PointPerThreadPhase2FusedWithSubNegateConstMult<<<gridDim, blockDim.x,
                                                        per_thread_storage>>>(
      op1.data(), first_stage_radix_size, batch, degree__, start_prime_idx,
      second_radix_size / per_thread_ntt_size, power_of_roots__.data(),
      power_of_roots_shoup__.data(), primes__.data(), op2.data(),
      epilogue.data(), epilogue_.data());
  CudaCheckError();
}

__global__ void sumAndReduceFused(const word64 *modup_out, const int degree,
                                  const int length, const int batch,
                                  const word64 *eval_ax, const word64 *eval_bx,
                                  const word64 *primes, const word64 *barret_ks,
                                  const word64 *barret_ratios, word64 *dst_ax,
                                  word64 *dst_bx) {
  STRIDED_LOOP_START(degree * length, i);
  const int stride_between_batch = degree * length;
  uint128_t accum_ax{0, 0};
  uint128_t accum_bx{0, 0};
  for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
    const int idx = i + stride_between_batch * batch_idx;
    const word64 op1 = modup_out[idx];
    const word64 op2_ax = eval_ax[idx];
    const auto mul_ax = mult_64_64_128(op1, op2_ax);
    accum_ax += mul_ax;
    const word64 op2_bx = eval_bx[idx];
    const auto mul_bx = mult_64_64_128(op1, op2_bx);
    accum_bx += mul_bx;
  }
  const int prime_idx = i / degree;
  const auto prime = primes[prime_idx];
  const auto barret_ratio = barret_ratios[prime_idx];
  const auto barret_k = barret_ks[prime_idx];
  const auto res_ax =
      barret_reduction_128_64(accum_ax, prime, barret_ratio, barret_k);
  const auto res_bx =
      barret_reduction_128_64(accum_bx, prime, barret_ratio, barret_k);
  dst_ax[i] = res_ax;
  dst_bx[i] = res_bx;
  STRIDED_LOOP_END;
}

template <bool Accum>
__global__ void mult_(const word64 *modup_out, const word64 *eval_poly_ax,
                      const word64 *eval_poly_bx, const int degree,
                      const int length, uint128_t *accum_ptr_ax,
                      uint128_t *accum_ptr_bx) {
  STRIDED_LOOP_START(degree * length, i);
  const word64 op1 = modup_out[i];
  const word64 op2_ax = eval_poly_ax[i];
  const word64 op2_bx = eval_poly_bx[i];
  const auto mul_ax = mult_64_64_128(op1, op2_ax);
  const auto mul_bx = mult_64_64_128(op1, op2_bx);
  if (Accum) {
    accum_ptr_ax[i] += mul_ax;
    accum_ptr_bx[i] += mul_bx;
  } else {
    accum_ptr_ax[i] = mul_ax;
    accum_ptr_bx[i] = mul_bx;
  }
  STRIDED_LOOP_END;
}

__global__ void Reduce(const uint128_t *accum, const int degree,
                       const int length, const word64 *primes,
                       const word64 *barret_ks, const word64 *barret_ratios,
                       word64 *res) {
  STRIDED_LOOP_START(degree * length, i);
  const int prime_idx = i / degree;
  const auto prime = primes[prime_idx];
  const auto barret_ratio = barret_ratios[prime_idx];
  const auto barret_k = barret_ks[prime_idx];
  const auto res_ax =
      barret_reduction_128_64(accum[i], prime, barret_ratio, barret_k);
  res[i] = res_ax;
  STRIDED_LOOP_END;
}

void Context::KeySwitch(const DeviceVector &modup_out, const EvaluationKey &evk,
                        DeviceVector &sum_ax, DeviceVector &sum_bx) const {
  assert(modup_out.size() > 0 && modup_out.size() % degree__ == 0);
  const int total_length = modup_out.size() / degree__;
  assert(total_length % param__.max_num_moduli_ == 0);
  const int beta = total_length / param__.max_num_moduli_;
  const int length = param__.max_num_moduli_;
  const int gridDim = 1024;
  const int blockDim = 256;
  const int size_after_reduced = param__.max_num_moduli_ * degree__;
  sum_ax.resize(size_after_reduced);
  sum_bx.resize(size_after_reduced);
  const auto &eval_ax = evk.getAxDevice();
  const auto &eval_bx = evk.getBxDevice();
  if (is_keyswitch_fused) {
    sumAndReduceFused<<<gridDim, blockDim>>>(
        modup_out.data(), degree__, length, beta, eval_ax.data(),
        eval_bx.data(), primes__.data(), barret_k__.data(),
        barret_ratio__.data(), sum_ax.data(), sum_bx.data());
  } else {
    const int quad_word_size_byte = sizeof(uint128_t);
    DeviceBuffer accum_ax(modup_out.size() * quad_word_size_byte);
    DeviceBuffer accum_bx(modup_out.size() * quad_word_size_byte);
    auto accum_ax_ptr = (uint128_t *)accum_ax.data();
    auto accum_bx_ptr = (uint128_t *)accum_bx.data();
    mult_<false><<<gridDim, blockDim>>>(modup_out.data(), eval_ax.data(),
                                        eval_bx.data(), degree__, length,
                                        accum_ax_ptr, accum_bx_ptr);
    for (int i = 1; i < beta; i++) {
      const auto d2_ptr = modup_out.data() + i * degree__ * length;
      const auto ax_ptr = eval_ax.data() + i * degree__ * length;
      const auto bx_ptr = eval_bx.data() + i * degree__ * length;
      mult_<true><<<gridDim, blockDim>>>(d2_ptr, ax_ptr, bx_ptr, degree__,
                                         length, accum_ax_ptr, accum_bx_ptr);
    }
    Reduce<<<gridDim, blockDim>>>(accum_ax_ptr, degree__, length,
                                  primes__.data(), barret_k__.data(),
                                  barret_ratio__.data(), sum_ax.data());
    Reduce<<<gridDim, blockDim>>>(accum_bx_ptr, degree__, length,
                                  primes__.data(), barret_k__.data(),
                                  barret_ratio__.data(), sum_bx.data());
  }
}

__global__ void hadamardMultAndAddBatch_(
    const KernelParams ring, const word64 **ax_addr, const word64 **bx_addr,
    const word64 **mx_addr, const int fold_size, const size_t size,
    const int log_degree, word64 *out_ax, word64 *out_bx) {
  STRIDED_LOOP_START(size, idx);
  const int prime_idx = idx >> log_degree;
  const uint64_t prime = ring.primes[prime_idx];
  uint128_t sum_ax = {0};
  uint128_t sum_bx = {0};
  for (int fold_idx = 0; fold_idx < fold_size; fold_idx++) {
    const word64 *ax = ax_addr[fold_idx];
    const word64 *bx = bx_addr[fold_idx];
    const word64 *mx = mx_addr[fold_idx];
    const word64 mx_element = mx[idx];
    sum_ax += mult_64_64_128(ax[idx], mx_element);
    sum_bx += mult_64_64_128(bx[idx], mx_element);
  }
  out_ax[idx] = barret_reduction_128_64(
      sum_ax, prime, ring.barret_ratio[prime_idx], ring.barret_k[prime_idx]);
  out_bx[idx] = barret_reduction_128_64(
      sum_bx, prime, ring.barret_ratio[prime_idx], ring.barret_k[prime_idx]);
  if (out_ax[idx] > prime) out_ax[idx] -= prime;
  if (out_bx[idx] > prime) out_bx[idx] -= prime;
  STRIDED_LOOP_END;
}

void Context::hadamardMultAndAddBatch(const std::vector<const word64 *> ax_addr,
                                      const std::vector<const word64 *> bx_addr,
                                      const std::vector<const word64 *> mx_addr,
                                      const int num_primes,
                                      DeviceVector &out_ax,
                                      DeviceVector &out_bx) const {
  assert(ax_addr.size() == bx_addr.size() && ax_addr.size() == mx_addr.size());
  if (out_ax.size() != (size_t)num_primes * degree__ ||
      out_bx.size() != (size_t)num_primes * degree__)
    throw std::logic_error("Output has no proper size");
  const int fold_size = ax_addr.size();
  const int each_operand_size = num_primes * degree__;
  size_t addr_buffer_size = fold_size * sizeof(word64 *);
  const DeviceBuffer d_ax_addr(ax_addr.data(), addr_buffer_size,
                               cudaStreamLegacy);
  const DeviceBuffer d_bx_addr(bx_addr.data(), addr_buffer_size,
                               cudaStreamLegacy);
  const DeviceBuffer d_mx_addr(mx_addr.data(), addr_buffer_size,
                               cudaStreamLegacy);
  const int block_dim = 256;
  const int grid_dim = each_operand_size / block_dim;
  hadamardMultAndAddBatch_<<<grid_dim, block_dim>>>(
      GetKernelParams(), (const word64 **)d_ax_addr.data(),
      (const word64 **)d_bx_addr.data(), (const word64 **)d_mx_addr.data(),
      fold_size, each_operand_size, param__.log_degree_, out_ax.data(),
      out_bx.data());
}

__global__ void hadamardMultFused_(size_t degree, size_t log_degree,
                              size_t num_primes, const uint64_t* primes,
                              const uint64_t* barret_ratio,
                              const uint64_t* barret_k, const uint64_t* op1,
                              const uint64_t* op2, const uint64_t* mx,
                              uint64_t* op1_out, uint64_t* op2_out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int prime_idx = i >> log_degree;
  const uint64_t prime = primes[prime_idx];
  uint64_t mx_element = mx[i];
  uint128_t out_op1 = mult_64_64_128(op1[i], mx_element);
  uint128_t out_op2 = mult_64_64_128(op2[i], mx_element);
  op1_out[i] = barret_reduction_128_64(out_op1, prime, barret_ratio[prime_idx],
                                   barret_k[prime_idx]);
  op2_out[i] = barret_reduction_128_64(out_op2, prime, barret_ratio[prime_idx],
                                   barret_k[prime_idx]);
}


void Context::PMult(const Ciphertext &ct, const Plaintext &pt,
                    Ciphertext &out) const {
  const auto &op1 = ct.getAxDevice();
  const auto &op2 = ct.getBxDevice();
  const auto &mx = pt.getMxDevice();
  auto &op1_out = out.getAxDevice();
  auto &op2_out = out.getBxDevice();
  op1_out.resize(op1.size());
  op2_out.resize(op1.size());
  int num_primes = op1.size() / degree__;
  hadamardMultFused_<<<degree__ * num_primes / 256, 256>>>(
      degree__, param__.log_degree_, num_primes, primes__.data(), barret_ratio__.data(),
      barret_k__.data(), op1.data(), op2.data(), mx.data(), op1_out.data(),
      op2_out.data());
}

__global__ void add_(const KernelParams params,
                     const int batch, const word64* op1, const word64* op2,
                     word64* op3) {
  STRIDED_LOOP_START(batch * params.degree, i);
  const int prime_idx = i >> params.log_degree;
  const uint64_t prime = params.primes[prime_idx];
  op3[i] = op1[i] + op2[i];
  if (prime - op3[i] >> 63) op3[i] -= prime;
  STRIDED_LOOP_END;
}

// Assume ct1 and ct2 has the same size
void Context::Add(const Ciphertext &ct1, const Ciphertext &ct2,
                  Ciphertext &out) const {
  const auto &op1_ax = ct1.getAxDevice();
  const auto &op2_ax = ct2.getAxDevice();
  const auto &op1_bx = ct1.getBxDevice();
  const auto &op2_bx = ct2.getBxDevice();
  auto &out_ax = out.getAxDevice();
  auto &out_bx = out.getBxDevice();
  out_ax.resize(op1_ax.size());
  out_bx.resize(op1_ax.size());
  const int length = op1_ax.size() / degree__;
  if (op1_ax.size() != op2_ax.size() || op1_ax.size() != out_ax.size())
    throw std::logic_error("Size does not match");
  int gridDim_ = 2048;
  int blockDim_ = 256;
  add_<<<gridDim_, blockDim_>>>(GetKernelParams(), length, op1_ax.data(),
                                op2_ax.data(), out_ax.data());
  add_<<<gridDim_, blockDim_>>>(GetKernelParams(), length, op1_bx.data(),
                                op2_bx.data(), out_bx.data());
}

void Context::EnableMemoryPool() {
  if (pool__ == nullptr) {
    pool__ = std::make_shared<MemoryPool>(param__);
    pool__->UseMemoryPool(true);
  } else {
    throw std::logic_error("Enable memory pool twice?");
  }
}