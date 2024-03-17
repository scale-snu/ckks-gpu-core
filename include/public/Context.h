/* Copyright (c) by CryptoLab Inc.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#pragma once

#include <complex>
#include <tuple>

#include "Define.h"
#include "DeviceVector.h"
#include "Parameter.h"
namespace ckks {

struct KernelParams {
  const int degree;
  const word64* primes;
  const word64* barret_ratio;
  const word64* barret_k;
  const int log_degree;
};

class Ciphertext;
class Plaintext;
class EvaluationKey;
class MemoryPool;

class Context {
  friend class MultPtxtBatch;

 public:
  Context(const Parameter& param);
  void Encode(uint64_t *out, std::complex<double> *mvec, const int slot) const;
  void Decode(std::complex<double> *out, uint64_t *v, const int slot) const;
  void KeySwitch(const DeviceVector& modup_out, const EvaluationKey& evk,
                 DeviceVector& sum_ax, DeviceVector& sum_bx) const;
  void PMult(const Ciphertext&, const Plaintext&, Ciphertext&) const;
  void Add(const Ciphertext&, const Ciphertext&, Ciphertext&) const;
  DeviceVector ModUp(const DeviceVector& in) const;
  void EnableMemoryPool();
  auto GetDegree() const { return degree__; }
  void ModDown(DeviceVector& from, DeviceVector& to,
               long target_chain_idx) const;
  bool is_modup_batched = true;
  bool is_moddown_fused = true;
  bool is_keyswitch_fused = true;

 private:
  DeviceVector FromNTT(const DeviceVector& from) const;
  // scales with scaling_constants after iNTT
  DeviceVector FromNTT(const DeviceVector& from,
                       const DeviceVector& scaling_constants,
                       const DeviceVector& scaling_constants_shoup) const;
  void ModUpImpl(const word64* from, word64* to, int idx) const;
  void ModUpBatchImpl(const DeviceVector& from, DeviceVector& to,
                      int beta) const;
  void ModUpLengthIsOne(const word64* ptr_after_intt,
                        const word64* ptr_before_intt, int begin_idx,
                        int end_length, word64* to) const;
  void ToNTTInplaceExceptSomeRange(word64* base_addr, int start_prime_idx,
                                   int batch, int excluded_range_start,
                                   int excluded_range_size) const;
  void FromNTTInplace(DeviceVector& op1, int start_prime_idx, int batch) const {
    FromNTTInplace(op1.data(), start_prime_idx, batch);
  }
  void FromNTTInplace(word64* op1, int start_prime_idx, int batch) const;
  void ToNTTInplace(word64* op1, int start_prime_idx, int batch) const;
  void ToNTTInplaceFused(DeviceVector& op1, const DeviceVector& op2,
                         const DeviceVector& epilogue,
                         const DeviceVector& epilogue_) const;
  void SubInplace(word64* op1, const word64* op2, const int batch) const;
  void NegateInplace(word64* op1, const int batch) const;
  void ConstMultBatch(const word64* op1, const DeviceVector& op2,
                      const DeviceVector& op2_psinv, int start_prime_idx,
                      int batch, word64* res) const;
  void ModUpMatMul(const word64* ptr, int beta_idx, word64* to) const;
  void hadamardMultAndAddBatch(const std::vector<const word64*> ax_addr,
                               const std::vector<const word64*> bx_addr,
                               const std::vector<const word64*> mx_addr,
                               const int num_primes, DeviceVector& out_ax,
                               DeviceVector& out_bx) const;
  auto GetKernelParams() const {
    return KernelParams{degree__, primes__.data(), barret_ratio__.data(),
                        barret_k__.data(), param__.log_degree_};
  }
  void GenModUpParams();
  void GenModDownParams();
  void GenEncodeParams();
  void fftSpecial(std::complex<double> *vals, const long size) const;
  void fftSpecialInv(std::complex<double> *vals, const long size) const;
  void arrayBitReverse(std::complex<double> *vals, const long size) const;

  std::shared_ptr<MemoryPool> pool__;
  int degree__;
  int num_moduli_after_modup__;
  int alpha__;
  Parameter param__;
  DeviceVector primes__;
  DeviceVector barret_ratio__;
  DeviceVector barret_k__;
  DeviceVector power_of_roots__;
  DeviceVector power_of_roots_shoup__;
  DeviceVector inverse_power_of_roots_div_two__;
  DeviceVector inverse_scaled_power_of_roots_div_two__;
  // for modup
  // {prod q_i}_{n * alpha <= i < (n+1) * alpha)} mod q_j
  // for j not in [n * alpha, n * alpha + alpha) for n in [0, dnum)
  std::vector<DeviceVector> prod_q_i_mod_q_j__;
  // prod q_i mod q_j for i in [n * alpha, (n+1) * alpha) && i != j
  std::vector<DeviceVector> hat_inverse_vec__;
  std::vector<DeviceVector> hat_inverse_vec_shoup__;
  DeviceVector hat_inverse_vec_batched__;
  DeviceVector hat_inverse_vec_shoup_batched__;
  // for moddown
  std::vector<DeviceVector> hat_inverse_vec_moddown__;
  std::vector<DeviceVector> hat_inverse_vec_shoup_moddown__;
  std::vector<DeviceVector> prod_q_i_mod_q_j_moddown__;
  std::vector<DeviceVector> prod_inv_moddown__;
  std::vector<DeviceVector> prod_inv_shoup_moddown__;

  // For en/decode
  uint64_t *rotGroup;
  std::complex<double> *ksiPows;
};

}  // namespace ckks