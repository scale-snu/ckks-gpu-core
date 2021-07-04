/* Copyright (c) by CryptoLab Inc. and Seoul National University R&DB Foundation.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#pragma once

#include <iostream>
#include <memory>
#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include "Parameter.h"
namespace ckks {
// After the creation of an MemoryPool object, all memory allocations on the
// current device uses binning_memory_resouce.
class MemoryPool {
  using DefaultUpstream = rmm::mr::cuda_memory_resource;
  using MemoryPoolBase = rmm::mr::binning_memory_resource<DefaultUpstream>;
  using MemoryBin = rmm::mr::fixed_size_memory_resource<DefaultUpstream>;
  constexpr static int preallocation_size = 1;

 public:
  MemoryPool(const Parameter& params) : device_pool__(&base__) {
    // Hueristically add bins to save memory and speed-up bootstrapping.
    const int degree = params.degree_;
    addBin(256);
    addBin(1024);
    addBin(sizeof(word64) * degree);
    addBin(sizeof(word64) * degree * params.alpha_ * params.dnum_);
    addBin(sizeof(word64) * degree *
           (params.alpha_ * params.dnum_ + params.num_special_moduli_));
    addBin(sizeof(word64) * degree *
           (params.alpha_ * params.dnum_ + params.num_special_moduli_) *
           (params.dnum_));
    addBin(2 * sizeof(word64) * degree *
           (params.alpha_ * params.dnum_ + params.num_special_moduli_) *
           (params.dnum_));
  }

  void UseMemoryPool(bool use) {
    if (!use)
      rmm::mr::set_current_device_resource(&base__);
    else
      rmm::mr::set_current_device_resource(&device_pool__);
  }

  ~MemoryPool() {
    // reset to cuda_device_resource
    rmm::mr::set_current_device_resource(nullptr);
  }

 private:
  void addBin(size_t size) {
    auto bin = std::make_shared<MemoryBin>(&base__, size, preallocation_size);
    bin__.push_back(bin);
    device_pool__.add_bin(size, bin.get());
  }

  DefaultUpstream base__;
  MemoryPoolBase device_pool__;
  std::vector<std::shared_ptr<MemoryBin>> bin__;
};

}  // namespace ckks