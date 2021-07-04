/* Copyright (c) by CryptoLab Inc. and Seoul National University R&DB Foundation.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include "Define.h"

namespace ckks {
using HostVector = thrust::host_vector<word64>;
using DeviceBuffer = rmm::device_buffer;

// A wrapper for device_vector.
class DeviceVector : public rmm::device_uvector<word64> {
  using Dtype = word64;
  using Base = rmm::device_uvector<Dtype>;

 public:
  // A constructor without initilization.
  explicit DeviceVector(int size = 0) : Base(size, cudaStreamLegacy) {}

  // A constructor with initilization.
  explicit DeviceVector(const DeviceVector& ref)
      : Base(ref, cudaStreamLegacy) {}
  DeviceVector(DeviceVector&& other) : Base(std::move(other)) {}

  DeviceVector& operator=(DeviceVector&& other) {
    Base::operator=(std::move(other));
    return *this;
  }

  explicit DeviceVector(const HostVector& ref)
      : Base(ref.size(), cudaStreamLegacy) {
    cudaMemcpyAsync(data(), ref.data(), ref.size() * sizeof(Dtype),
                    cudaMemcpyHostToDevice, stream_);
  }

  operator HostVector() const {
    HostVector host(size());
    cudaMemcpyAsync(host.data(), data(), size() * sizeof(Dtype),
                    cudaMemcpyDeviceToHost, stream_);
    return host;
  }

  void resize(int size) { Base::resize(size, stream_); }

  bool operator==(const DeviceVector& other) const{
    return HostVector(*this) == HostVector(other);
  }

  void append(const DeviceVector& out);

 private:
  const cudaStream_t stream_ = cudaStreamLegacy;
  };

    }  // namespace ckks