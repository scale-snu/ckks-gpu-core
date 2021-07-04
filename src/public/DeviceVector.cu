/* Copyright (c) by CryptoLab Inc. and Seoul National University R&DB Foundation.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include "DeviceVector.h"

using namespace ckks;

void DeviceVector::append(const DeviceVector& out) {
  size_t old_size = size();
  resize(size() + out.size());
  cudaMemcpyAsync(data() + old_size, out.data(), out.size() * sizeof(Dtype),
                  cudaMemcpyDefault, stream_);
}