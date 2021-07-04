/* Copyright (c) by CryptoLab Inc.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#pragma once

#include <tuple>

#include "Define.h"
#include "DeviceVector.h"
#include "Parameter.h"

namespace ckks {

class EvaluationKey {
 public:
  EvaluationKey() = default;

  const DeviceVector& getAxDevice() const { return ax__; }
  const DeviceVector& getBxDevice() const { return bx__; }
  DeviceVector& getAxDevice() { return ax__; }
  DeviceVector& getBxDevice() { return bx__; }

 private:
  DeviceVector ax__;
  DeviceVector bx__;
};

}  // namespace ckks