/* Copyright (c) by CryptoLab Inc.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#pragma once

#include <cstdint>
#include <string>
#include <chrono>
#include <iostream>
namespace ckks {

using word64 = uint64_t;
using word128 = __uint128_t;

void CudaNvtxStart(std::string msg = "");
void CudaNvtxStop();
void CudaHostSync();

}  // namespace ckks