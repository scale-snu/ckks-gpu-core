/* Copyright (c) by CryptoLab Inc.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#pragma once

#include <cstdint>

#include "Define.h"
#include "DeviceVector.h"

namespace ckks {
class Parameter {
 public:
  Parameter(int log_degree, int level, int dnum,
            const std::vector<word64>& primes)
      : log_degree_{log_degree},
        degree_{1 << log_degree_},
        level_{level},
        dnum_{dnum},
        alpha_{(level_ + 1) / dnum_},
        max_num_moduli_{level_ + 1 + alpha_},
        chain_length_{level_ + 1},
        num_special_moduli_{alpha_},
        primes_{primes} {
    if ((level_ + 1) % dnum_ != 0) throw std::logic_error("wrong dnum value.");
    if (primes.size() != (size_t)chain_length_ + num_special_moduli_)
      throw std::logic_error("the size of the primes passed is wrong");
  };

  const int log_degree_;
  const int degree_;
  const int level_;
  const int dnum_;
  const int alpha_;
  const int max_num_moduli_;
  const int chain_length_;
  const int num_special_moduli_;
  HostVector primes_;
};

  }  // namespace ckks