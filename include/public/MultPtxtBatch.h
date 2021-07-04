/* Copyright (c) by CryptoLab Inc. and Seoul National University R&DB Foundation.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#pragma once

#include <thrust/host_vector.h>

#include <vector>

#include "Ciphertext.h"
#include "Context.h"

namespace ckks {

class MultPtxtBatch {
  using AddrVector = std::vector<const word64*>;

 public:
  MultPtxtBatch(const Context* context) : context_{context} {}

  // register the operands to be multiplied and summed up.
  void push(const Ciphertext& op1, const Plaintext& op2);
  void flush(Ciphertext& out);

 private:
  const Context* context_;
  size_t required_size = 0;
  AddrVector ax__;
  AddrVector bx__;
  AddrVector mx__;
};

}  // namespace ckks