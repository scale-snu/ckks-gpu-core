/* Copyright (c) by CryptoLab Inc. and Seoul National University R&DB Foundation.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#include "Ciphertext.h"
#include "DeviceVector.h"
#include "MultPtxtBatch.h"

using namespace ckks;
using namespace std;

void MultPtxtBatch::push(const Ciphertext& op1, const Plaintext& op2) {
  const auto& mx = op2.getMxDevice();
  auto& ax = op1.getAxDevice();
  auto& bx = op1.getBxDevice();
  if (ax.size() != bx.size() || ax.size() != mx.size())
    throw std::logic_error("Size does not match");
  if (ax__.size() == 0) {
    required_size = ax.size();
  }
  ax__.push_back(ax.data());
  bx__.push_back(bx.data());
  mx__.push_back(mx.data());
}

void MultPtxtBatch::flush(Ciphertext& out) {
  auto& out_ax = out.getAxDevice();
  auto& out_bx = out.getBxDevice();
  if (out_ax.size() != required_size) out_ax.resize(required_size);
  if (out_bx.size() != required_size) out_bx.resize(required_size);
  const int level = required_size / context_->GetDegree();
  context_->hadamardMultAndAddBatch(ax__, bx__, mx__, level, out_ax, out_bx);
}