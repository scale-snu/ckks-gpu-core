/* Copyright (c) by CryptoLab Inc. and Seoul National University R&DB Foundation.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */


#include <random>

#include "gtest/gtest.h"
#include "public/Ciphertext.h"
#include "public/Context.h"
#include "public/Define.h"
#include "public/EvaluationKey.h"
#include "public/MultPtxtBatch.h"
#include "public/Parameter.h"
#include "public/Test.h"

class FusionTest : public ckks::Test,
                   public ::testing::TestWithParam<ckks::Parameter> {
 protected:
  FusionTest() : ckks::Test(GetParam()){};

  void COMPARE(const ckks::DeviceVector& ref,
               const ckks::DeviceVector& out) const {
    ASSERT_EQ(ckks::HostVector(ref), ckks::HostVector(out));
  }
};

TEST_P(FusionTest, ModUp) {
  for (int beta = 0;
       beta < std::min(2 /* else it takes too long*/, param.dnum_); beta++) {
    const int num_moduli = (beta + 1) * param.alpha_;
    // assumed rns-decomposed already.
    ckks::DeviceVector from{GetRandomPolyRNS(num_moduli)};
    context.is_modup_batched = false;
    auto ref = context.ModUp(from);
    context.is_modup_batched = true;
    auto batched = context.ModUp(from);
    COMPARE(ref, batched);
  }
}

TEST_P(FusionTest, ModDown) {
  auto from = GetRandomPolyRNS(param.max_num_moduli_);
  ckks::DeviceVector from_fused(from);
  ckks::DeviceVector to;
  ckks::DeviceVector to_fused;
  const int target_chain_length = 2;
  context.is_moddown_fused = true;
  context.ModDown(from_fused, to_fused, target_chain_length);
  context.is_moddown_fused = false;
  context.ModDown(from, to, target_chain_length);
  COMPARE(to, to_fused);
}

TEST_P(FusionTest, KeySwitch) {
  int beta = 2;
  auto key = GetRandomKey();
  auto in = GetRandomPolyAfterModUp(beta);
  ckks::DeviceVector ax, bx;
  ckks::DeviceVector ax_ref, bx_ref;
  context.is_keyswitch_fused = false;
  context.KeySwitch(in, key, ax_ref, bx_ref);
  context.is_keyswitch_fused = true;
  context.KeySwitch(in, key, ax, bx);
  COMPARE(ax, ax_ref);
  COMPARE(bx, bx_ref);
}

TEST_P(FusionTest, PtxtCtxtBatching) {
  using namespace ckks;
  int batch_size = 3;
  vector<Ciphertext> op1(batch_size);
  vector<Plaintext> op2(batch_size);
  // setup
  for (int i = 0; i < batch_size; i++) {
    op1[i] = GetRandomCiphertext();
    op2[i] = GetRandomPlaintext();
  }
  // reference
  Ciphertext accum, out;
  context.PMult(op1[0], op2[0], accum);
  for (int i = 1; i < batch_size; i++) {
    context.PMult(op1[i], op2[i], out);
    context.Add(accum, out, accum);
  }
  // with batching
  Ciphertext accum_new;
  {
    MultPtxtBatch batcher(&context);
    for (int i = 0; i < batch_size; i++) {
      batcher.push(op1[i], op2[i]);
    }
    batcher.flush(accum_new);
  }
  COMPARE(accum_new.getAxDevice(), accum.getAxDevice());
  COMPARE(accum_new.getBxDevice(), accum.getBxDevice());
}

INSTANTIATE_TEST_SUITE_P(Params, FusionTest,
                         ::testing::Values(PARAM_LARGE_DNUM, PARAM_SMALL_DNUM));