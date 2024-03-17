/* Copyright (c) by CryptoLab Inc. and Seoul National University R&DB Foundation.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */


#include <complex>
#include <random>

#include "gtest/gtest.h"
#include "public/Ciphertext.h"
#include "public/Context.h"
#include "public/Define.h"
#include "public/EvaluationKey.h"
#include "public/MultPtxtBatch.h"
#include "public/Parameter.h"
#include "public/Test.h"

class E2ETest : public ckks::Test,
                   public ::testing::TestWithParam<ckks::Parameter> {
 protected:
  E2ETest() : ckks::Test(GetParam()){};

  void COMPARE(const ckks::DeviceVector& ref,
               const ckks::DeviceVector& out) const {
    ASSERT_EQ(ckks::HostVector(ref), ckks::HostVector(out));
  }

  void COMPARE_APPROXIMATE(std::complex<double> *ref,
                           std::complex<double> *out, int size) const {
    for (size_t i = 0; i < size; i++) {
      ASSERT_NEAR(ref[i].real(), out[i].real(), 1e-3);
      ASSERT_NEAR(ref[i].imag(), out[i].imag(), 1e-3);
    }
  }

  template <typename T>
  void print_vector(std::vector<T> vec, std::size_t print_size = 4){
    std::size_t slot_count = vec.size();
    for (std::size_t i = 0; i < std::min(slot_count, print_size); i++) {
      std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
  }
};

TEST_P(E2ETest, Encode) {
  int slots = 8;
  std::complex<double> *mvec = new std::complex<double>[slots];
  std::complex<double> *mvec_ref = new std::complex<double>[slots];
  std::complex<double> *mvec_decoded = new std::complex<double>[slots];
  uint64_t *mvec_encoded = new uint64_t[param.chain_length_ << param.log_degree_];


  for (int i = 0; i < slots; i++) {
    mvec[i] = std::complex<double>(i, i);
    mvec_ref[i] = std::complex<double>(i, i);
  }

  context.Encode(mvec_encoded, mvec, slots);
  context.Decode(mvec_decoded, mvec_encoded, slots);

  COMPARE_APPROXIMATE(mvec_ref, mvec_decoded, slots);
}

INSTANTIATE_TEST_SUITE_P(Params, E2ETest,
                         ::testing::Values(PARAM_LARGE_DNUM, PARAM_SMALL_DNUM));