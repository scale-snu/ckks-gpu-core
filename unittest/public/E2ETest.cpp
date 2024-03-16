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

#include "seal/seal.h"

class E2ETest : public ckks::Test,
                   public ::testing::TestWithParam<ckks::Parameter> {
 protected:
  E2ETest() : ckks::Test(GetParam()){};

  void COMPARE(const ckks::DeviceVector& ref,
               const ckks::DeviceVector& out) const {
    ASSERT_EQ(ckks::HostVector(ref), ckks::HostVector(out));
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

TEST_P(E2ETest, Client) {
  const int slots = 8;
  std::complex<double> *mvec = new std::complex<double>[slots];
  for (int i = 0; i < slots; i++) {
    mvec[i] = std::complex<double>(i, i);
  }

  // Client operation using SEAL
  seal::EncryptionParameters parms(seal::scheme_type::ckks);
  size_t poly_modulus_degree = 8192;
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));

  seal::SEALContext context(parms);
  seal::CKKSEncoder encoder(context);

  seal::KeyGenerator keygen(context);
  auto secret_key = keygen.secret_key();
  seal::PublicKey public_key;
  keygen.create_public_key(public_key);
  seal::Encryptor encryptor(context, public_key);
  seal::Decryptor decryptor(context, secret_key);

  double scale = pow(2.0, 40);
  size_t slot_count = encoder.slot_count();
  cout << "Number of slots: " << slot_count << endl;
  
  vector<double> input_x;
  input_x.reserve(slot_count);
  for (size_t i = 0; i < slot_count; i++)
  {
      input_x.push_back(i);
  }
  print_vector(input_x, 10);

  // Encode
  seal::Plaintext encode_x;
  encoder.encode(input_x, scale, encode_x);

  // Encrypt
  seal::Ciphertext encrypt_x;
  encryptor.encrypt(encode_x, encrypt_x);

  // Some operations
  seal::Ciphertext result_x = encrypt_x;

  // Decrypt
  seal::Plaintext decrypt_x;
  decryptor.decrypt(result_x, decrypt_x);

  // Decode with SEAL
  vector<double> decode_x;
  encoder.decode(decrypt_x, decode_x);
  print_vector(decode_x, 10);
  delete[] mvec;  
}

INSTANTIATE_TEST_SUITE_P(Params, E2ETest,
                         ::testing::Values(PARAM_LARGE_DNUM, PARAM_SMALL_DNUM));