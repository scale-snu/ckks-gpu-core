/* Copyright (c) by CryptoLab Inc. and Seoul National University R&DB Foundation.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#pragma once

#include <random>

#include "public/Ciphertext.h"
#include "public/Context.h"
#include "public/Define.h"
#include "public/EvaluationKey.h"
#include "public/MultPtxtBatch.h"
#include "public/Parameter.h"

namespace {

ckks::Parameter SPECIAL_PARAM{
    17, 29, 3, {2305843009146585089, 2251799756013569,    2251799787995137,
                2251800352915457,    2251799780917249,    2251799666884609,
                2251799678943233,    2251799696244737,    2251800082382849,
                2251799776198657,    2251799929028609,    2251799774887937,
                2251799849336833,    2251799883153409,    2251799777771521,
                2251799879483393,    2251799772266497,    2251799763091457,
                2251799844093953,    2251799823384577,    2251799851958273,
                2251799789568001,    2251799797432321,    2251799799267329,
                2251799836753921,    2251799806345217,    2251799807131649,
                2251799818928129,    2251799816568833,    2251799815520257,
                2305843009221820417, 2305843009224179713, 2305843009229946881,
                2305843009255636993, 2305843009350008833, 2305843009448574977,
                2305843009746370561, 2305843009751089153, 2305843010287697921,
                2305843010288484353}};

ckks::Parameter PARAM_LARGE_DNUM{
    16,
    44,
    45,
    {
        2305843009218281473, 2251799661248513, 2251799661641729,
        2251799665180673,    2251799682088961, 2251799678943233,
        2251799717609473,    2251799710138369, 2251799708827649,
        2251799707385857,    2251799713677313, 2251799712366593,
        2251799716691969,    2251799714856961, 2251799726522369,
        2251799726129153,    2251799747493889, 2251799741857793,
        2251799740416001,    2251799746707457, 2251799756013569,
        2251799775805441,    2251799763091457, 2251799767154689,
        2251799765975041,    2251799770562561, 2251799769776129,
        2251799772266497,    2251799775281153, 2251799774887937,
        2251799797432321,    2251799787995137, 2251799787601921,
        2251799791403009,    2251799789568001, 2251799795466241,
        2251799807131649,    2251799806345217, 2251799805165569,
        2251799813554177,    2251799809884161, 2251799810670593,
        2251799818928129,    2251799816568833, 2251799815520257,
        2305843009218936833,
    }};

ckks::Parameter PARAM_SMALL_DNUM{
    16,
    44,
    3,
    {
        2305843009218281473, 2251799661248513,    2251799661641729,
        2251799665180673,    2251799682088961,    2251799678943233,
        2251799717609473,    2251799710138369,    2251799708827649,
        2251799707385857,    2251799713677313,    2251799712366593,
        2251799716691969,    2251799714856961,    2251799726522369,
        2251799726129153,    2251799747493889,    2251799741857793,
        2251799740416001,    2251799746707457,    2251799756013569,
        2251799775805441,    2251799763091457,    2251799767154689,
        2251799765975041,    2251799770562561,    2251799769776129,
        2251799772266497,    2251799775281153,    2251799774887937,
        2251799797432321,    2251799787995137,    2251799787601921,
        2251799791403009,    2251799789568001,    2251799795466241,
        2251799807131649,    2251799806345217,    2251799805165569,
        2251799813554177,    2251799809884161,    2251799810670593,
        2251799818928129,    2251799816568833,    2251799815520257,
        2305843009218936833, 2305843009220116481, 2305843009221820417,
        2305843009224179713, 2305843009225228289, 2305843009227980801,
        2305843009229160449, 2305843009229946881, 2305843009231650817,
        2305843009235189761, 2305843009240301569, 2305843009242923009,
        2305843009244889089, 2305843009245413377, 2305843009247641601,
    }};

using namespace std;

default_random_engine gen{random_device()()};

auto GetRandomPolyRNS(int np, const ckks::Parameter& param) {
  const int degree = param.degree_;
  ckks::HostVector poly(np * degree);
  for (int prime_idx = 0; prime_idx < np; ++prime_idx) {
    auto prime = param.primes_[prime_idx];
    uniform_int_distribution<uint64_t> dist{0, prime - 1};
    auto iter = poly.begin() + prime_idx * degree;
    std::generate_n(iter, degree, [&]() { return dist(gen); });
  }
  return ckks::DeviceVector(poly);
}

auto GetRandomKey(const ckks::Parameter& param) {
  ckks::EvaluationKey key;
  const int np = param.max_num_moduli_;
  auto& ax = key.getAxDevice();
  auto& bx = key.getBxDevice();
  for (int i = 0; i < param.dnum_; i++) {
    ax.append(GetRandomPolyRNS(np, param));
    bx.append(GetRandomPolyRNS(np, param));
  }
  return key;
}

auto GetRandomPolyAfterModUp(int beta, const ckks::Parameter& param) {
  ckks::DeviceVector out;
  for (int i = 0; i < beta; i++) {
    auto in = GetRandomPolyRNS(param.max_num_moduli_, param);
    out.append(in);
  }
  return out;
}

}  // namespace

namespace ckks {

class Test {
 public:
  Test(const Parameter& param) : param{param}, context{param} {};

  auto GetRandomPolyRNS(int num_moduli) const {
    return ::GetRandomPolyRNS(num_moduli, param);
  }

  auto GetRandomPolyAfterModUp(int beta) const {
    return ::GetRandomPolyAfterModUp(beta, param);
  }

  auto GetRandomKey() const { return ::GetRandomKey(param); }

  auto GetRandomCiphertext() const {
    ckks::Ciphertext ctxt;
    auto& ax = ctxt.getAxDevice();
    auto& bx = ctxt.getBxDevice();
    ax = GetRandomPolyRNS(param.chain_length_);
    bx = GetRandomPolyRNS(param.chain_length_);
    return ctxt;
  }

  auto GetRandomPlaintext() const {
    ckks::Plaintext ptxt;
    auto& mx = ptxt.getMxDevice();
    mx = GetRandomPolyRNS(param.chain_length_);
    return ptxt;
  }

  auto GetRandomPoly() const { return GetRandomPolyRNS(param.chain_length_); }

  default_random_engine gen{random_device()()};
  ckks::Parameter param;
  ckks::Context context;
};

}  // namespace ckks