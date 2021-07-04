/* Copyright (c) by CryptoLab Inc. and Seoul National University R&DB Foundation.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#include "Define.h"
#include "nvToolsExt.h"

namespace ckks {

void CudaNvtxStart(std::string msg) { nvtxRangePushA(msg.c_str()); }
void CudaNvtxStop() { nvtxRangePop(); }
void CudaHostSync() { cudaDeviceSynchronize(); }

}