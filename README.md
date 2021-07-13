## (Experimental) CKKS-GPU-CORE
This repository includes GPU implementations of the core functions for CKKS (e.g., base conversions, NTT, key-switching, etc.) to reproduce the results in the paper **Over 100x Faster Bootstrapping in Fully Homomorphic Encryption through Memory-centric Optimization with GPUs** submitted to TCHES 2021.

The implementations included largely contribute to [CRYPTOLAB INC.](https://www.cryptolab.co.kr/eng/) and [HEAAN](https://github.com/snucrypto/HEAAN).

### Requirement
- NVIDIA GPU (compute capability over 7.0) 
- cmake v3.18 or over.

Tested under ubuntu 18.04 with g++ 9.3.0 and NVIDIA V100.

### How to build
Check out this [notebook](Run.ipynb).

Alternatively, use the included dockerfile (make sure [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) is installed).
```sh
docker build -t ckks-gpu-core .
docker run --gpus all --rm ckks-gpu-core build/benchmark/Benchmark
```

### Contact
Wonkyung Jung
jungwk@scale.snu.ac.kr
