### How to run amplitude simulation

Prerequisites:
- Python 3.10+
- Install dependencies: quimb, cotengra, autoray, numpy; optionally torch for GPU

Install (CPU-only):
```bash
pip install quimb cotengra autoray numpy
```

Install (with GPU):
```bash
pip install quimb cotengra autoray numpy
pip install torch --index-url https://download.pytorch.org/whl/cu121  # pick CUDA matching your system
```

Compute amplitude for the provided 37-qubit circuit:
```bash
python simulate_amplitude.py --qasm peaked_circuit_diff=37.qasm --bitstring 0001000001101100001011110111111110110 --backend numpy
```

On GPU (if available):
```bash
python simulate_amplitude.py --qasm peaked_circuit_diff=37.qasm --bitstring 0001000001101100001011110111111110110 --backend torch
```

Outputs include the complex amplitude, its squared magnitude, timing breakdown, and GPU peak memory if using torch+CUDA.

Validation tip (small circuits):
- Lower difficulty in `circuit_gen/main.py`, export QASM, and compare TN amplitude with dense simulation in quimb to confirm correctness within ~1e-6.


