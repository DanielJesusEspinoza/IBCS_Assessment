### Validation on small circuits

Use dense simulation to cross-check amplitudes from `simulate_amplitude.py` for ≤14 qubits.

Example (pseudo):
```python
import quimb as qu
import quimb.tensor as qtn
from simulate_amplitude import parse_qasm

qasm = open('small.qasm').read()
n, ops = parse_qasm(qasm)

# Build dense state
psi = qu.basis(2**n, 0)
for g, params, qubits in ops:
    # apply using quimb gates to dense tensor (left to the reader)
    pass

bitstring = '0101...'
idx = int(bitstring, 2)
amp_dense = psi[idx]

# Compare to TN amplitude from simulate_amplitude.py
```

Alternatively, generate a tiny circuit via `circuit_gen/main.py` with lower difficulty, export QASM, then compare amplitudes between dense and TN contractions. Aim for tolerance ≤1e-6.


