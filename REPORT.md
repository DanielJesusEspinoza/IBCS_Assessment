### Quantum Simulation Assignment Report

#### Circuit workflow explanation
- Circuit creation: `circuit_gen/lib/circuit_gen.py` builds peaked circuits by optimizing a PQC attached to an RQC so that a target computational basis state has high probability. Optimization runs over tensor networks (quimb) with cotengra planning.
- Representation: Resulting two-qubit SU(4) blocks are converted to gate-level circuits; `circuit_gen/lib/circuit.py` decomposes SU(4) via either a CNOT-based or Ising-like scheme into standard gates.
- Serialization: `PeakedCircuit.to_qasm()` emits OpenQASM 2.0 including measurement lines. We then use that QASM as input for amplitude simulation.

#### Results of the amplitude calculation
- Target basis state: `0001000001101100001011110111111110110` (n = 37)
- Command used:
  - CPU: `python simulate_amplitude.py --qasm peaked_circuit_diff=37.qasm --bitstring 0001000001101100001011110111111110110 --backend numpy`
  - GPU: `python simulate_amplitude.py --qasm peaked_circuit_diff=37.qasm --bitstring 0001000001101100001011110111111110110 --backend torch`
- Observed (fill after running):
  - Amplitude: <complex>
  - |Amplitude|^2: <float>
  - t_total_s: <seconds>
  - gpu_peak_MB: <MB if GPU>

#### Performance benchmarks and hardware requirements
- Method: MPS contraction for ⟨bitstring|ψ⟩ with cotengra greedy optimizer; we report init/apply/contract/total times, and GPU peak VRAM when using torch+CUDA.
- Observations and scaling:
  - Runtime is dominated by contraction and increases with entanglement (bond dims) and depth; not strictly 2^n unless the circuit is highly entangling.
  - 37-qubit circuits are feasible with MPS if bond dimensions remain moderate.
- Suggested minimum hardware:
  - CPU-only: 8+ cores, 32+ GB RAM recommended.
  - GPU: NVIDIA with ≥12 GB VRAM (e.g., RTX 3080/3090/A5000) for faster contraction and reduced wall-clock.
- Memory: CPU RAM modest for MPS; GPU usage depends on intermediates; we record peak via torch CUDA stats.

#### Validation
- For ≤14 qubits, cross-check by dense statevector simulation (quimb) and compare amplitudes against the TN method within tolerance ~1e-6.

