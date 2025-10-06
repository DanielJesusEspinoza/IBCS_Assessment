import argparse
import math
import os
import re
import time
from typing import List, Tuple

import autoray as ar
import numpy as np

try:
    import torch
except Exception:  # torch is optional
    torch = None

import cotengra as ctg
import quimb.tensor as qtn


# ------------------------------ Gate Libraries ------------------------------

PI = math.pi


def as_backend_array(x: np.ndarray, backend: str):
    if backend == "torch":
        if torch is None:
            raise RuntimeError("Torch backend requested but torch is not available.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.tensor(x, dtype=torch.complex64, device=device)
    # numpy (default)
    return np.asarray(x, dtype=np.complex64)


def gate_matrix(name: str, params: List[float] | None, backend: str):
    n = name.lower()
    if n == "h":
        m = (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex64)
        return as_backend_array(m, backend)
    if n == "s":
        m = np.array([[1, 0], [0, 1j]], dtype=np.complex64)
        return as_backend_array(m, backend)
    if n in ("sdg", "sdg"):
        m = np.array([[1, 0], [0, -1j]], dtype=np.complex64)
        return as_backend_array(m, backend)
    if n == "rx":
        (theta,) = params
        c = math.cos(theta / 2)
        s = -1j * math.sin(theta / 2)
        m = np.array([[c, s], [s, c]], dtype=np.complex64)
        return as_backend_array(m, backend)
    if n == "ry":
        (theta,) = params
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        m = np.array([[c, -s], [s, c]], dtype=np.complex64)
        return as_backend_array(m, backend)
    if n == "rz":
        (theta,) = params
        m = np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=np.complex64)
        return as_backend_array(m, backend)
    if n == "u3":
        # Implement via Euler angles by composing elementary rotations
        # U3(theta, phi, lam) = Rz(phi - pi/2) Rx(pi/2) Rz(pi - theta) Rx(pi/2) Rz(lam - pi/2)
        theta, phi, lam = params
        return (n, [theta, phi, lam])  # special marker handled by caller
    if n == "cx":
        m = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex64
        )
        return as_backend_array(m, backend)
    raise ValueError(f"Unsupported gate: {name}")


def apply_u3(psi: qtn.MPS, q: int, theta: float, phi: float, lam: float, backend: str):
    rz1 = gate_matrix("rz", [phi - PI / 2], backend)
    rx = gate_matrix("rx", [PI / 2], backend)
    rz2 = gate_matrix("rz", [PI - theta], backend)
    rz3 = gate_matrix("rz", [lam - PI / 2], backend)
    psi.gate_(rz1, q)
    psi.gate_(rx, q)
    psi.gate_(rz2, q)
    psi.gate_(rx, q)
    psi.gate_(rz3, q)


# ------------------------------ QASM Parsing ------------------------------

_RE_QREG = re.compile(r"^qreg\\s+q\\[(\\d+)\\];")
_RE_GATE = re.compile(r"^(\\w+)(?:\\(([^)]*)\\))?\\s+(.+);$")


def parse_qasm(qasm_text: str) -> Tuple[int, List[Tuple[str, List[float] | None, List[int]]]]:
    lines = [ln.strip() for ln in qasm_text.splitlines()]
    n_qubits = None
    ops: List[Tuple[str, List[float] | None, List[int]]] = []
    for ln in lines:
        if not ln or ln.startswith("//"):
            continue
        if ln.startswith("OPENQASM") or ln.startswith("include") or ln.startswith("creg"):
            continue
        if ln.startswith("qreg"):
            m = _RE_QREG.match(ln)
            if m:
                n_qubits = int(m.group(1))
            continue
        if ln.startswith("measure"):
            break
        m = _RE_GATE.match(ln)
        if not m:
            continue
        gname = m.group(1)
        par_str = m.group(2)
        ops_str = m.group(3)
        params = None
        if par_str is not None and par_str.strip() != "":
            params = [float(x) for x in par_str.split(",")]
        # operands like q[0],q[1]
        qubits = []
        for tok in ops_str.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok.startswith("q[") and tok.endswith("]"):
                qubits.append(int(tok[2:-1]))
        ops.append((gname, params, qubits))
    if n_qubits is None:
        raise ValueError("Failed to parse qreg from QASM")
    return n_qubits, ops


# ------------------------------ Simulation Core ------------------------------


def build_initial_mps(n_qubits: int, backend: str) -> qtn.MPS:
    psi = qtn.MPS_computational_state("0" * n_qubits).astype_("complex64")
    if backend == "torch":
        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        psi.apply_to_arrays(lambda a: torch.as_tensor(a, dtype=torch.complex64, device=device))
    return psi


def apply_qasm_ops(psi: qtn.MPS, ops: List[Tuple[str, List[float] | None, List[int]]], backend: str):
    for gname, params, qubits in ops:
        g = gname.lower()
        if g == "u3":
            q = qubits[0]
            apply_u3(psi, q, params[0], params[1], params[2], backend)
            continue
        mat = gate_matrix(g, params, backend)
        if g == "cx":
            psi.gate_(mat, (qubits[0], qubits[1]))
        else:
            psi.gate_(mat, qubits[0])


def compute_amplitude(psi: qtn.MPS, bitstring: str) -> complex:
    # Contract <bitstring|psi> using an MPS for the bra
    phi = qtn.MPS_computational_state(bitstring).astype_("complex64")
    # Use tensor network contraction with a greedy optimizer
    opt = ctg.ReusableHyperOptimizer(methods=["greedy"], progbar=False)
    amp = (phi.H & psi).contract(all, optimize=opt)
    return complex(amp)


def measure_memory_gpu() -> int:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated())
    return 0


def simulate(qasm_path: str, bitstring: str, backend: str) -> dict:
    with open(qasm_path, "r", encoding="utf-8") as f:
        qasm = f.read()
    n_qubits, ops = parse_qasm(qasm)
    if len(bitstring) != n_qubits:
        raise ValueError(f"Bitstring length {len(bitstring)} != circuit qubits {n_qubits}")

    if backend not in ("numpy", "torch"):
        raise ValueError("backend must be 'numpy' or 'torch'")

    if backend == "torch" and torch is None:
        raise RuntimeError("Torch backend requested but torch is not installed.")

    if backend == "torch" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    psi = build_initial_mps(n_qubits, backend)
    t1 = time.perf_counter()
    apply_qasm_ops(psi, ops, backend)
    t2 = time.perf_counter()
    amp = compute_amplitude(psi, bitstring)
    t3 = time.perf_counter()

    gpu_mem = measure_memory_gpu() if backend == "torch" else 0

    return {
        "n_qubits": n_qubits,
        "amplitude": amp,
        "abs2": float(abs(amp) ** 2),
        "t_init_s": t1 - t0,
        "t_apply_s": t2 - t1,
        "t_contract_s": t3 - t2,
        "t_total_s": t3 - t0,
        "gpu_peak_bytes": gpu_mem,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute amplitude of a basis state for a QASM circuit via tensor networks")
    parser.add_argument("--qasm", required=True, help="Path to QASM file")
    parser.add_argument("--bitstring", required=True, help="Computational basis state, e.g. 0001...")
    parser.add_argument("--backend", default="numpy", choices=["numpy", "torch"], help="Array backend")
    args = parser.parse_args()

    res = simulate(args.qasm, args.bitstring.strip(), args.backend)

    print("Amplitude:", res["amplitude"]) 
    print("|Amplitude|^2:", res["abs2"]) 
    print("n_qubits:", res["n_qubits"]) 
    print("t_init_s:", res["t_init_s"]) 
    print("t_apply_s:", res["t_apply_s"]) 
    print("t_contract_s:", res["t_contract_s"]) 
    print("t_total_s:", res["t_total_s"]) 
    if res["gpu_peak_bytes"]:
        print("gpu_peak_MB:", round(res["gpu_peak_bytes"] / (1024 ** 2), 2))


if __name__ == "__main__":
    main()


