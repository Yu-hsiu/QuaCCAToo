"""Run every code listing of arXiv:2507.18759v2 Sec. III against this repository.

Each listing is copied from the paper as printed, only with the whitespace the PDF extraction
broke put back. Where the paper gives a value as an equation instead of code (hyperfine tensors,
w0, tau1, tau2, phi_rf) the value comes from the matching tutorial, which is said in a comment.

The grids are thinned so the whole script takes 20-40 minutes, mostly XY8; each section checks a
number the paper states, or the data the authors stored in docs/tutorials/sim_data_tutorials.

    python verify_paper_listings.py            # all sections
    python verify_paper_listings.py A C        # only Sec. III A and III C (also: B, XY8)
"""

import copy
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from qutip import basis, fidelity, fock_dm, jmat, ptrace, qeye, tensor

from quaccatoo import (
    CPMG,
    NV,
    XY8,
    Hahn,
    PulsedSim,
    Rabi,
    compose_sys,
    load_quaccatoo,
    square_pulse,
)

TUTORIAL_DATA = Path(__file__).parent / "docs" / "tutorials" / "sim_data_tutorials"
RESULTS = []


def check(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def sec_A():
    """Sec. III A, two-qubit conditional gates, FIG. 3."""
    # System definition
    sys = NV(B0=200, units_B0="mT", N=0)

    # H2 Hamiltonian and initial state definition
    GAMMA_C = 10.7084e-3
    azz = 130
    H2 = azz * tensor(jmat(1, "z"), jmat(1 / 2, "z")) - GAMMA_C * sys.B0 * tensor(
        qeye(3), jmat(1 / 2, "z")
    )
    sys.add_spin(H2)
    sys.rho0 = tensor(basis(3, 1), basis(2, 0))

    # h1 Hamiltonian definition
    w1_S = 20
    w1_I = 0.8
    h1 = w1_S * tensor(jmat(1, "x") * 2**0.5, qeye(2)) + w1_I * tensor(
        qeye(3), jmat(1 / 2, "x") * 2
    )
    w0_mw = sys.energy_levels[2]
    w0_rf = sys.energy_levels[1]

    # Rabi electron simulation; a single mesolve over the grid, so the full 1000 points are cheap
    tp_S = np.linspace(0, 0.15, 1000)
    rabi_S_sim = Rabi(
        system=sys,
        pulse_duration=tp_S,
        h1=h1,
        pulse_params={"f_pulse": w0_mw},
        pulse_shape=square_pulse,
    )
    rabi_S_sim.run()
    r = np.asarray(rabi_S_sim.results)
    # FIG. 3(a): full oscillation between mS=0 and mS=-1, period 1/w1_S = 0.05 us
    i_pi = np.argmin(np.abs(tp_S - 1 / (2 * w1_S)))
    check(
        "III A electron Rabi contrast",
        r.max() > 0.99 and r.min() < 0.01,
        f"min {r.min():.4f} max {r.max():.4f}",
    )
    check(
        "III A electron pi pulse at 1/(2 w1_S)",
        r[i_pi] < 0.02,
        f"F_S(t={tp_S[i_pi]:.4f}) = {r[i_pi]:.4f}",
    )

    # Rabi nuclear simulation; a sparser grid makes the solver exceed nsteps between outputs
    sys.rho0 = tensor(basis(3, 2), basis(2, 0))
    sys.observable = tensor(qeye(3), basis(2, 0) * basis(2, 0).dag())
    gamma2 = 0.5
    sys.c_ops = gamma2 * tensor(qeye(3), jmat(1 / 2, "z"))
    tp_I = np.linspace(0, 2.5, 1000)
    rabi_I_sim = Rabi(system=sys, pulse_duration=tp_I, h1=h1, pulse_params={"f_pulse": w0_rf})
    rabi_I_sim.run()
    r = np.asarray(rabi_I_sim.results)
    # FIG. 3(b): damped oscillation starting at 1, first minimum near 1/(2 w1_I) = 0.625 us
    t_min = tp_I[np.argmin(r[: len(r) // 2])]
    check("III A nuclear Rabi starts polarised", abs(r[0] - 1) < 1e-6, f"F_I(0) = {r[0]:.6f}")
    check(
        "III A nuclear Rabi first minimum",
        abs(t_min - 0.625) < 0.1 and r.min() < 0.2,
        f"t = {t_min:.3f} us, F_I = {r.min():.3f}",
    )
    check(
        "III A nuclear Rabi damped",
        r[-1] > r.min() + 0.2 and r[-1] < 0.9,
        f"F_I(2.5 us) = {r[-1]:.3f}",
    )


def sec_B_hahn():
    """Sec. III B, Hahn echo of the NV-13C pair, FIG. 4(a)."""
    # System definition
    sys1 = NV(B0=4.2, units_B0="mT", theta=-45, units_angles="deg", N=14, temp=300, units_temp="K")

    # hyperfine tensor and 13C Zeeman: given as an equation in the paper, code from tutorial 04
    alpha = np.array([[5, -6.3, -2.9], [-6.3, 4.2, -2.3], [-2.9, -2.3, 8.2]])
    S = [jmat(1, "x"), jmat(1, "y"), jmat(1, "z")]
    I = [jmat(1 / 2, "x"), jmat(1 / 2, "y"), jmat(1 / 2, "z")]
    H2 = sum(alpha[i, j] * tensor(S[i], qeye(3), I[j]) for i in range(3) for j in range(3))
    H2 -= (
        10.705e-3
        * sys1.B0
        * tensor(qeye(3), qeye(3), np.cos(sys1.theta) * I[2] + np.sin(sys1.theta) * I[0])
    )
    sys1.add_spin(H2)

    # w1 and w0: given as an equation in the paper, code from tutorial 04
    w1 = 15
    w0 = np.sum(sys1.energy_levels[12:]) / 6 - np.sum(sys1.energy_levels[1:6]) / 6

    ref = np.load(TUTORIAL_DATA / "Hahn_13C.npz")
    idx = np.arange(0, 2000, 80)  # 25 of the 2000 tau values of the paper

    def hahn(map_kw):
        hahn_sim = Hahn(
            free_duration=ref["tau"][idx],
            pi_pulse_duration=0.0316,
            system=sys1,
            h1=w1 * sys1.MW_h1,
            pulse_params={"f_pulse": w0},
            projection_pulse=True,
            time_steps=1000,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            hahn_sim.run(map_kw=map_kw)
        return np.asarray(hahn_sim.results), [str(w.message)[:80] for w in caught]

    serial, _ = hahn(None)
    dev = np.max(np.abs(serial - ref["results"][idx]))
    check(
        "III B Hahn vs authors' data (serial)",
        dev < 1e-3,
        f"max |diff| = {dev:.2e} over {len(idx)} tau",
    )

    # the listing itself asks for 32 workers
    parallel, caught = hahn({"num_cpus": 32})
    dev = np.max(np.abs(parallel - ref["results"][idx]))
    check(
        "III B Hahn listing run(map_kw={'num_cpus': 32})",
        dev < 1e-3,
        f"max |diff| = {dev:.2e}; warnings: {caught}",
    )


def sec_B_cpmg():
    """Sec. III B, CPMG of a weakly coupled 13C, FIG. 4(b)."""
    # sys2 and H2: described in prose and as an equation in the paper, code from tutorial 04
    sys2 = NV(B0=40.1, units_B0="mT", N=14)
    tpi = 0.01
    w1 = 1 / tpi / 2
    axz = 0.055 * np.sin(np.pi * 54 / 180)
    azz = 0.055 * np.cos(np.pi * 54 / 180)
    H2 = (
        axz
        * (
            tensor(jmat(1, "x"), qeye(3), jmat(1 / 2, "z"))
            + tensor(jmat(1, "z"), qeye(3), jmat(1 / 2, "x"))
        )
        + azz * tensor(jmat(1, "z"), qeye(3), jmat(1 / 2, "z"))
        - 10.705e-3 * sys2.B0 * tensor(qeye(3), qeye(3), jmat(1 / 2, "z"))
    )
    sys2.add_spin(H2)
    w0 = np.mean(sys2.energy_levels[6:12]) - np.sum(sys2.energy_levels[1:6]) / 6

    ref = load_quaccatoo(str(TUTORIAL_DATA / "CPMG-16_13C"))
    idx = np.array([0, 60, 90, 100, 110, 199])  # 6 of the 200 tau values, across the resonance
    M = 16

    # CPMG simulation, as printed
    sol_opt = {"atol": 1e-16, "rtol": 1e-16, "nsteps": 1e8, "order": 30}
    tau_cpmg = np.asarray(ref.variable)[idx]
    cpmg_sim = CPMG(
        free_duration=tau_cpmg,
        pi_pulse_duration=0.01,
        system=sys2,
        h1=w1 * sys2.MW_h1,
        pulse_params={"f_pulse": w0},
        M=M,
        time_steps=1000,
        options=sol_opt,
    )
    cpmg_sim.run()
    dev = np.max(np.abs(np.asarray(cpmg_sim.results) - np.asarray(ref.results)[idx]))
    # the paper prints 0.01, the authors' data used 0.00999483, hence the looser bound
    check(
        "III B CPMG-16 vs authors' data", dev < 5e-3, f"max |diff| = {dev:.2e} over {len(idx)} tau"
    )


def sec_B_xy8():
    """Sec. III B, XY8-12 sensing of a classical field B2(t), FIG. 5(b)."""
    qsys3 = NV(N=15, B0=40, units_B0="mT")

    # The lab-frame MW carrier (~1.7 GHz, period 0.6 ns) makes the signal jump between tau values
    # that differ by less than a nanosecond, so the check samples exactly the tau grid of tutorial
    # 05 (100 points over 0.06-0.17 us) whose stored plot matches FIG. 5: the baseline, the
    # resonance at k = 28, 29 and the spurious harmonics near 5/4 and 3/2 tau0.
    grid = np.linspace(0.06, 0.17, 100)
    base, peak, harm = [0, 5, 10], [28, 29], [47, 48, 49, 68, 69, 70]
    tau_xy8 = grid[base + peak + harm]  # paper: 1000 points
    n_b, n_p = len(base), len(peak)

    def run_xy8(B2, RXY8=False):
        H2 = [tensor(jmat(1, "z"), qeye(2)), B2]
        w1 = 20
        xy8_sim = XY8(
            M=12,
            free_duration=tau_xy8,
            pi_pulse_duration=1 / 2 / w1,
            system=qsys3,
            h1=w1 * qsys3.MW_h1,
            pulse_params={"f_pulse": qsys3.MW_freqs[0]},
            H2=H2,
            RXY8=RXY8,
            seed=1 if RXY8 else None,  # not in the listing, keeps the random phases reproducible
        )
        xy8_sim.run()
        r = np.asarray(xy8_sim.results)
        return r[:n_b], r[n_b : n_b + n_p], r[n_b + n_p :]

    # B2(t) field definition, as printed: the defaults are swapped with respect to the text
    def B2_printed(t, gamma_B2=5.5, w2=0.3):
        return gamma_B2 * np.sin(w2 * t)

    # as the text states it: gamma_e B2 = 0.3 MHz and w2 = 5.5 MHz
    def B2(t, gamma_B2=0.3, w2=5.5):
        return gamma_B2 * np.sin(w2 * t)

    # starting in mS=0 the signal stays near 0 and the resonance shows up as a peak towards 1
    b, p, h = run_xy8(B2)
    check(
        "III B XY8-12 resonance at 1/(2 w2)",
        p.min() > 0.9 and b.max() < 0.1,
        f"F_S = {np.round(p, 3).tolist()} at tau = {np.round(grid[peak], 4).tolist()} us, baseline <= {b.max():.3f}",
    )

    b2, p2, h2 = run_xy8(B2_printed)
    check(
        "III B XY8-12 with the B2 defaults as printed misses FIG. 5",
        p2.max() < 0.9,
        f"F_S at the resonance = {np.round(p2, 3).tolist()}, baseline {np.round(b2, 3).tolist()}, "
        f"harmonics {np.round(h2, 3).tolist()}: a typo of the paper (tutorial 05 is right)",
    )

    _, p_r, h_r = run_xy8(B2, RXY8=True)
    check(
        "III B RXY8-12 keeps the resonance, suppresses harmonics",
        p_r.min() > 0.9 and h_r.max() < h.max(),
        f"peak F_S = {np.round(p_r, 3).tolist()}; harmonics max XY8 {h.max():.3f} vs RXY8 {h_r.max():.3f}",
    )


def teleport_until_measurement(alpha, beta, phi_rf):
    """Sec. III C listings up to Alice's measurement."""
    # System definition
    NVb = NV(B0=18, units_B0="mT", N=0)
    NVa = NV(B0=25, units_B0="mT", N=14)
    NVb.truncate(mS=1)
    NVa.truncate(mS=1, mI=1)
    sys = compose_sys(NVb, NVa)

    # Initial state definition
    Psi_ = tensor(basis(2, 0), basis(2, 1)) - tensor(basis(2, 1), basis(2, 0))
    psi = alpha * basis(2, 0) + beta * basis(2, 1)
    sys.rho0 = tensor(Psi_, psi).unit()

    # Sequence instantiation
    seq = PulsedSim(sys)

    # CNOT gate parameters
    sol_opt = {"nsteps": 1e9}
    w0_cnot = NVa.energy_levels[2]
    w1_cnot = 2.14 / 3**0.5
    tpi_cnot = 1 / (2 * w1_cnot)
    h1_cnot = w1_cnot * tensor(qeye(2), NVa.MW_h1)

    # CNOT execution
    seq.add_pulse(
        duration=tpi_cnot,
        h1=h1_cnot,
        pulse_params={"f_pulse": w0_cnot, "phi_t": np.pi / 2},
        options=sol_opt,
    )

    # Hadamard gate parameters (printed after the refocus, needed here for tau1 and tau2)
    w0_rf = NVa.RF_freqs[2]
    w0_mwa = NVa.MW_freqs[0]
    w1_rf = 0.2
    w1_mwa = 16
    tpi_rf = 1 / (2 * w1_rf)
    tpi_mwa = 1 / (2 * w1_mwa)
    h1_rf = w1_rf * tensor(qeye(2), NVa.RF_h1)
    h1_mwa = w1_mwa * tensor(qeye(2), NVa.MW_h1)

    # Refocus scheme; tau1 and tau2 from the equation in the text and the 2.26 us total (tutorial 06)
    w1_mwb = 22
    tpi_mwb = 1 / (2 * w1_mwb)
    w0_mwb = NVb.MW_freqs[0]
    h1_mwb = w1_mwb * tensor(NVb.MW_h1, qeye(2), qeye(2))
    tau_refocus = 2.26
    T = tpi_cnot + tau_refocus + tpi_rf / 2 + tpi_mwa + tpi_rf / 2
    tau1 = T / 2 - tpi_cnot - tpi_mwb / 2
    tau2 = tau_refocus - tau1 - tpi_mwb

    seq.add_free_evolution(tau1)
    seq.add_pulse(
        duration=tpi_mwb,
        h1=h1_mwb,
        pulse_params={"f_pulse": w0_mwb, "phi_t": np.pi / 2},
        options=sol_opt,
    )
    seq.add_free_evolution(tau2)

    # Hadamard gate
    seq.add_pulse(
        duration=tpi_rf / 2,
        h1=h1_rf,
        pulse_params={"f_pulse": w0_rf, "phi_t": phi_rf},
        options=sol_opt,
    )
    seq.add_pulse(
        duration=tpi_mwa,
        h1=h1_mwa,
        pulse_params={"f_pulse": w0_mwa, "phi_t": np.pi / 2},
        options=sol_opt,
    )
    seq.add_pulse(
        duration=tpi_rf / 2,
        h1=h1_rf,
        pulse_params={"f_pulse": w0_rf, "phi_t": phi_rf},
        options=sol_opt,
    )

    bob = {"tpi": tpi_mwb, "h1": h1_mwb, "w0": w0_mwb, "sol_opt": sol_opt}
    return seq, psi, bob


def bob_reconstruct(seq, label, c0, c1, bob):
    """U(c0, c1) of Table I, with the pulses and tau3 of tutorial 06."""
    tpi, h1, w0, opt = bob["tpi"], bob["h1"], bob["w0"], bob["sol_opt"]
    if label == "+Z":
        # State reconstruction, as printed
        if c0 == 0:
            seq.add_pulse(tpi, h1, pulse_params={"f_pulse": w0, "phi_t": np.pi / 2}, options=opt)
        return
    tau3 = np.ceil(w0 * tpi / 2) / w0 - tpi / 2
    rz = 1 / (2 * w0)  # R_z(pi) as a free evolution
    # phi_t of the pi/2 pulse and the extra R_z(pi), per outcome
    table = {
        "+X": {
            (0, 0): (-np.pi / 2, 0),
            (0, 1): (np.pi / 2, rz),
            (1, 0): (np.pi / 2, rz),
            (1, 1): (-np.pi / 2, 0),
        },
        "+Y": {(0, 0): (0, rz), (0, 1): (np.pi, 0), (1, 0): (0, rz), (1, 1): (np.pi, 0)},
    }
    phi, extra = table[label][(c0, c1)]
    seq.add_pulse(tpi / 2, h1, pulse_params={"f_pulse": w0, "phi_t": phi}, options=opt)
    seq.add_free_evolution(tau3 + extra)


def sec_C():
    """Sec. III C, teleportation between two NVs, FIG. 7 and Table I."""
    phi_rf = 2.95  # paper: calibrated to 2.95 rad (169 deg)
    table_I = {
        "+X": [0.9624, 0.9814, 0.9889, 0.9878],
        "+Y": [0.9585, 0.9713, 0.9779, 0.9945],
        "+Z": [0.9999, 0.9998, 0.9985, 0.9981],
    }
    inputs = {"+X": (1 / 2**0.5, 1 / 2**0.5), "+Y": (1 / 2**0.5, 1j / 2**0.5), "+Z": (1, 0)}

    for label, (a, b) in inputs.items():
        seq, psi, bob = teleport_until_measurement(a, b, phi_rf)

        # Alice measurements, as printed (the outcome is random)
        listing = copy.deepcopy(seq)
        obs0 = tensor(qeye(2), fock_dm(2, 0), qeye(2))
        c0 = 1 - listing.measure_qsys(observable=obs0)
        obs1 = tensor(qeye(2), qeye(2), fock_dm(2, 1))
        c1 = listing.measure_qsys(observable=obs1)
        check(
            f"III C {label} measure_qsys gives classical bits",
            {c0, c1} <= {0.0, 1.0},
            f"c0 = {c0}, c1 = {c1}",
        )

        # every outcome of Table I, by projecting instead of sampling
        fids = []
        for c0, c1 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            branch = copy.deepcopy(seq)
            branch.rho = (tensor(qeye(2), fock_dm(2, c0), fock_dm(2, c1)) * branch.rho).unit()
            bob_reconstruct(branch, label, c0, c1, bob)
            rho_bob = ptrace(branch.rho * branch.rho.dag(), 0)
            # Table I quotes QuTiP's fidelity, sqrt(<psi|rho|psi>) for a pure target
            fids.append(fidelity(rho_bob, psi * psi.dag()))
        dev = np.max(np.abs(np.array(fids) - table_I[label]))
        check(
            f"III C Table I {label}",
            dev < 1e-3,
            f"sim {np.round(fids, 4).tolist()} paper {table_I[label]}",
        )


SECTIONS = {
    "A": [sec_A],
    "B": [sec_B_hahn, sec_B_cpmg, sec_B_xy8],
    "C": [sec_C],
    "XY8": [sec_B_xy8],
}

if __name__ == "__main__":
    for key in sys.argv[1:] or ["A", "B", "C"]:
        for fn in SECTIONS[key]:
            t0 = time.time()
            print(f"== {fn.__doc__.splitlines()[0]}", flush=True)
            fn()
            print(f"   ({time.time() - t0:.0f} s)", flush=True)
    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for name in failed:
        print("  FAILED:", name)
    sys.exit(1 if failed else 0)
