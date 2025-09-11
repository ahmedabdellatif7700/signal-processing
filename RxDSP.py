import numpy as np
from equalizer import LinearEqualizer

class RxDSP:
    """
    Receiver DSP module with optional linear equalizer (LMS/RLS/CMA)
    and decision-directed adaptation after training.
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        """Reset all internal state."""
        self._format = "QPSK"
        self._ber = np.float32(0.0)
        self._decoded_bits = np.array([], dtype=np.int32)

        # QPSK constellation + Gray mapping
        self._C = np.array([1+1j, -1+1j, -1-1j, 1-1j])
        self._B = np.array([[0,0],[0,1],[1,1],[1,0]])

        # Equalizer instance
        self._eq = None

    def configure(self, eq_type="None", **kwargs):
        """
        Configure equalizer.
        eq_type: "FFE" uses LinearEqualizer
        kwargs: num_taps, algorithm, step_size, forgetting_factor, delta, ref_tap
        """
        if eq_type == "FFE":
            algorithm = kwargs.get("algorithm", "LMS").upper()
            num_taps = kwargs.get("num_taps", kwargs.get("taps", 7))
            step_size = kwargs.get("step_size", kwargs.get("mus", [0.01])[0])
            forgetting_factor = kwargs.get("forgetting_factor", 0.99)
            delta = kwargs.get("delta", 1e3)
            ref_tap = kwargs.get("ref_tap", None)

            self._eq = LinearEqualizer(
                num_taps=num_taps,
                algorithm=algorithm,
                step_size=step_size,
                forgetting_factor=forgetting_factor,
                delta=delta,
                reference_tap=ref_tap,
                constellation=self._C
            )

            # Print equalizer parameters
            print(f"[RxDSP] LinearEqualizer configured:")
            print(f"  Algorithm: {self._eq.algorithm}")
            print(f"  NumTaps: {self._eq.num_taps}")
            print(f"  StepSize / Mu: {self._eq.step_size}")
            if self._eq.algorithm == "RLS":
                print(f"  ForgettingFactor: {self._eq.forgetting_factor}")
                print(f"  InitialInverseCorrelationMatrix: {self._eq.delta}")
            print(f"  Constellation: {self._eq.constellation}")
            print(f"  ReferenceTap: {self._eq.reference_tap}")
        else:
            self._eq = None

    def normalize_signal(self, r_k):
        """Remove DC offset and normalize by standard deviation."""
        r_k = r_k - np.mean(r_k)
        return r_k / np.std(r_k)

    def _cross_correlation_align(self, r_k, t_k):
        """Align received signal to training sequence."""
        corr = np.correlate(r_k, t_k, mode='full')
        delay = np.argmax(np.abs(corr)) - (len(t_k) - 1)
        if delay >= 0:
            r_aligned = r_k[delay:]
            t_aligned = t_k[:len(r_aligned)]
        else:
            r_aligned = r_k[-delay:]
            t_aligned = t_k[:len(r_aligned)]
        return r_aligned, t_aligned

    def _min_distance_decision(self, r_k):
        """Map received symbols to nearest constellation points."""
        yd = np.zeros_like(r_k, dtype=complex)
        for i, sym in enumerate(r_k):
            idx = np.argmin(np.abs(sym - self._C))
            yd[i] = self._C[idx]
        return yd

    def _symbols_to_bits(self, yd):
        bits = []
        for sym in yd:
            idx = np.argmin(np.abs(sym - self._C))
            bits.extend(self._B[idx])
        return np.array(bits, dtype=int)

    def process_signal(self, t_k, r_k, P=None, train_len=None, disp_out=False):
        """
        Process received signal with optional linear equalizer.
        Supports supervised training + decision-directed adaptation.
        """
        t_k = np.ravel(t_k)
        r_k = np.ravel(r_k)

        # Extract parameters from P dict
        if P is not None:
            train_len = P.get("Ks", [200])[0]
            disp_out = P.get("disp_out", False)
            taps = P.get("taps", 7)
            methods = P.get("methods", ["lms"])
            self.configure(
                eq_type=P.get("eq_type", "FFE"),
                num_taps=taps,
                algorithm=methods[0].upper(),
                step_size=P.get("mus", [0.01])[0]
            )

        # Normalize and align signals
        r_k = self.normalize_signal(r_k)
        r_aligned, t_aligned = self._cross_correlation_align(r_k, t_k)
        N = min(len(r_aligned), len(t_aligned))
        r_aligned, t_aligned = r_aligned[:N], t_aligned[:N]

        # Equalization
        if self._eq is None:
            y = r_aligned
            mode = "No equalization"
        else:
            y, e = self._eq.equalize(r_aligned, d=t_aligned, Ks=train_len)
            mode = "Training + Decision-Directed"

        # Decisions
        yd = self._min_distance_decision(y)
        decoded_bits = self._symbols_to_bits(yd)
        t_bits = self._symbols_to_bits(t_aligned)

        # BER calculation
        min_len = min(len(decoded_bits), len(t_bits))
        ber = np.sum(np.abs(decoded_bits[:min_len] - t_bits[:min_len])) / min_len
        self._ber = np.float32(ber)
        self._decoded_bits = decoded_bits

        if disp_out:
            print(f"[RxDSP] BER={self._ber:.4e}, Mode={mode}, Alg={self._eq.algorithm if self._eq else 'None'}")

        return self._ber, decoded_bits
