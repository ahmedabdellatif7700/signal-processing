import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.signal import lfilter

# ======================= LinearEqualizer =======================
class LinearEqualizer:
    """
    Linear equalizer with LMS (can be extended to RLS/CMA) and
    hybrid supervised -> decision-directed adaptation.
    """

    def __init__(self,
                 num_taps=8,
                 algorithm='LMS',
                 step_size=0.01,
                 forgetting_factor=0.3,
                 delta=1e3,
                 reference_tap=None,
                 constellation=None):
        self.num_taps = int(num_taps)
        self.algorithm = algorithm.upper()
        assert self.algorithm in ('LMS', 'RLS', 'CMA'), "Algorithm must be 'LMS', 'RLS', or 'CMA'"
        self.step_size = float(step_size)
        self.forgetting_factor = float(forgetting_factor)
        self.delta = float(delta)
        if reference_tap is None:
            self.reference_tap = (self.num_taps // 2)
        else:
            self.reference_tap = max(0, int(reference_tap) - 1)
        self.constellation = None if constellation is None else np.array(constellation, dtype=complex)
        self.reset()

        # Print configuration
        print("=== LinearEqualizer Configuration ===")
        print(f"Algorithm: {self.algorithm}")
        print(f"NumTaps: {self.num_taps}")
        print(f"StepSize / ForgettingFactor: {self.step_size} / {self.forgetting_factor}")
        print(f"ReferenceTap (0-based): {self.reference_tap}")
        if self.constellation is not None:
            print(f"Constellation: {self.constellation}")
        print("====================================\n")

    def reset(self):
        """Initialize weights and RLS matrix if needed."""
        self.weights = np.zeros(self.num_taps, dtype=complex)
        self.weights[self.reference_tap] = 1.0 + 0j
        self.P = (self.delta * np.eye(self.num_taps, dtype=complex)) if self.algorithm == 'RLS' else None

    def _decision(self, y):
        """Decision-directed mapping to nearest constellation point."""
        if self.constellation is not None:
            dists = np.abs(y.reshape(-1, 1) - self.constellation.reshape(1, -1))
            idx = dists.argmin(axis=1)
            return self.constellation[idx]
        else:
            return np.sign(y.real) + 1j*np.sign(y.imag)

    def _compute_cma_R(self):
        if self.constellation is None:
            return 1.0
        magsq = np.abs(self.constellation)**2
        return (np.mean(magsq**2) / np.mean(magsq))

    def equalize(self, x, d=None, Ks=None):
        """
        Equalize a sequence with optional supervised training and decision-directed mode.

        Parameters
        ----------
        x : array_like
            Received complex samples (1D)
        d : array_like, optional
            Desired sequence for supervised training
        Ks : int, optional
            Number of supervised training symbols. After Ks, decision-directed mode is used.

        Returns
        -------
        y_out : np.ndarray
            Equalizer output sequence
        e_out : np.ndarray
            Error sequence used for adaptation
        """
        x = np.asarray(x, dtype=complex).ravel()
        if d is not None:
            d = np.asarray(d, dtype=complex).ravel()
        N = x.size
        pad = np.zeros(self.num_taps - 1, dtype=complex)
        buffer = np.concatenate([pad, x])
        y_out = np.empty(N, dtype=complex)
        e_out = np.empty(N, dtype=complex)

        if Ks is None:
            Ks = N if d is not None else 0
        Ks = int(Ks)

        print(f"Starting equalization: {N} samples, Ks={Ks} supervised samples\n")

        for n in range(N):
            u = buffer[n : n + self.num_taps][::-1]  # newest first
            y = np.vdot(self.weights, u)

            # Determine error
            if (d is not None) and (n < Ks):
                desired = d[n]
                err = desired - y
                mode = "Training"
            else:
                if self.constellation is None:
                    err = 0.0 + 0j
                    mode = "No adaptation (no constellation)"
                else:
                    decided = self._decision(np.array([y]))[0]
                    err = decided - y
                    mode = "Decision-Directed"

            # Update weights (LMS only for now)
            if self.algorithm == "LMS":
                self.weights += self.step_size * np.conj(err) * u

            y_out[n] = y
            e_out[n] = err

            # Debug print every 10% of samples
            if n % max(1, N//10) == 0:
                center = self.num_taps // 2
                print(f"Sample {n+1}/{N}: y={y:.4f}, err={err:.4f}, mode={mode}")
                print(f"Current weights (center 3 taps): {self.weights[center-1:center+2]}\n")

        print("Equalization completed.\n")
        return y_out, e_out

# ======================= TxDSP =======================
class TxDSP:
    def __init__(self):
        self._reset()

    def _reset(self):
        """Reset all attributes to default values."""
        self.P = {
            'format': 0,
            'Rb': 0.0,
            'fs': 0.0,
            'M': 0,
            'k': 0,
            'Nbits': 0,
            'Nsym': 0,
            'Rs': 0.0,
            'Ts': 0.0,
            'ts': 0.0,
            'sps': 0,
            'symbols': np.array([], dtype=np.complex64),
            'OriginalSymbols': np.array([], dtype=np.complex64),
        }

    def configure(self, Nbits: int, format: int = 1, Rb: float = 100e9, fs: float = 8*100e9):
        """
        Configure TxDSP for QPSK.
        """
        self.P['format'] = format
        self.P['Rb'] = Rb
        self.P['fs'] = fs
        self.P['Nbits'] = Nbits
        if self.P['format'] != 1:
            raise ValueError("Only QPSK (format=1) is supported.")
        # QPSK parameters
        self.P['M'] = 4
        self.P['k'] = 2
        self.P['Rs'] = self.P['Rb'] / self.P['k']
        self.P['Ts'] = 1.0 / self.P['Rs']
        self.P['ts'] = 1.0 / self.P['fs']
        self.P['sps'] = round(self.P['fs'] / self.P['Rs'])
        self.P['Nsym'] = self.P['Nbits'] // self.P['k']

    def _bits_to_symbols(self, bits: np.ndarray):
        """
        Map bits to QPSK symbols using Gray coding.
        """
        if len(bits) % self.P['k'] != 0:
            raise ValueError("Bits must be divisible by k.")
        B = bits.reshape(-1, self.P['k'])
        symbols = np.empty(B.shape[0], dtype=np.complex64)
        mapping = {
            (0, 0): 1 + 1j,
            (0, 1): -1 + 1j,
            (1, 1): -1 - 1j,
            (1, 0): 1 - 1j
        }
        for i in range(B.shape[0]):
            b1, b2 = B[i, 0], B[i, 1]
            symbols[i] = mapping[(b1, b2)]
        original = symbols.copy()
        normalized = symbols / np.max(np.abs(symbols))
        return normalized, original

    def generate_signal(self, bits: np.ndarray):
        """
        Generate QPSK symbols and return as a tuple.
        """
        Tx_s, OriginalSymbols = self._bits_to_symbols(bits)
        self.P['symbols'] = Tx_s
        self.P['OriginalSymbols'] = OriginalSymbols
        return None, Tx_s

# ======================= Channel =======================
class Channel:
    _channel_responses = {
        1: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        2: np.array([0.447, 0.894, 0.0], dtype=np.float32),
        3: np.array([0.209, 0.995, 0.209], dtype=np.float32),
        4: np.array([0.260, 0.930, 0.260], dtype=np.float32),
        5: np.array([0.304, 0.903, 0.304], dtype=np.float32),
        6: np.array([0.341, 0.876, 0.341], dtype=np.float32),
    }

    def __init__(self):
        pass

    def configure(self, choice, nl, awgn):
        if choice not in self._channel_responses:
            raise ValueError("Invalid channel choice. Must be 1-6.")
        self._choice = choice
        self._nl = nl
        self._awgn = awgn
        self._impulse_response = self._channel_responses[self._choice]

        print(f"Channel Impulse Response (choice={choice}): {self._impulse_response}")
        if nl == 0:
            print("Nonlinearity: Linear (NL=0)")
        elif nl == 1:
            print("Nonlinearity: tanh (NL=1)")
        elif nl == 2:
            print("Nonlinearity: Polynomial (NL=2)")
        elif nl == 3:
            print("Nonlinearity: Polynomial + cosine (NL=3)")
        else:
            raise ValueError("Invalid NL choice. Must be 0-3.")
        self._write_config()
        return self

    def _write_config(self):
        pass

    def apply_channel(self, tx_symbols, snr_db):
        if len(tx_symbols) == 0:
            raise ValueError("Transmitted symbols are empty.")

        rx_symbols = lfilter(
            self._impulse_response.astype(np.complex64),
            [1.0],
            tx_symbols.astype(np.complex64)
        )

        rx_symbols = np.asarray(rx_symbols, dtype=np.complex64)

        if self._nl == 0:
            rx_symbols_nl = rx_symbols
        elif self._nl == 1:
            rx_symbols_nl = np.tanh(rx_symbols)
            print("tanh nonlinearity")
        elif self._nl == 2:
            rx_symbols_nl = rx_symbols + 0.2 * rx_symbols**2 - 0.1 * rx_symbols**3
            print("Polynomial nonlinearity")
        elif self._nl == 3:
            rx_symbols_nl = rx_symbols + 0.2 * rx_symbols**2 - 0.1 * rx_symbols**3 + 0.5 * np.cos(np.pi * rx_symbols)
            print("Polynomial + cosine error")
        else:
            raise ValueError("Invalid NL choice. Must be 0-3.")

        if self._awgn:
            snr_linear = 10 ** (snr_db / 10)
            power = np.mean(np.abs(rx_symbols_nl) ** 2)
            noise_std = np.sqrt(power / (2 * snr_linear)) if power > 0 else 0.0
            noise = noise_std * (np.random.randn(len(rx_symbols_nl)) + 1j * np.random.randn(len(rx_symbols_nl)))
            rx_symbols_nl += noise.astype(np.complex64)

        self.rx_symbols_nl = rx_symbols_nl
        return self, rx_symbols_nl

    def get_impulse_response(self):
        return self._impulse_response.copy()

    def get_channel_choice(self):
        return self._choice

# ======================= RxDSP =======================
class RxDSP:
    """
    Receiver DSP module with optional linear equalizer (LMS)
    and decision-directed adaptation after training.
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self._format = "QPSK"
        self._ber = 0.0
        self._decoded_bits = np.array([], dtype=int)

        self._C = np.array([1+1j, -1+1j, -1-1j, 1-1j])
        self._B = np.array([[0,0],[0,1],[1,1],[1,0]])

        self._eq = None

    def configure(self, eq_type="FFE", **kwargs):
        if eq_type == "FFE":
            self._eq = LinearEqualizer(
                num_taps=kwargs.get("num_taps", 7),
                algorithm=kwargs.get("algorithm", "LMS"),
                step_size=kwargs.get("step_size", 0.01),
                reference_tap=kwargs.get("ref_tap", None),
                constellation=self._C
            )

    def normalize_signal(self, r_k):
        r_k = r_k - np.mean(r_k)
        return r_k / np.std(r_k)

    def _cross_correlation_align(self, r_k, t_k):
        corr = np.correlate(r_k, t_k, mode='full')
        delay = np.argmax(np.abs(corr)) - (len(t_k) - 1)
        if delay >= 0:
            r_aligned = r_k[delay:]
            t_aligned = t_k[:len(r_aligned)]
        else:
            r_aligned = r_k[-delay:]
            t_aligned = t_k[:len(r_aligned)]
        return r_aligned, t_aligned

    def _symbols_to_bits(self, yd):
        bits = []
        for sym in yd:
            idx = np.argmin(np.abs(sym - self._C))
            bits.extend(self._B[idx])
        return np.array(bits, dtype=int)

    def _min_distance_decision(self, r_k):
        yd = np.zeros_like(r_k, dtype=complex)
        for i, sym in enumerate(r_k):
            idx = np.argmin(np.abs(sym - self._C))
            yd[i] = self._C[idx]
        return yd

    def process_signal(self, t_k, r_k, P=None, train_len=None, disp_out=False):
        t_k = np.ravel(t_k)
        r_k = np.ravel(r_k)

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

        r_k = self.normalize_signal(r_k)
        r_aligned, t_aligned = self._cross_correlation_align(r_k, t_k)
        N = min(len(r_aligned), len(t_aligned))
        r_aligned, t_aligned = r_aligned[:N], t_aligned[:N]

        if self._eq is None:
            y = r_aligned
            mode = "No equalization"
        else:
            y, _ = self._eq.equalize(r_aligned, d=t_aligned, Ks=train_len)
            mode = "Training + Decision-Directed"

        yd = self._min_distance_decision(y)
        decoded_bits = self._symbols_to_bits(yd)
        t_bits = self._symbols_to_bits(t_aligned)

        min_len = min(len(decoded_bits), len(t_bits))
        ber = np.sum(np.abs(decoded_bits[:min_len] - t_bits[:min_len])) / min_len
        self._ber = float(ber)
        self._decoded_bits = decoded_bits

        if disp_out:
            print(f"[RxDSP] BER={self._ber:.4e}, Mode={mode}, Alg={self._eq.algorithm if self._eq else 'None'}")

        return self._ber, decoded_bits

# ======================= SysOrch =======================
class SysOrch:
    def __init__(self):
        """Initialize system components and simulation parameters."""
        self.tx_dsp = TxDSP()
        self.channel = Channel()
        self.rx_dsp = RxDSP()
        self._Nbits = 2**14
        self._ber_threshold = 1e-4
        self.P = {
            "mus": [0.01],
            "eq_type": "FFE",
            "disp_out": True,
            "taps": 6,
            "Ks": [1000],
            "methods": ["lms"],
            "C": [1+1j, -1+1j, -1-1j, 1-1j]
        }

    def run(self):
        """Run the simulation loop."""
        for choice in range(1, 2):
            for nl in range(0, 2):
                print(f"\n--- Running for choice={choice}, nl={nl} ---")
                self.tx_dsp.configure(self._Nbits)
                self.channel.configure(choice, nl, awgn=True)
                self.rx_dsp.configure(eq_type="FFE")
                bits_per_symbol = 2
                Eb_No_dB = np.arange(0, 16, 1, dtype=float)
                SNR_dB = Eb_No_dB + 10 * np.log10(bits_per_symbol)
                BER_results = []

                for snr_db in SNR_dB:
                    np.random.seed(0)
                    bits = np.random.randint(0, 2, self._Nbits, dtype=int)
                    _, tx_symbols = self.tx_dsp.generate_signal(bits)
                    _, rx_symbols = self.channel.apply_channel(tx_symbols, snr_db=snr_db)
                    ber_val, _ = self.rx_dsp.process_signal(tx_symbols, rx_symbols, self.P)
                    BER_results.append(ber_val)
                    print(f"QPSK SNR = {snr_db:.1f} dB, BER = {ber_val:.6e}")
                    if ber_val < self._ber_threshold:
                        print(f"Early stopping: BER < {self._ber_threshold:.1e} at SNR = {snr_db:.1f} dB")
                        break

                nl_type = {0: "Linear", 1: "tanh"}[nl]
                plt.figure()
                plt.semilogy(SNR_dB[:len(BER_results)], BER_results, 'o-', 
                             label=f'Simulated (Channel={choice}, NL={nl_type})')
                theory_ber = 0.5 * erfc(np.sqrt(10 ** (Eb_No_dB[:len(BER_results)] / 10)))
                plt.semilogy(SNR_dB[:len(BER_results)], theory_ber, '--', label='Theoretical')
                plt.grid(True, which="both", ls="--")
                plt.xlabel('SNR dB (Es/N0)')
                plt.ylabel('Bit Error Rate (BER)')
                plt.title(f'BER vs SNR (Channel={choice}, NL={nl_type})')
                plt.legend()
                plt.tight_layout()
                plt.show()


if __name__ == "__main__":
    orchestrator = SysOrch()
    orchestrator.run()
