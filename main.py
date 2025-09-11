import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.signal import lfilter

# ======================= LinearEqualizer (LMS FFE only) =======================
class LinearEqualizer:
    """LMS-based Feedforward Equalizer (FFE) for QPSK."""
    def __init__(self, num_taps=7, step_size=0.005, ref_tap=None, constellation=None):
        self.num_taps = num_taps
        self.step_size = step_size
        self.reference_tap = num_taps // 2 if ref_tap is None else max(0, ref_tap-1)
        self.constellation = np.array(constellation) if constellation is not None else None
        self.reset()

    def reset(self):
        self.weights = np.zeros(self.num_taps, dtype=complex)
        self.weights[self.reference_tap] = 1 + 0j

    def _decision(self, y):
        y = np.atleast_1d(y)
        dists = np.abs(y.reshape(-1,1) - self.constellation.reshape(1,-1))
        idx = dists.argmin(axis=1)
        return self.constellation[idx]

    def equalize(self, x, d=None, Ks=None):
        x = np.asarray(x, dtype=complex).ravel()
        if d is not None: d = np.asarray(d, dtype=complex).ravel()
        N = len(x)
        pad = np.zeros(self.num_taps-1, dtype=complex)
        buffer = np.concatenate([pad, x])
        y_out = np.zeros(N, dtype=complex)
        e_out = np.zeros(N, dtype=complex)

        Ks = N if (Ks is None and d is not None) else (Ks or 0)

        for n in range(N):
            u = buffer[n:n+self.num_taps][::-1]  # newest sample first
            y = np.vdot(self.weights, u)

            # Compute error
            if d is not None and n < Ks:
                err = d[n] - y
            else:
                decided = self._decision(y)[0]
                err = decided - y

            # LMS update
            self.weights += self.step_size * np.conj(err) * u

            y_out[n] = y
            e_out[n] = err

        return y_out, e_out

# ======================= TxDSP =======================
class TxDSP:
    """QPSK transmitter."""
    def __init__(self):
        self.P = {}

    def configure(self, Nbits):
        self.P['Nbits'] = Nbits
        self.P['k'] = 2
        self.P['M'] = 4
        self.P['Nsym'] = Nbits // self.P['k']

    def _bits_to_symbols(self, bits):
        B = bits.reshape(-1, 2)
        mapping = {(0,0):1+1j, (0,1):-1+1j, (1,1):-1-1j, (1,0):1-1j}
        symbols = np.array([mapping[tuple(b)] for b in B], dtype=complex)
        return symbols / np.max(np.abs(symbols)), symbols

    def generate_signal(self, bits):
        tx, orig = self._bits_to_symbols(bits)
        self.P['symbols'] = tx
        self.P['OriginalSymbols'] = orig
        return None, tx

# ======================= Channel =======================
class Channel:
    """Linear/nonlinear channel with optional AWGN."""
    _responses = {1: np.array([1.0,0,0], dtype=float),
                  2: np.array([0.447,0.894,0], dtype=float)}

    def configure(self, choice, nl, awgn):
        self._choice = choice
        self._nl = nl
        self._awgn = awgn
        self._impulse_response = self._responses[choice]
        print(f"Channel configured: choice={choice}, NL={nl}, impulse_response={self._impulse_response}")
        return self

    def apply_channel(self, tx, snr_db):
        rx = lfilter(self._impulse_response, [1.0], tx)
        if self._nl==1: rx = np.tanh(rx)
        elif self._nl==2: rx = rx + 0.2*rx**2 - 0.1*rx**3
        if self._awgn:
            snr_lin = 10**(snr_db/10)
            noise_std = np.sqrt(np.mean(np.abs(rx)**2)/(2*snr_lin))
            rx += noise_std*(np.random.randn(len(rx))+1j*np.random.randn(len(rx)))
        return self, rx

# ======================= RxDSP =======================
class RxDSP:
    """Receiver DSP with optional LMS FFE equalizer."""
    def __init__(self):
        self._C = np.array([1+1j, -1+1j, -1-1j, 1-1j])
        self._B = np.array([[0,0],[0,1],[1,1],[1,0]])
        self._eq = None

    def configure(self, num_taps=7, step_size=0.005):
        self._eq = LinearEqualizer(num_taps=num_taps, step_size=step_size, constellation=self._C)

    def normalize_signal(self, r):
        return r - np.mean(r)  # remove DC only

    def _cross_correlation_align(self, r, t):
        corr = np.correlate(r, t, mode='full')
        delay = np.argmax(np.abs(corr)) - (len(t)-1)
        if delay >=0: return r[delay:], t[:len(r[delay:])]
        else: return r[-delay:], t[:len(r[-delay:])]

    def _symbols_to_bits(self, sym):
        bits=[]
        for s in sym: bits.extend(self._B[np.argmin(np.abs(s-self._C))])
        return np.array(bits, dtype=int)

    def _min_distance_decision(self, r):
        return np.array([self._C[np.argmin(np.abs(s-self._C))] for s in r])

    def process_signal(self, t, r, Ks=1000, disp_out=False):
        r = self.normalize_signal(np.ravel(r))
        r, t = self._cross_correlation_align(r, np.ravel(t))
        N = min(len(r), len(t))
        r, t = r[:N], t[:N]
        if self._eq:
            y, e = self._eq.equalize(r, d=t, Ks=Ks)
        else:
            y = r
            e = np.zeros_like(r)
        yd = self._min_distance_decision(y)
        decoded_bits = self._symbols_to_bits(yd)
        t_bits = self._symbols_to_bits(t)
        min_len = min(len(decoded_bits), len(t_bits))
        ber = np.sum(np.abs(decoded_bits[:min_len]-t_bits[:min_len])) / min_len
        if disp_out: print(f"BER={ber:.4e}")
        return ber, decoded_bits

# ======================= Simulation Orchestrator =======================
class SysOrch:
    """Run full QPSK simulation with LMS FFE or without equalization."""
    def __init__(self):
        self.tx_dsp = TxDSP()
        self.channel = Channel()
        self.rx_dsp = RxDSP()
        self.Nbits = 2**14
        self.ber_threshold = 1e-4

    def run(self):
        self.tx_dsp.configure(self.Nbits)
        EbNo_dB = np.arange(0, 16, 1)
        SNR_dB = EbNo_dB + 10*np.log10(2)

        for choice in [2]:  # choose channel
            for nl in [0, 1]:  # linear and tanh
                print(f"\n--- Channel choice={choice}, NL={nl} ---")
                BER_dict = {"NoEq": [], "FFE": []}

                for eq_type in ["NoEq", "FFE"]:
                    print(f"\nRunning with equalizer: {eq_type}")
                    if eq_type=="FFE":
                        self.rx_dsp.configure(num_taps=7, step_size=0.005)
                    else:
                        self.rx_dsp._eq = None

                    for snr in SNR_dB:
                        bits = np.random.randint(0,2,self.Nbits)
                        _, tx_symbols = self.tx_dsp.generate_signal(bits)
                        self.channel.configure(choice=choice, nl=nl, awgn=True)
                        _, rx_symbols = self.channel.apply_channel(tx_symbols, snr)

                        ber, _ = self.rx_dsp.process_signal(tx_symbols, rx_symbols, Ks=1000, disp_out=True)
                        BER_dict[eq_type].append(ber)

                        print(f"SNR={snr:.1f} dB, BER={ber:.6e}")
                        if ber < self.ber_threshold:
                            print(f"Early stopping: BER<{self.ber_threshold} at SNR={snr:.1f} dB")
                            break

                # Plot FFE vs NoEq
                plt.figure()
                for eq_type, BER_values in BER_dict.items():
                    plt.semilogy(SNR_dB[:len(BER_values)], BER_values, 'o-', label=eq_type)
                theory_ber = 0.5*erfc(np.sqrt(10**(EbNo_dB[:len(BER_values)]/10)))
                plt.semilogy(SNR_dB[:len(BER_values)], theory_ber, '--', label='Theoretical BER')
                plt.grid(True, which='both', ls='--')
                plt.xlabel("SNR dB")
                plt.ylabel("Bit Error Rate (BER)")
                plt.ylim(bottom=1e-5)  # <-- fix y-axis lower limit
                nl_type = {0:"Linear",1:"tanh"}[nl]
                plt.title(f"BER vs SNR (Channel={choice}, NL={nl_type})")
                plt.legend(); plt.tight_layout(); plt.show()


if __name__=="__main__":
    sim = SysOrch()
    sim.run()
