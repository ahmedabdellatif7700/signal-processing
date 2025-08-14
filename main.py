import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from typing import Any


# ----------------------------
# Data storage class
# ----------------------------
class Parameters:
    def __init__(self):
        # Main simulation parameters
        self.format: int = 1                  # Modulation format
        self.Rb: float = 100e9                # Bit rate [bits/s]
        self.fs: float = 8 * self.Rb          # Sampling frequency [Hz]

        # Modulation setup
        self.FormatStr: str = "QAM-4"
        self.M: int = 4                          # Modulation order
        self.k: int = int(np.log2(self.M))       # Bits per symbol
        self.a: float = 1.0                      # Normalization constant
        self.Boundaries: np.ndarray = np.empty(0, dtype=np.complex64)  # Will be set by ConfigERx

        # Bit & symbol counts
        self.Nbits: int = 2**15
        self.Nsym: int = self.Nbits // self.k

        # Timing parameters
        self.Rs: float = self.Rb / self.k
        self.Ts: float = 1.0 / self.Rs
        self.ts: float = 1.0 / self.fs
        self.sps: float = self.fs / self.Rs

        # Simulation state
        self.t_k: np.ndarray = np.empty(0, dtype=np.complex64)  # Tx symbols
        self.r_k: np.ndarray = np.empty(0, dtype=np.complex64)  # Rx symbols
        self.SNR: float = 0.0
        self.BER: float = 0.0

        # Channel impulse responses
        ch1 = np.array([1.0, 0.0, 0.0], dtype=float)
        ch2 = np.array([0.447, 0.894, 0.0], dtype=float)
        ch3 = np.array([0.209, 0.995, 0.209], dtype=float)
        ch4 = np.array([0.260, 0.930, 0.260], dtype=float)
        ch5 = np.array([0.304, 0.903, 0.304], dtype=float)
        ch6 = np.array([0.341, 0.876, 0.341], dtype=float)

        channel_choice: int = 1
        self.h: np.ndarray = {1: ch1, 2: ch2, 3: ch3, 4: ch4, 5: ch5, 6: ch6}[channel_choice]

        print(f"Selected Channel CH{channel_choice} impulse response h: {self.h}")

    def set_param(self, name: str, value: Any) -> None:
        setattr(self, name, value)

    def get_param(self, name: str) -> Any:
        return getattr(self, name)

# ----------------------------
# Transmitter configuration
# ----------------------------
class ConfigETx:
    def __init__(self, params: Parameters, format: int = 1):
        """
        Configure transmitter modulation format.
        format: 1 = QPSK (only case supported for now)
        """
        self.format: int = format
        self.params = params
        self.configure_modulation()

    def configure_modulation(self) -> None:
        if self.format == 1:  # QPSK
            self.params.M = 4
            self.params.k = int(np.log2(self.params.M))
            self.params.a = 1 / np.sqrt(2)

            # Compute derived parameters
            self.params.Nsym = self.params.Nbits // self.params.k
        else:
            pass

# ----------------------------
# Transmitter DSP (QPSK only)
# ----------------------------
class TxDSP:
    def __init__(self):
        pass  # 'a' already in Parameters

    def generate_signal(self, params: Parameters, bits: np.ndarray) -> np.ndarray:
        k: int = params.k
        a: float = params.a

        if bits.size % k != 0:
            pass
        # QPSK Mapping (Gray coding)
        B = bits.reshape(-1, k)
        mapping = {
            (0, 0): 1 + 1j,
            (0, 1): -1 + 1j,
            (1, 1): -1 - 1j,
            (1, 0): 1 - 1j
        }
        self._symbols: np.ndarray = a * np.array([mapping[tuple(b)] for b in B], dtype=np.complex64)

        # Save to Parameters
        params.set_param("t_k", self._symbols)

        # Return symbols
        return self._symbols

class Channel:
    """Simple AWGN channel"""
    def add_noise(self, params: Parameters, rx_symbols:np.ndarray) -> np.ndarray:
     
        snr_linear = 10 ** (params.SNR / 10)
        power = np.mean(np.abs(rx_symbols)**2)
        noise_std = np.sqrt(power / (2 * snr_linear))
        noise = noise_std * (np.random.randn(*rx_symbols.shape) + 1j*np.random.randn(*rx_symbols.shape))
        
        self._r_k = rx_symbols + noise
        # Store in parameters
        params.r_k = rx_symbols + noise
        return self._r_k
# ----------------------------
# Receiver configuration
# ----------------------------
class ConfigERx:
    def __init__(self, params: Parameters, format: int = 1):
        """
        Configure receiver for the given modulation format.
        For QPSK, store decision boundaries in Parameters.
        """
        self.params = params
        self.format: int = format
        self.configure_rx()

    def configure_rx(self) -> None:
        if self.format == 1:  # QPSK
            # Define decision boundaries as a proper def
            def boundaries(sym: complex) -> tuple[int, int]:
                if sym.real > 0 and sym.imag > 0:
                    return (0, 0)
                elif sym.real < 0 and sym.imag > 0:
                    return (0, 1)
                elif sym.real < 0 and sym.imag < 0:
                    return (1, 1)
                else:
                    return (1, 0)

            # Store in Parameters
            self.params.set_param("Boundaries", boundaries)
        else:
            raise ValueError("Only QPSK (format=1) supported")


# ----------------------------
# Receiver DSP (QPSK only)
# ----------------------------
class RxDSP:
    def __init__(self):
        pass  # 'a' already in Parameters

    def process_signal(self, params: Parameters, original_bits: np.ndarray) -> None:
        rx: np.ndarray = params.r_k  # Rx symbols
        decoded_bits_list: list[int] = []

        # Retrieve decision boundaries
        boundaries = params.get_param("Boundaries")

        # Decode symbols
        for sym in rx:
            decoded_bits_list.extend(boundaries(sym))

        # Convert to numpy array
        decoded_bits: np.ndarray = np.array(decoded_bits_list, dtype=int)

        # Compute BER
        ber: float = float(np.sum(decoded_bits != original_bits) / original_bits.size)
        params.set_param("BER", ber)


class Orchestrator:
    def run(self) -> None:
        np.random.seed(0)
        params = Parameters()
        ConfigETx(params)
        tx_dsp = TxDSP()
        channel = Channel()
        ConfigERx(params)
        rx_dsp = RxDSP()

        bits_per_symbol = params.k
        Eb_No_dB = np.arange(-6, 11, 1, dtype=float)
        SNR_dB = Eb_No_dB + 10 * np.log10(bits_per_symbol)
        BER_results: list[float] = []

        for snr_db in SNR_dB:
            # User Nbits from params
            Nbits = params.get_param("Nbits")
            bits = np.random.randint(0, 2, Nbits, dtype=int)

            t_k = tx_dsp.generate_signal(params,bits)
            params.set_param("SNR", float(snr_db))

            _ = channel.add_noise(params,t_k)
            # Boundaries already set in ConfigERx __init__
            rx_dsp.process_signal(params, bits)

            ber_val: float = float(params.get_param("BER"))
            BER_results.append(ber_val)
            print(f"QPSK SNR = {snr_db:.1f} dB, BER = {ber_val:.6e}")

        # Plot
        plt.figure()
        plt.semilogy(SNR_dB, BER_results, 'or', label='Simulated')
        plt.grid(True)
        plt.xlabel('SNR dB (Es/N0)')
        plt.ylabel('Bit Error Rate (BER)')
        plt.title('BER vs SNR for QPSK in AWGN')

        # Theoretical BER
        theory_ber = 0.5 * erfc(np.sqrt(10 ** (Eb_No_dB / 10)))
        plt.semilogy(SNR_dB, theory_ber, label='Theoretical')

        plt.legend()
        plt.tight_layout()
        plt.show()

# ----------------------------
# Run simulations (QPSK only)
# ----------------------------
if __name__ == "__main__":
    orchestrator = Orchestrator()
    orchestrator.run()
