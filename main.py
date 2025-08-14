import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from typing import Any


# ----------------------------
# Data storage class
# ----------------------------
class Parameters:
    def __init__(self):
        self.SNR: float = 0.0  # dB, Es/N0
        self.TxSignal: np.ndarray = np.empty(0, dtype=np.complex128)
        self.RxSignal: np.ndarray = np.empty(0, dtype=np.complex128)
        self.BER: float = 0.0  # Bit Error Rate

    def set_param(self, name: str, value: Any) -> None:
        setattr(self, name, value)

    def get_param(self, name: str) -> Any:
        return getattr(self, name)


# ----------------------------
# Transmitter configuration
# ----------------------------
class ConfigETx:
    def __init__(self):
        self._bit_count: int = 0

    def configure_tx(self, params: Parameters, bit_count: int = 4000) -> int:
        self._bit_count = bit_count
        params.set_param("SNR", 0.0)
        return self._bit_count


# ----------------------------
# Transmitter DSP (QPSK only)
# ----------------------------
class TxDSP:
    def __init__(self):
        self.M: int = 4
        self.k: int = int(np.log2(self.M))  # bits per symbol
        self.a: float = 1 / np.sqrt(2)

    def generate_signal(self, params: Parameters, bits: np.ndarray) -> None:
        if bits.size % self.k != 0:
            raise ValueError(f"Number of bits must be multiple of {self.k}")

        B = bits.reshape(-1, self.k)
        mapping = {
            (0, 0): 1 + 1j,
            (0, 1): -1 + 1j,
            (1, 1): -1 - 1j,
            (1, 0): 1 - 1j
        }
        symbols = self.a * np.array([mapping[tuple(b)] for b in B], dtype=np.complex128)
        params.set_param("TxSignal", symbols)


# ----------------------------
# Channel
# ----------------------------
class Channel:
    def add_noise(self, params: Parameters) -> None:
        snr_db: float = float(params.get_param("SNR"))
        tx: np.ndarray = params.get_param("TxSignal")

        snr_linear: float = 10 ** (snr_db / 10)
        Es: float = float(np.mean(np.abs(tx) ** 2))
        N0: float = Es / snr_linear
        noise: np.ndarray = np.sqrt(N0 / 2) * (
            np.random.randn(tx.size) + 1j * np.random.randn(tx.size)
        )
        rx: np.ndarray = tx + noise
        params.set_param("RxSignal", rx)


# ----------------------------
# Receiver configuration
# ----------------------------
class ConfigERx:
    def configure_rx(self, params: Parameters) -> None:
        _snr_check: float = float(params.get_param("SNR"))  # read SNR (optional)
        return


# ----------------------------
# Receiver DSP (QPSK only)
# ----------------------------
class RxDSP:
    def __init__(self):
        self.a: float = 1 / np.sqrt(2)

    def process_signal(self, params: Parameters, original_bits: np.ndarray) -> None:
        rx: np.ndarray = params.get_param("RxSignal")
        decoded_bits_list: list[int] = []

        def decision_qpsk(sym: complex) -> tuple[int, int]:
            if sym.real > 0 and sym.imag > 0:
                return (0, 0)
            elif sym.real < 0 and sym.imag > 0:
                return (0, 1)
            elif sym.real < 0 and sym.imag < 0:
                return (1, 1)
            else:
                return (1, 0)

        for sym in rx:
            decoded_bits_list.extend(decision_qpsk(sym))

        decoded_bits = np.array(decoded_bits_list, dtype=int)
        ber: float = float(np.sum(decoded_bits != original_bits) / original_bits.size)
        params.set_param("BER", ber)


# ----------------------------
# Orchestrator
# ----------------------------
class Orchestrator:
    def run(self) -> None:
        np.random.seed(0)
        params = Parameters()
        config_tx = ConfigETx()
        tx_dsp = TxDSP()
        channel = Channel()
        config_rx = ConfigERx()
        rx_dsp = RxDSP()

        bits_per_symbol = tx_dsp.k
        Eb_No_dB = np.arange(-6, 11, 1, dtype=float)
        SNR_dB = Eb_No_dB + 10 * np.log10(bits_per_symbol)
        BER_results: list[float] = []

        for snr_db in SNR_dB:
            bit_count = config_tx.configure_tx(params, bit_count=4000)
            bits = np.random.randint(0, 2, bit_count, dtype=int)

            params.set_param("SNR", float(snr_db))
            tx_dsp.generate_signal(params, bits)
            channel.add_noise(params)
            config_rx.configure_rx(params)
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
