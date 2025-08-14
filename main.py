import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from typing import Any


class Parameters:
    def __init__(self):
        self.SNR: float = 0.0                                   # dB, interpreted as Es/N0 (symbol SNR)
        self.TxSignal: np.ndarray = np.empty(0, dtype=np.complex128)  # complex baseband symbols
        self.RxSignal: np.ndarray = np.empty(0, dtype=np.complex128)  # noisy received symbols
        self.BER: float = 0.0                                   # bit error rate

    def set_param(self, name: str, value: Any) -> None:
        setattr(self, name, value)

    def get_param(self, name: str) -> Any:
        return getattr(self, name)


class ConfigETx:
    def __init__(self):
        self._bit_count: int = 0

    def configure_tx(self, params: Parameters, bit_count: int = 4000) -> int:
        self._bit_count = bit_count
        # (Optionally) initialize SNR here; the orchestrator will overwrite per sweep value
        params.set_param("SNR", 0.0)
        return self._bit_count


class TxDSP:
    """Correct 16-QAM mapping (Gray-coded)"""
    def __init__(self):
        self.M: int = 16
        self.k: int = 4                                # bits per symbol
        self.a: float = 1.0 / np.sqrt(10.0)            # normalization for unit average symbol power

    def generate_signal(self, params: Parameters, bits: np.ndarray) -> None:
        if (bits.size % self.k) != 0:
            raise ValueError("Number of bits must be a multiple of 4")

        B: np.ndarray = bits.reshape(-1, 4)
        mapping = {
            (0, 0): -3,
            (0, 1): -1,
            (1, 1):  1,
            (1, 0):  3
        }

        I: np.ndarray = np.array([mapping[tuple(b[:2])] for b in B], dtype=float)
        Q: np.ndarray = np.array([mapping[tuple(b[2:])] for b in B], dtype=float)
        symbols: np.ndarray = self.a * (I + 1j * Q)
        params.set_param("TxSignal", symbols)


class Channel:
    def add_noise(self, params: Parameters) -> None:
        snr_db: float = float(params.get_param("SNR"))          # Es/N0 in dB
        tx: np.ndarray = params.get_param("TxSignal")

        snr_linear: float = 10.0 ** (snr_db / 10.0)             # Es/N0 (linear)
        Es: float = float(np.mean(np.abs(tx) ** 2))
        N0: float = Es / snr_linear
        noise: np.ndarray = np.sqrt(N0 / 2.0) * (
            np.random.randn(tx.size) + 1j * np.random.randn(tx.size)
        )
        rx: np.ndarray = tx + noise
        params.set_param("RxSignal", rx)


class ConfigERx:
    def configure_rx(self, params: Parameters) -> None:
        # Placeholder: in a fuller model, set RX gain/EQ params in Parameters.
        # Keeping as a "uses Parameters" step to reflect the UML.
        _snr_check: float = float(params.get_param("SNR"))  # read (no write needed here)
        return


class RxDSP:
    """Correct 16-QAM demapping"""
    def __init__(self):
        self.a: float = 1.0 / np.sqrt(10.0)

    def process_signal(self, params: Parameters, original_bits: np.ndarray) -> None:
        rx: np.ndarray = params.get_param("RxSignal")

        def decision(x: float) -> tuple[int, int]:
            if x < -2.0 * self.a:
                return (0, 0)
            elif x < 0.0:
                return (0, 1)
            elif x < 2.0 * self.a:
                return (1, 1)
            else:
                return (1, 0)

        decoded_bits_list: list[int] = []
        for sym in rx:
            i_bits = decision(float(sym.real))
            q_bits = decision(float(sym.imag))
            decoded_bits_list.extend(i_bits + q_bits)

        decoded_bits: np.ndarray = np.array(decoded_bits_list, dtype=int)
        ber: float = float(np.sum(decoded_bits != original_bits) / original_bits.size)
        params.set_param("BER", ber)


class Orchestrator:
    def run(self) -> None:
        np.random.seed(0)
        params: Parameters = Parameters()

        # Instantiate blocks
        config_tx: ConfigETx = ConfigETx()
        tx_dsp: TxDSP = TxDSP()
        channel: Channel = Channel()
        config_rx: ConfigERx = ConfigERx()
        rx_dsp: RxDSP = RxDSP()

        # Sweep
        Eb_No_dB: np.ndarray = np.arange(-6, 11, 1, dtype=float)
        bits_per_symbol: int = 4
        SNR_dB: np.ndarray = Eb_No_dB + 10.0 * np.log10(bits_per_symbol)  # Es/N0 in dB
        BER_results: list[float] = []

        for snr_db in SNR_dB:
            bit_count: int = config_tx.configure_tx(params)
            bits: np.ndarray = np.random.randint(0, 2, bit_count, dtype=int)

            params.set_param("SNR", float(snr_db))
            tx_dsp.generate_signal(params, bits)
            channel.add_noise(params)
            config_rx.configure_rx(params)
            rx_dsp.process_signal(params, bits)

            ber_val: float = float(params.get_param("BER"))
            BER_results.append(ber_val)
            print(f"SNR = {snr_db:.1f} dB, BER = {ber_val:.6e}")

        # Plot results
        plt.figure()
        plt.semilogy(SNR_dB, BER_results, 'or', label='Simulated')
        plt.grid(True)
        plt.title('BER vs SNR for 16-QAM in AWGN')
        plt.xlabel('SNR dB (Es/N0)')
        plt.ylabel('Bit Error Rate (BER)')

        # Theoretical 16-QAM BER (Gray-coded, AWGN) approximation using Eb/N0
        theory_ber: np.ndarray = (3.0 / 8.0) * erfc(np.sqrt(0.4 * 10.0 ** (Eb_No_dB / 10.0)))
        plt.semilogy(SNR_dB, theory_ber, label='Theoretical')
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    orchestrator = Orchestrator()
    orchestrator.run()
