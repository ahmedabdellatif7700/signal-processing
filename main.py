import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc


class BitSource:
    """A - Bit Source"""

    def __init__(self, bit_count=4000):
        self.bit_count = bit_count

    def generate_bits(self):
        return np.random.randint(0, 2, self.bit_count)


class SymbolMapping16QAM:
    """B - Symbol Mapping QAM"""

    def __init__(self):
        self._a = np.sqrt(1 / 10)  # Normalization factor for unit power
        self.bits_per_symbol = 4

    def map_bits_to_symbols(self, bits: np.ndarray) -> np.ndarray:
        if len(bits) % 4 != 0:
            raise ValueError("Bit length must be multiple of 4.")
        B = bits.reshape(4, -1)
        B1, B2, B3, B4 = B[0], B[1], B[2], B[3]
        symbols = self._a * (
            -2 * (B3 - 0.5) * (3 - 2 * B4) - 1j * 2 * (B1 - 0.5) * (3 - 2 * B2)
        )
        return symbols

    def demap_symbols_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        a = 1 / np.sqrt(10)
        B5 = (symbols.imag < 0).astype(int)
        B6 = ((symbols.imag < 2 * a) & (symbols.imag > -2 * a)).astype(int)
        B7 = (symbols.real < 0).astype(int)
        B8 = ((symbols.real < 2 * a) & (symbols.real > -2 * a)).astype(int)

        bits_est = np.vstack([B5, B6, B7, B8])
        return bits_est.reshape(-1, order="F")


class Upsampling:
    """C - Upsampling"""

    def process(self, symbols: np.ndarray) -> np.ndarray:
        # TODO: Implement actual upsampling
        return symbols


class PulseShapingFilter:
    """D - Pulse Shaping Filter"""

    def apply(self, signal: np.ndarray) -> np.ndarray:
        # TODO: Implement pulse shaping filter
        return signal


class TXAmplification:
    """TX Amplification + AC Coupling (Placeholder)"""

    def apply(self, signal: np.ndarray) -> np.ndarray:
        # TODO: Implement TX amplification and AC coupling
        return signal


class AWGNChannel:
    """E - AWGN Channel"""

    def __init__(self, snr_db: float):
        self.snr_db = snr_db
        self._N0 = 1 / (10 ** (snr_db / 10))

    def transmit(self, signal: np.ndarray) -> np.ndarray:
        noise = np.sqrt(self._N0 / 2) * (
            np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
        )
        return signal + noise


class Realignment:
    """F - Realignment"""

    def apply(self, signal: np.ndarray) -> np.ndarray:
        # TODO: Implement realignment
        return signal


class Resampling:
    """G - Resampling"""

    def process(self, signal: np.ndarray) -> np.ndarray:
        # TODO: Implement resampling
        return signal


class RXSetup:
    """RX Setup Params + Equalizer Initialization (Placeholder)"""

    def initialize(self):
        # TODO: Initialize RX parameters and equalizer
        pass


class Decision:
    """H - Decision"""

    def __init__(self):
        self.bits_per_symbol = 4

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        a = 1 / np.sqrt(10)
        B5 = (symbols.imag < 0).astype(int)
        B6 = ((symbols.imag < 2 * a) & (symbols.imag > -2 * a)).astype(int)
        B7 = (symbols.real < 0).astype(int)
        B8 = ((symbols.real < 2 * a) & (symbols.real > -2 * a)).astype(int)

        bits_est = np.vstack([B5, B6, B7, B8])
        return bits_est.reshape(-1, order="F")


class ErrorCounting:
    """I - Error Counting BER and SER"""

    def __init__(self, bits_per_symbol=4):
        self.bits_per_symbol = bits_per_symbol

    def compute_ber_ser(
        self, original_bits: np.ndarray, decoded_bits: np.ndarray
    ) -> tuple:
        if original_bits.shape != decoded_bits.shape:
            raise ValueError("Original and decoded bits shape mismatch.")

        total_bits = len(original_bits)
        bit_errors = np.sum(original_bits != decoded_bits)
        ber = bit_errors / total_bits

        total_symbols = total_bits // self.bits_per_symbol
        orig_sym = original_bits.reshape((total_symbols, self.bits_per_symbol))
        dec_sym = decoded_bits.reshape((total_symbols, self.bits_per_symbol))
        symbol_errors = np.sum(np.any(orig_sym != dec_sym, axis=1))
        ser = symbol_errors / total_symbols

        return ber, ser


def run_simulation():
    np.random.seed(0)
    bit_source = BitSource()
    symbol_mapping = SymbolMapping16QAM()
    upsampling = Upsampling()
    pulse_shaping_filter = PulseShapingFilter()
    tx_amplification = TXAmplification()         # Added TX Amplification placeholder
    realignment = Realignment()
    resampling = Resampling()
    rx_setup = RXSetup()                         # Added RX Setup placeholder
    rx_setup.initialize()
    decision = Decision()
    error_counting = ErrorCounting()

    Eb_No_dB = np.arange(-6, 11, 1)
    bits_per_symbol = 4
    SNR_dB = Eb_No_dB + 10 * np.log10(bits_per_symbol)
    BER = np.zeros(len(SNR_dB))

    for idx, snr_db in enumerate(SNR_dB):
        awgn_channel = AWGNChannel(snr_db)
        total_errors = 0
        orig_buffer = []
        decoded_buffer = []

        while total_errors < 100:
            bits = bit_source.generate_bits()               # A
            symbols = symbol_mapping.map_bits_to_symbols(bits)  # B
            symbols_up = upsampling.process(symbols)        # C
            shaped = pulse_shaping_filter.apply(symbols_up) # D
            amplified = tx_amplification.apply(shaped)      # TX Amplification + AC Coupling (TODO)
            received = awgn_channel.transmit(amplified)     # E
            realigned = realignment.apply(received)         # F
            resampled = resampling.process(realigned)       # G
            decoded_bits = decision.demodulate(resampled)   # H

            orig_bits = bits.reshape(4, -1).reshape(-1, order="F")

            errors = np.sum(orig_bits != decoded_bits)
            total_errors += errors

            orig_buffer.append(orig_bits)
            decoded_buffer.append(decoded_bits)

        all_orig_bits = np.concatenate(orig_buffer)
        all_decoded_bits = np.concatenate(decoded_buffer)
        BER[idx], _ = error_counting.compute_ber_ser(all_orig_bits, all_decoded_bits)
        print(f"SNR = {snr_db:.1f} dB, BER = {BER[idx]:.6e}")

    plt.figure()
    plt.semilogy(SNR_dB, BER, "or", label="Simulated")
    plt.grid(True)
    plt.title("BER vs SNR for 16-QAM in AWGN")
    plt.xlabel("SNR dB")
    plt.ylabel("Bit Error Rate BER")

    theory_ber = (3 / 8) * erfc(np.sqrt(0.4 * (10 ** (Eb_No_dB / 10))))
    plt.semilogy(SNR_dB, theory_ber, label="Theoretical")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_simulation()
