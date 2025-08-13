import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc


class Parameters:
    def __init__(self):
        self.SNR = None
        self.TxSignal = None
        self.RxSignal = None
        self.BER = None

    def set_param(self, name, value):
        setattr(self, name, value)

    def get_param(self, name):
        return getattr(self, name)


class ConfigETx:
    def configure_tx(self, params: Parameters, bit_count=4000):
        self._bit_count = bit_count
        params.set_param("SNR", None)
        return self._bit_count


class TxDSP:
    """Correct 16-QAM mapping (Gray-coded)"""
    def __init__(self):
        self.M = 16
        self.k = 4  # bits per symbol
        self.a = 1 / np.sqrt(10)  # normalization for unit average power

    def generate_signal(self, params: Parameters, bits: np.ndarray):
        if len(bits) % self.k != 0:
            raise ValueError("Number of bits must be a multiple of 4")

        B = bits.reshape(-1, 4)
        mapping = {
            (0, 0): -3,
            (0, 1): -1,
            (1, 1): 1,
            (1, 0): 3
        }

        I = np.array([mapping[tuple(b[:2])] for b in B])
        Q = np.array([mapping[tuple(b[2:])] for b in B])
        symbols = self.a * (I + 1j * Q)
        params.set_param("TxSignal", symbols)


class Channel:
    def add_noise(self, params: Parameters, snr_db: float):
        tx = params.get_param("TxSignal")
        snr_linear = 10 ** (snr_db / 10)
        Es = np.mean(np.abs(tx) ** 2)
        N0 = Es / snr_linear
        noise = np.sqrt(N0 / 2) * (np.random.randn(len(tx)) + 1j * np.random.randn(len(tx)))
        rx = tx + noise
        params.set_param("RxSignal", rx)


class ConfigERx:
    def configure_rx(self, params: Parameters):
        pass


class RxDSP:
    """Correct 16-QAM demapping"""
    def __init__(self):
        self.a = 1 / np.sqrt(10)

    def process_signal(self, params: Parameters, original_bits: np.ndarray):
        rx = params.get_param("RxSignal")

        def decision(x):
            if x < -2 * self.a:
                return (0, 0)
            elif x < 0:
                return (0, 1)
            elif x < 2 * self.a:
                return (1, 1)
            else:
                return (1, 0)

        decoded_bits = []
        for sym in rx:
            i_bits = decision(sym.real)
            q_bits = decision(sym.imag)
            decoded_bits.extend(i_bits + q_bits)

        decoded_bits = np.array(decoded_bits)
        ber = np.sum(decoded_bits != original_bits) / len(original_bits)
        params.set_param("BER", ber)


class Orchestrator:
    def run(self):
        np.random.seed(0)
        params = Parameters()
        config_tx = ConfigETx()
        tx_dsp = TxDSP()
        channel = Channel()
        config_rx = ConfigERx()
        rx_dsp = RxDSP()

        Eb_No_dB = np.arange(-6, 11, 1)
        bits_per_symbol = 4
        SNR_dB = Eb_No_dB + 10 * np.log10(bits_per_symbol)
        BER = []

        for snr_db in SNR_dB:
            bit_count = config_tx.configure_tx(params)
            bits = np.random.randint(0, 2, bit_count)

            tx_dsp.generate_signal(params, bits)
            channel.add_noise(params, snr_db)
            config_rx.configure_rx(params)
            rx_dsp.process_signal(params, bits)

            BER.append(params.get_param("BER"))
            print(f"SNR = {snr_db:.1f} dB, BER = {params.get_param('BER'):.6e}")

        plt.figure()
        plt.semilogy(SNR_dB, BER, 'or', label='Simulated')
        plt.grid(True)
        plt.title('BER vs SNR for 16-QAM in AWGN')
        plt.xlabel('SNR dB')
        plt.ylabel('Bit Error Rate (BER)')

        theory_ber = (3 / 8) * erfc(np.sqrt(0.4 * 10 ** (Eb_No_dB / 10)))
        plt.semilogy(SNR_dB, theory_ber, label='Theoretical')
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    orchestrator = Orchestrator()
    orchestrator.run()
