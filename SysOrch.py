import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from TxDSP import TxDSP
from RxDSP import Rx
from Channel import Channel

# -------------------------
# System Orchestrator Class
# -------------------------
class SysOrch:
    def __init__(self):
        """Initialize system components and simulation parameters."""
        # -----------------------------
        # Tx, Channel, Rx modules
        # -----------------------------
        self.tx_dsp = TxDSP()
        self.channel = Channel()
        self.rx_dsp = Rx()

        # Simulation parameters
        self._Nbits = 2**14  # Total bits per SNR
        self._ber_threshold = 1e-3  # Early stopping threshold

    # -------------------------
    # Run simulation
    # -------------------------
    def run(self):
        # -----------------------------
        # Configuration: Tx → Channel → Rx
        # -----------------------------
        self.tx_dsp.configure(self._Nbits)           # Configure transmitter
        self.channel.configure(choice=1, nl= 0, awgn=True)  # Configure channel
        self.rx_dsp.configure()              # Configure receiver

        # -----------------------------
        # SNR setup
        # -----------------------------
        bits_per_symbol = 2  # QPSK
        Eb_No_dB = np.arange(-6, 11, 1, dtype=float)
        SNR_dB = Eb_No_dB + 10 * np.log10(bits_per_symbol)

        # Store BER results
        BER_results = []

        # -----------------------------
        # Main loop: Tx → Channel → Rx for each SNR
        # -----------------------------
        for snr_db in SNR_dB:
            # -------------------------
            # Generate random bits
            # -------------------------
            np.random.seed(0)  # Reproducible results
            bits = np.random.randint(0, 2, self._Nbits, dtype=int)

            # -------------------------
            # Transmitter DSP
            # -------------------------
            _, tx_symbols = self.tx_dsp.generate_signal(bits)

            # -------------------------
            # Channel: add noise (AWGN)
            # -------------------------
            _, rx_symbols = self.channel.apply_channel(tx_symbols, snr_db=snr_db)

            # -------------------------
            # Receiver DSP: decode & compute BER
            # -------------------------
            channel_taps = self.channel.get_impulse_response()
            _, ber_val = self.rx_dsp.process_signal(bits, rx_symbols, channel_taps)
            BER_results.append(ber_val)
            print(f"QPSK SNR = {snr_db:.1f} dB, BER = {ber_val:.6e}")

            # -------------------------
            # Early stopping check
            # -------------------------
            if ber_val < self._ber_threshold:
                print(f"Early stopping: BER < {self._ber_threshold:.1e} at SNR = {snr_db:.1f} dB")
                break  # Stop simulation for higher SNRs

        # -----------------------------
        # Post-processing: plot BER
        # -----------------------------
        plt.figure()
        plt.semilogy(SNR_dB[:len(BER_results)], BER_results, 'or', label='Simulated')
        plt.grid(True)
        plt.xlabel('SNR dB (Es/N0)')
        plt.ylabel('Bit Error Rate (BER)')
        plt.title('BER vs SNR for QPSK in AWGN')

        # Theoretical BER for QPSK in AWGN
        theory_ber = 0.5 * erfc(np.sqrt(10 ** (Eb_No_dB[:len(BER_results)] / 10)))
        plt.semilogy(SNR_dB[:len(BER_results)], theory_ber, label='Theoretical')

        plt.legend()
        plt.tight_layout()
        plt.show()


# -----------------------------
# Run simulation
# -----------------------------
if __name__ == "__main__":
    orchestrator = SysOrch()
    orchestrator.run()
