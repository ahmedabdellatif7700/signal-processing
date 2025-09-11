import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from TxDSP import TxDSP
from RxDSP import RxDSP
from Channel import Channel

class SysOrch:
    def __init__(self):
        """Initialize system components and simulation parameters."""
        # -----------------------------
        # Tx, Channel, Rx modules
        # -----------------------------
        self.tx_dsp = TxDSP()
        self.channel = Channel()
        self.rx_dsp = RxDSP()
        # Simulation parameters
        self._Nbits = 2**14  # Total bits per SNR
        self._ber_threshold = 1e-7  # Early stopping threshold
        # -----------------------------
        # Initialize P for RxDSP
        # -----------------------------
        self.P = {
            "mus": [0.01],            # Step size for equalizer
            "eq_type": "None",          # Equalizer type
            "disp_out": True,          # Display output
            "taps": 6,                 # Number of equalizer taps
            "Ks": [1000],              # Training symbols
            "methods": ["lms"],        # Equalizer algorithm
            "C": [1+1j, -1+1j, -1-1j, 1-1j]  # QPSK constellation
        }

    def run(self):
        """Run the simulation loop."""
        for choice in range(1, 2):  # Channel type
            for nl in range(0, 2):  # Nonlinearity type
                print(f"\n--- Running for choice={choice}, nl={nl} ---")
                # -----------------------------
                # Configuration: Tx → Channel → Rx
                # -----------------------------
                self.tx_dsp.configure(self._Nbits)
                self.channel.configure(choice, nl, awgn=True)
                self.rx_dsp.configure(eq_type="FFE")
                # -----------------------------
                # SNR setup
                # -----------------------------
                bits_per_symbol = 2  # QPSK
                Eb_No_dB = np.arange(0, 16, 1, dtype=float)
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
                    np.random.seed(0)
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
                    ber_val, _ = self.rx_dsp.process_signal(tx_symbols, rx_symbols, self.P)
                    BER_results.append(ber_val)
                    print(f"QPSK SNR = {snr_db:.1f} dB, BER = {ber_val:.6e}")
                    # -------------------------
                    # Early stopping check
                    # -------------------------
                    if ber_val < self._ber_threshold:
                        print(f"Early stopping: BER < {self._ber_threshold:.1e} at SNR = {snr_db:.1f} dB")
                        break
                # -----------------------------
                # Plot BER for this configuration
                # -----------------------------
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
