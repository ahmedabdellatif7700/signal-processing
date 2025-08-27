import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from TxDSP import TxDSP
from RxDSP import RxDSP
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
        self.rx_dsp = RxDSP()

        # Simulation parameters
        self._Nbits = 2**14  # Total bits per SNR
        self._ber_threshold = 1e-4 # Early stopping threshold

    # -------------------------
    # Run simulation
    # -------------------------
    def run(self):
         # Loop over all channel choices and nonlinearities
        for choice in range(1, 7):  # 1-6
            for nl in range(0, 4):  # 0-3
                print(f"\n--- Running for choice={choice}, nl={nl} ---")
                # -----------------------------
                # Configuration: Tx → Channel → Rx
                # -----------------------------
                self.tx_dsp.configure(self._Nbits)           # Configure transmitter
                self.channel.configure(choice, nl, awgn=True)  # Configure channel
                self.rx_dsp.configure()              # Configure receiver

                # -----------------------------
                # SNR setup
                # -----------------------------
                bits_per_symbol = 2  # QPSK
                Eb_No_dB = np.arange( 0, 16, 1, dtype=float)
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
                    choice = self.channel.get_channel_choice()
                    _, ber_val = self.rx_dsp.process_signal(tx_symbols, rx_symbols, choice)
                    BER_results.append(ber_val)
                    print(f"QPSK SNR = {snr_db:.1f} dB, BER = {ber_val:.6e}")

                    # -------------------------
                    # Early stopping check
                    # -------------------------
                    if ber_val < self._ber_threshold:
                        print(f"Early stopping: BER < {self._ber_threshold:.1e} at SNR = {snr_db:.1f} dB")
                        break  # Stop simulation for higher SNRs

                # -----------------------------
                # Plot BER for this configuration
                # -----------------------------
                nl_type = {
                    0: "Linear",
                    1: "tanh",
                    2: "Polynomial",
                    3: "Polynomial + cosine"
                }[nl]

                # Define a flag to control plotting behavior
                hold_on = True  # Set to True to plot all NLs for a channel in one figure

                if not hold_on or nl == 0:  # Create a new figure for each channel or if hold_on is False
                    plt.figure()

                # Plot simulated BER
                plt.semilogy(
                    SNR_dB[:len(BER_results)],
                    BER_results,
                    'o-',  # Use line + markers for clarity
                    label=f'Simulated (Channel={choice}, NL={nl_type})'
                )

                # Plot theoretical BER
                theory_ber = 0.5 * erfc(np.sqrt(10 ** (Eb_No_dB[:len(BER_results)] / 10)))
                plt.semilogy(
                    SNR_dB[:len(BER_results)],
                    theory_ber,
                    '--',  # Dashed line for theoretical
                    label='Theoretical'
                )

                plt.grid(True, which="both", ls="--")  # Enable grid for both major and minor ticks
                plt.xlabel('SNR dB (Es/N0)')
                plt.ylabel('Bit Error Rate (BER)')

                if hold_on:
                    plt.title(f'BER vs SNR (Channel={choice})')  # One title for all NLs in this channel
                else:
                    plt.title(f'BER vs SNR (Channel={choice}, NL={nl_type})')  # Separate title for each NL

                plt.legend()
                plt.tight_layout()

                if not hold_on or nl == 3:  # Show plot at the end of the inner loop if hold_on is False, or after last NL
                    plt.show()
# -----------------------------
# Run simulation
# -----------------------------
if __name__ == "__main__":
    orchestrator = SysOrch()
    orchestrator.run()
