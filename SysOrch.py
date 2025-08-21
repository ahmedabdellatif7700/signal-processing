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
        self.tx_dsp = TxDSP()
        self.channel = Channel()
        self.rx_dsp = Rx()

        self._Nbits = 2**14


    def run(self):
        # Configure Tx channel and Rx
        self.tx_dsp.configure(self._Nbits)
        self.channel.configure(choice = 1, awgn = True)
        self.rx_dsp.configure(format=1)


        # Eb/No → SNR
        bits_per_symbol = 2
        Eb_No_dB = np.arange(-6, 11, 1, dtype=float)
        SNR_dB = Eb_No_dB + 10 * np.log10(bits_per_symbol)
        BER_results = []

        for snr_db in SNR_dB:
            
            # Generate random bits
            np.random.seed(0)
            bits = np.random.randint(0, 2,  self._Nbits, dtype=int)

            # Transmit
            _, tx_symbols = self.tx_dsp.generate_signal(bits)

            # Channel
            _, rx_symbols = self.channel.apply_channel(tx_symbols, snr_db=snr_db)

            # Receive
            _, ber_val = self.rx_dsp.process_signal(bits, rx_symbols)
            BER_results.append(ber_val)
            print(f"QPSK SNR = {snr_db:.1f} dB, BER = {ber_val:.6e}")

        # Plot results
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


if __name__ == "__main__":
    orchestrator = SysOrch()
    orchestrator.run()
