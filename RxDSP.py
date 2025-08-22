import numpy as np

class Rx:
    def __init__(self):
        self._reset()

    def _reset(self):
        """Reset all private attributes to default values."""
        self._format = np.int16(0)
        self._ber = np.float32(0.0)
        self._B_hat = np.array([], dtype=np.int32)
        self._a = np.float32(1.0 / np.sqrt(2.0))

    def configure(self):
        # self._write_config()  
        pass


    # def _write_config(self,params):
    #     """Optional placeholder if you want to store config somewhere."""
    #     pass

    def process_signal(self, original_bits, rx, channel_taps):
        """Process received QPSK signal using Gray-coded mapping and compute BER."""
        if len(rx) == 0:
            raise ValueError("Received signal is empty.")
        
        # Normalization
        rx = rx / np.sqrt(np.mean(np.abs(rx)**2))
        # # Function to decode a single symbol according to Gray coding
        
        
        def decode_symbol(sym):
            if sym.real > 0 and sym.imag > 0:
                return [0, 0]
            elif sym.real < 0 and sym.imag > 0:
                return [0, 1]
            elif sym.real < 0 and sym.imag < 0:
                return [1, 1]
            else:  # sym.real > 0 and sym.imag < 0
                return [1, 0]

        # Decode all received symbols
        decoded_bits_list = []
        for sym in rx:
            decoded_bits_list.extend(decode_symbol(sym))

        self._B_hat = np.array(decoded_bits_list, dtype=np.int32)

        # Compute BER
        B_orig = original_bits.reshape(-1, 2).flatten()
        errors = np.sum(np.abs(B_orig - self._B_hat))
        self._ber = np.float32(errors / len(B_orig))

        return self, self._ber
